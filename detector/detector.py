import asyncio
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone

import cv2
import httpx
import numpy as np
import redis.asyncio as redis
from redis.exceptions import ResponseError
from ultralytics import YOLO

# ---------- Structured JSON Logging ----------
SERVICE_NAME = "detector"

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "service": SERVICE_NAME,
            "msg":     record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if record.exc_info:
            log_entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logging.basicConfig(handlers=[handler], level=logging.INFO)
log = logging.getLogger(SERVICE_NAME)

# ---------- Configuration ----------
REDIS_HOST        = os.getenv("REDIS_HOST", "redis-service")
IMAGE_SERVER_URL  = os.getenv("IMAGE_SERVER_URL", "http://image-server:8000")

SEG_MODEL         = os.getenv("SEG_MODEL",  "yolo26x-seg.pt")
POSE_MODEL        = os.getenv("POSE_MODEL", "yolo26x-pose.pt")
OV_MODEL          = os.getenv("OV_MODEL",   "yoloe-26x-seg-pf.pt")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))

# YOLOE prompt-directed confidence can be lower since it's class-specific
QUERY_CONFIDENCE_THRESHOLD = float(os.getenv("QUERY_CONFIDENCE_THRESHOLD", "0.2"))

# Number of instances of each model to hold in pool.
# Single-worker setup: 1 is fine. Increase if you run multiple consumers.
MODEL_POOL_SIZE = int(os.getenv("MODEL_POOL_SIZE", "1"))

INPUT_STREAM    = "stream:camera:raw"
OUTPUT_STREAM   = "stream:embed:input"
DLQ_STREAM      = "stream:dlq:detector"
CONSUMER_GROUP  = "detector_workers"
CONSUMER_NAME   = f"detector_{os.getenv('HOSTNAME', 'unknown')}"

MAX_PENDING_AGE = 300_000   # ms — reclaim messages stuck longer than 5 minutes

# ---------- Visual concept extraction ----------
#
# These patterns target the kinds of descriptions a person naturally uses
# when asking a camera system about what it sees. The goal is to produce
# class strings that YOLOE's text encoder can ground visually.
#
COLOR_NAMES = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "grey", "gray", "cyan", "magenta", "beige", "gold",
    "silver", "teal", "maroon", "navy", "olive", "coral", "turquoise",
    "lavender", "violet", "indigo", "lime", "tan",
]

# Patterns that capture "color + noun" or standalone meaningful nouns
_COLOR_PATTERN = re.compile(
    r"\b(" + "|".join(COLOR_NAMES) + r")\b[\s\-]*([\w]+(?:\s[\w]+)?)",
    re.IGNORECASE,
)

# Common interrogative / filler words to strip when extracting noun targets
_STOPWORDS = {
    "how", "many", "much", "what", "where", "when", "why", "which", "who",
    "is", "are", "do", "does", "can", "you", "see", "there", "any", "the",
    "a", "an", "in", "on", "at", "of", "to", "i", "me", "my", "right",
    "now", "currently", "visible", "room", "here", "show", "tell", "find",
    "look", "for", "that", "this", "those", "these", "some", "all",
}


def extract_visual_concepts(prompt: str) -> list[str]:
    """
    Extract visual class strings from a natural language prompt for use
    as YOLOE text-prompted detection classes.

    Strategy:
      1. Find explicit "color + object" phrases — highest signal.
      2. Find standalone color references (user may mean "anything pink").
      3. Extract remaining meaningful nouns as general object targets.
      4. Always include the full prompt as a fallback class so YOLOE can
         interpret the whole query holistically if the above miss.

    Examples:
      "How many pink items do you see?" →
        ["pink item", "pink", "pink object", "pink items"]

      "Is there a red cup on the desk?" →
        ["red cup", "red", "cup", "desk", "red cup on the desk"]

      "Where did I put my keys?" →
        ["keys", "key", "where did I put my keys"]
    """
    concepts: list[str] = []
    lower = prompt.lower()

    # 1. Color + noun pairs
    for match in _COLOR_PATTERN.finditer(lower):
        color  = match.group(1).strip()
        noun   = match.group(2).strip()
        phrase = f"{color} {noun}"
        concepts.append(phrase)
        # Also add generic color + "object" so YOLOE sweeps broadly
        if f"{color} object" not in concepts:
            concepts.append(f"{color} object")
        # Standalone color — catches "anything pink"
        if color not in concepts:
            concepts.append(color)

    # 2. Meaningful standalone nouns (after stripping stopwords + color words)
    words = re.findall(r"\b[a-z]{3,}\b", lower)
    for word in words:
        if word not in _STOPWORDS and word not in COLOR_NAMES and word not in concepts:
            # Skip words already captured as part of a color phrase
            already_covered = any(word in c for c in concepts)
            if not already_covered:
                concepts.append(word)

    # 3. Full prompt as holistic fallback (YOLOE handles natural language well)
    concepts.append(prompt.strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in concepts:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique[:20]   # cap at 20 classes to keep inference fast


# ---------- Model pool ----------
#
# Each model instance is wrapped in an asyncio.Queue so callers acquire
# and release them safely without any locking complexity. If MODEL_POOL_SIZE
# is 1 (default), behaviour is identical to the original single-model setup
# but is now actually safe under concurrent access.
#

class ModelPool:
    def __init__(self, model_path: str, pool_size: int = 1):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._path  = model_path
        self._size  = pool_size

    async def init(self):
        log.info(f"Loading {self._size}x {self._path}")
        for _ in range(self._size):
            model = await asyncio.to_thread(YOLO, self._path)
            await self._queue.put(model)
        log.info(f"Model pool ready: {self._path}")

    async def run(self, image: np.ndarray, **kwargs) -> list:
        """Acquire a model, run inference, release back to pool."""
        model = await self._queue.get()
        try:
            results = await asyncio.to_thread(model, image, **kwargs)
            return results
        finally:
            await self._queue.put(model)

    async def run_with_classes(self, image: np.ndarray, classes: list[str], **kwargs) -> list:
        """
        Run YOLOE with custom text-prompted classes.
        set_classes mutates the model, so we hold it for the full duration
        of class-setting + inference before releasing.
        """
        model = await self._queue.get()
        try:
            await asyncio.to_thread(model.set_classes, classes)
            results = await asyncio.to_thread(model, image, **kwargs)
            return results
        finally:
            await self._queue.put(model)


# Pools initialised in main() before the message loop
pool_seg:  ModelPool | None = None
pool_pose: ModelPool | None = None
pool_ov:   ModelPool | None = None


# ---------- Spatial helpers ----------

def _position_label(cx_norm: float) -> str:
    """Horizontal position from normalised x centre [0, 1]."""
    if cx_norm < 0.33:
        return "left"
    elif cx_norm < 0.67:
        return "center"
    return "right"


def _depth_label(area_norm: float) -> str:
    """Rough depth estimate from normalised bounding-box area [0, 1]."""
    if area_norm > 0.15:
        return "foreground"
    elif area_norm > 0.04:
        return "midground"
    return "background"


def _confidence_label(conf: float) -> str:
    if conf >= 0.70:
        return "high"
    elif conf >= 0.40:
        return "mid"
    return "low"


def _bbox_to_spatial(bbox: list[float], img_w: int, img_h: int) -> tuple[str, str]:
    x1, y1, x2, y2 = bbox
    cx_norm   = ((x1 + x2) / 2) / img_w
    area_norm = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
    return _position_label(cx_norm), _depth_label(area_norm)


# ---------- Detection extractors ----------

def extract_seg_detections(results, img_w: int, img_h: int, model_names: dict) -> list[dict]:
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        boxes   = r.boxes.xyxy.cpu().numpy().tolist()
        confs   = r.boxes.conf.cpu().numpy().tolist()
        classes = r.boxes.cls.cpu().numpy().tolist()
        for i in range(len(boxes)):
            conf = confs[i]
            if conf < CONFIDENCE_THRESHOLD:
                continue
            pos, depth = _bbox_to_spatial(boxes[i], img_w, img_h)
            dets.append({
                "source":     "seg",
                "class":      model_names[int(classes[i])],
                "confidence": conf,
                "conf_label": _confidence_label(conf),
                "position":   pos,
                "depth":      depth,
                "bbox":       boxes[i],
                "pose":       None,
            })
    return dets


def extract_pose_detections(results, img_w: int, img_h: int) -> list[dict]:
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        boxes     = r.boxes.xyxy.cpu().numpy().tolist()
        confs     = r.boxes.conf.cpu().numpy().tolist()
        classes   = r.boxes.cls.cpu().numpy().tolist()
        keypoints = r.keypoints
        kp_data   = keypoints.data.cpu().numpy().tolist() if keypoints is not None else [None] * len(boxes)

        for i in range(len(boxes)):
            if int(classes[i]) != 0:   # persons only
                continue
            conf = confs[i]
            if conf < CONFIDENCE_THRESHOLD:
                continue
            kp = kp_data[i]
            pose_data = (
                [{"x": pt[0], "y": pt[1], "visibility": int(pt[2])} for pt in kp]
                if kp else None
            )
            # Infer rough pose state from keypoints
            pose_state = _infer_pose_state(kp) if kp else "unknown"
            pos, depth = _bbox_to_spatial(boxes[i], img_w, img_h)
            dets.append({
                "bbox":       boxes[i],
                "confidence": conf,
                "conf_label": _confidence_label(conf),
                "position":   pos,
                "depth":      depth,
                "keypoints":  pose_data,
                "pose_state": pose_state,
            })
    return dets


def _infer_pose_state(keypoints: list) -> str:
    """
    Rough seated / standing / lying heuristic from COCO keypoints.
    Keypoints: 0=nose, 5/6=shoulders, 11/12=hips, 15/16=ankles
    """
    try:
        # y increases downward in image coordinates
        nose      = keypoints[0]
        l_hip     = keypoints[11]
        r_hip     = keypoints[12]
        l_ankle   = keypoints[15]
        r_ankle   = keypoints[16]

        visible = lambda kp: kp[2] > 0.3   # visibility threshold

        hips_visible   = visible(l_hip)   or visible(r_hip)
        ankles_visible = visible(l_ankle) or visible(r_ankle)

        if not hips_visible:
            return "partial"

        hip_y   = ((l_hip[1] if visible(l_hip) else 0) + (r_hip[1] if visible(r_hip) else 0)) / max(
            (1 if visible(l_hip) else 0) + (1 if visible(r_hip) else 0), 1
        )
        nose_y  = nose[1] if visible(nose) else 0

        if ankles_visible:
            ankle_y = (
                (l_ankle[1] if visible(l_ankle) else 0) +
                (r_ankle[1] if visible(r_ankle) else 0)
            ) / max(
                (1 if visible(l_ankle) else 0) + (1 if visible(r_ankle) else 0), 1
            )
            # If vertical span of nose→ankle is small relative to hip→ankle, likely seated
            if nose_y > 0 and ankle_y > hip_y:
                torso_to_leg_ratio = (hip_y - nose_y) / max(ankle_y - hip_y, 1)
                if torso_to_leg_ratio > 1.5:
                    return "seated"
                return "standing"

        return "seated"   # ankles not visible, likely sitting or cropped
    except (IndexError, ZeroDivisionError):
        return "unknown"


def extract_ov_detections(results, img_w: int, img_h: int, model_names: dict) -> list[dict]:
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        boxes   = r.boxes.xyxy.cpu().numpy().tolist()
        confs   = r.boxes.conf.cpu().numpy().tolist()
        classes = r.boxes.cls.cpu().numpy().tolist()
        for i in range(len(boxes)):
            conf = confs[i]
            if conf < CONFIDENCE_THRESHOLD:
                continue
            pos, depth = _bbox_to_spatial(boxes[i], img_w, img_h)
            dets.append({
                "source":     "openvocab",
                "class":      model_names[int(classes[i])],
                "confidence": conf,
                "conf_label": _confidence_label(conf),
                "position":   pos,
                "depth":      depth,
                "bbox":       boxes[i],
                "pose":       None,
            })
    return dets


def extract_query_detections(results, img_w: int, img_h: int, model_names: dict) -> list[dict]:
    """
    Extract detections from a query-directed YOLOE inference.
    Marked with source='query' so downstream can distinguish them.
    Uses a lower confidence threshold since the classes are query-specific.
    """
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        boxes   = r.boxes.xyxy.cpu().numpy().tolist()
        confs   = r.boxes.conf.cpu().numpy().tolist()
        classes = r.boxes.cls.cpu().numpy().tolist()
        for i in range(len(boxes)):
            conf = confs[i]
            if conf < QUERY_CONFIDENCE_THRESHOLD:
                continue
            pos, depth = _bbox_to_spatial(boxes[i], img_w, img_h)
            dets.append({
                "source":     "query",
                "class":      model_names[int(classes[i])],
                "confidence": conf,
                "conf_label": _confidence_label(conf),
                "position":   pos,
                "depth":      depth,
                "bbox":       boxes[i],
                "pose":       None,
            })
    return dets


# ---------- IoU ----------

def iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ---------- Merge ----------

def merge_detections(
    seg_dets: list[dict],
    pose_dets: list[dict],
    ov_dets: list[dict],
    query_dets: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """
    1. Attach pose keypoints / state to matching seg person detections.
    2. Append unmatched pose detections as standalone entries.
    3. Append open-vocab detections.
    4. Prepend query-directed detections at the front so they appear first
       in the summary — they are most relevant to what the user asked.
    """
    seg_person_indices = [i for i, d in enumerate(seg_dets) if d["class"] == "person"]

    for p in pose_dets:
        best_iou, best_idx = 0.0, -1
        for idx in seg_person_indices:
            val = iou(p["bbox"], seg_dets[idx]["bbox"])
            if val > best_iou:
                best_iou, best_idx = val, idx
        if best_iou >= iou_threshold:
            seg_dets[best_idx]["pose"]       = p["keypoints"]
            seg_dets[best_idx]["pose_state"] = p.get("pose_state", "unknown")
        else:
            seg_dets.append({
                "source":     "pose",
                "class":      "person",
                "confidence": p["confidence"],
                "conf_label": p["conf_label"],
                "position":   p["position"],
                "depth":      p["depth"],
                "bbox":       p["bbox"],
                "pose":       p["keypoints"],
                "pose_state": p.get("pose_state", "unknown"),
            })

    # Query detections go first — most semantically relevant to the prompt
    return query_dets + seg_dets + ov_dets


# ---------- Summary builder ----------

def build_detection_summary(detections: list[dict], query_concepts: list[str]) -> str:
    """
    Build an enriched detection summary string.

    Format: "class (position, depth, confidence), ..."

    Query-directed detections are listed first and prefixed with [Q] so the
    VLM and embedding model know these were specifically sought. This gives
    the language model a strong prior that these detections are relevant to
    the user's question.

    Low-confidence detections are filtered unless they came from a query run
    (where lower confidence is expected for rare/specific classes).

    Example output:
      "[Q] pink item (center, foreground, mid), [Q] pink object (left, midground, low),
       person (center, foreground, high, seated), laptop (left, midground, high),
       chair (right, background, mid)"
    """
    parts   = []
    seen_classes: set[str] = set()

    for det in detections:
        cls    = det.get("class", "")
        source = det.get("source", "")
        conf   = det.get("confidence", 0.0)
        conf_l = det.get("conf_label", _confidence_label(conf))
        pos    = det.get("position", "")
        depth  = det.get("depth",    "")

        # Filter low-confidence non-query detections
        if source != "query" and conf_l == "low":
            continue

        label = f"[Q] {cls}" if source == "query" else cls

        attrs = [pos, depth, conf_l]
        if source != "query" and det.get("pose_state") and det["pose_state"] not in ("unknown", "partial"):
            attrs.append(det["pose_state"])

        part = f"{label} ({', '.join(a for a in attrs if a)})"

        # Deduplicate — keep first occurrence (query detections are already first)
        key = cls.lower()
        if source == "query":
            # Always include query detections even if class name was seen before
            parts.append(part)
        elif key not in seen_classes:
            seen_classes.add(key)
            parts.append(part)

    summary = ", ".join(parts) if parts else "no detections"

    # Append a note about what was specifically searched for — helps VLM context
    if query_concepts:
        # Show only the first 3 most specific concepts to keep it readable
        searched = ", ".join(f'"{c}"' for c in query_concepts[:3])
        summary += f" | searched for: {searched}"

    return summary


# ---------- Redis setup ----------
r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
http_client = httpx.AsyncClient(timeout=30.0)


async def ensure_consumer_group():
    try:
        await r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


async def recover_pending_messages():
    try:
        pending = await r.xpending(INPUT_STREAM, CONSUMER_GROUP)
        count   = pending.get("pending", 0)
        if count == 0:
            return
        log.info(f"Found {count} pending messages — checking for stale entries")
        claimed = await r.xautoclaim(
            INPUT_STREAM, CONSUMER_GROUP, CONSUMER_NAME,
            min_idle_time=MAX_PENDING_AGE, start_id="0-0", count=100,
        )
        messages = claimed[1] if isinstance(claimed, (list, tuple)) else []
        for mid, fields in messages:
            log.warning(f"Moving stale message {mid} to DLQ")
            await r.xadd(DLQ_STREAM, {"original_id": mid, "data": fields.get("data", "")})
            await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)
        if messages:
            log.info(f"Recovered {len(messages)} stale messages → DLQ")
    except Exception as e:
        log.warning(f"Pending recovery failed (non-fatal): {e}")


# ---------- Core processing ----------

async def process_one_message(data: dict):
    filepath   = data.get("filepath")
    timestamp  = data.get("timestamp")
    request_id = data.get("id") or str(uuid.uuid4())
    prompt     = data.get("prompt", "")

    rlog = logging.LoggerAdapter(log, {"request_id": request_id})

    if not filepath or not timestamp:
        rlog.warning("Missing filepath or timestamp — skipping")
        return

    filename  = os.path.basename(filepath)
    image_url = f"{IMAGE_SERVER_URL}/{filename}"

    # Download image
    try:
        resp = await http_client.get(image_url)
        resp.raise_for_status()
        img_bytes = resp.content
    except Exception as e:
        rlog.error(f"Image download failed: {e}")
        return

    img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        rlog.error(f"Failed to decode image: {image_url}")
        return

    img_h, img_w = img_array.shape[:2]

    # ---------- Extract visual concepts from prompt (if present) ----------
    query_concepts: list[str] = []
    if prompt and prompt.strip():
        query_concepts = extract_visual_concepts(prompt)
        rlog.info(f"Extracted {len(query_concepts)} visual concepts from prompt: {query_concepts[:5]}")

    # ---------- Run detection models concurrently ----------
    rlog.info(f"Running detection on {image_url} (prompt={'yes' if prompt else 'no'})")

    tasks = [
        pool_seg.run(img_array),
        pool_pose.run(img_array),
        pool_ov.run(img_array),
    ]

    # Add query-directed YOLOE run only when there's a prompt
    if query_concepts:
        tasks.append(pool_ov.run_with_classes(img_array, query_concepts))
    else:
        tasks.append(asyncio.sleep(0))   # placeholder to keep index alignment

    results = await asyncio.gather(*tasks, return_exceptions=True)
    results_seg, results_pose, results_ov, results_query = results

    seg_dets: list[dict] = []
    if isinstance(results_seg, Exception):
        rlog.error(f"Segmentation model failed: {results_seg}")
    else:
        # Need model names — grab from pool (model is back in pool by now, safe to read names)
        seg_model_temp = await pool_seg._queue.get()
        seg_names = seg_model_temp.names
        await pool_seg._queue.put(seg_model_temp)
        seg_dets = extract_seg_detections(results_seg, img_w, img_h, seg_names)

    pose_dets: list[dict] = []
    if isinstance(results_pose, Exception):
        rlog.error(f"Pose model failed: {results_pose}")
    else:
        pose_dets = extract_pose_detections(results_pose, img_w, img_h)

    ov_dets: list[dict] = []
    if isinstance(results_ov, Exception):
        rlog.error(f"Open-vocab model failed: {results_ov}")
    else:
        ov_model_temp = await pool_ov._queue.get()
        ov_names = ov_model_temp.names
        await pool_ov._queue.put(ov_model_temp)
        ov_dets = extract_ov_detections(results_ov, img_w, img_h, ov_names)

    query_dets: list[dict] = []
    if query_concepts:
        if isinstance(results_query, Exception):
            rlog.error(f"Query-directed model failed: {results_query}")
        else:
            # After run_with_classes, names reflect the custom classes set
            q_model_temp = await pool_ov._queue.get()
            q_names = q_model_temp.names
            await pool_ov._queue.put(q_model_temp)
            query_dets = extract_query_detections(results_query, img_w, img_h, q_names)
            rlog.info(f"Query detection found {len(query_dets)} matches for concepts {query_concepts[:3]}")

    # ---------- Merge ----------
    all_dets = merge_detections(seg_dets, pose_dets, ov_dets, query_dets)
    summary  = build_detection_summary(all_dets, query_concepts)

    rlog.info(f"Detection summary: {summary[:120]}...")

    # ---------- Publish ----------
    out_msg = {
        "id":                request_id,
        "filepath":          filepath,
        "timestamp":         timestamp,
        "prompt":            prompt,
        "detection_json":    json.dumps(all_dets),
        "detection_summary": summary,
    }
    await r.xadd(OUTPUT_STREAM, {"data": json.dumps(out_msg)})
    rlog.info(f"Published enriched message for {filename}")


# ---------- Main ----------

async def main():
    global pool_seg, pool_pose, pool_ov

    # Initialise model pools before starting the message loop
    pool_seg  = ModelPool(SEG_MODEL,  MODEL_POOL_SIZE)
    pool_pose = ModelPool(POSE_MODEL, MODEL_POOL_SIZE)
    pool_ov   = ModelPool(OV_MODEL,   MODEL_POOL_SIZE)

    await asyncio.gather(
        pool_seg.init(),
        pool_pose.init(),
        pool_ov.init(),
    )

    await ensure_consumer_group()
    await recover_pending_messages()
    log.info(f"Detector listening on '{INPUT_STREAM}' (consumer={CONSUMER_NAME})")

    while True:
        try:
            messages = await r.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={INPUT_STREAM: ">"},
                count=1,
                block=2000,
            )
            if not messages:
                continue

            for stream, msg_list in messages:
                for msg_id, fields in msg_list:
                    data = json.loads(fields["data"])
                    try:
                        await process_one_message(data)
                        await r.xack(INPUT_STREAM, CONSUMER_GROUP, msg_id)
                    except Exception as e:
                        request_id = data.get("id", "unknown")
                        rlog = logging.LoggerAdapter(log, {"request_id": request_id})
                        rlog.error(f"Processing failed, routing to DLQ: {e}", exc_info=True)
                        await r.xadd(
                            DLQ_STREAM,
                            {"original_id": msg_id, "error": str(e), "data": fields["data"]},
                        )
                        await r.xack(INPUT_STREAM, CONSUMER_GROUP, msg_id)

        except Exception as e:
            log.error(f"Outer loop error: {e}", exc_info=True)
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(http_client.aclose())
