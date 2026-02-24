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
REDIS_HOST       = os.getenv("REDIS_HOST", "redis-service")
IMAGE_SERVER_URL = os.getenv("IMAGE_SERVER_URL", "http://image-server:8000")

SEG_MODEL  = os.getenv("SEG_MODEL",      "yolo26x-seg.pt")
POSE_MODEL = os.getenv("POSE_MODEL",     "yolo26x-pose.pt")

# Two separate OV model variants:
#   OV_MODEL      — prompt-free (-pf) variant, used for background detection.
#                   Faster, fixed vocabulary, does NOT support set_classes().
#   OV_TEXT_MODEL — text-prompted variant (no -pf suffix), used for
#                   query-directed detection. Supports set_classes().
#
# This split is required because calling set_classes() on the prompt-free
# model raises: "Prompt-free model does not support setting classes."
OV_MODEL      = os.getenv("OV_MODEL",      "yoloe-26x-seg-pf.pt")
OV_TEXT_MODEL = os.getenv("OV_TEXT_MODEL", "yoloe-26x-seg.pt")

CONFIDENCE_THRESHOLD       = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
QUERY_CONFIDENCE_THRESHOLD = float(os.getenv("QUERY_CONFIDENCE_THRESHOLD", "0.2"))
MODEL_POOL_SIZE            = int(os.getenv("MODEL_POOL_SIZE", "1"))

INPUT_STREAM   = "stream:camera:raw"
OUTPUT_STREAM  = "stream:embed:input"
DLQ_STREAM     = "stream:dlq:detector"
CONSUMER_GROUP = "detector_workers"
CONSUMER_NAME  = f"detector_{os.getenv('HOSTNAME', 'unknown')}"

MAX_PENDING_AGE = 300_000  # ms

# ---------- Visual concept extraction ----------
COLOR_NAMES = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "grey", "gray", "cyan", "magenta", "beige", "gold",
    "silver", "teal", "maroon", "navy", "olive", "coral", "turquoise",
    "lavender", "violet", "indigo", "lime", "tan",
]

_COLOR_PATTERN = re.compile(
    r"\b(" + "|".join(COLOR_NAMES) + r")\b[\s\-]*([\w]+(?:\s[\w]+)?)",
    re.IGNORECASE,
)

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
      2. Find standalone color references.
      3. Extract remaining meaningful nouns.
      4. Include full prompt as holistic fallback.

    Examples:
      "How many pink items do you see?" →
        ["pink item", "pink object", "pink", "items", "How many pink items..."]

      "Is there a red cup on the desk?" →
        ["red cup", "red object", "red", "cup", "desk", "Is there a red..."]
    """
    concepts: list[str] = []
    lower = prompt.lower()

    for match in _COLOR_PATTERN.finditer(lower):
        color  = match.group(1).strip()
        noun   = match.group(2).strip()
        phrase = f"{color} {noun}"
        concepts.append(phrase)
        if f"{color} object" not in concepts:
            concepts.append(f"{color} object")
        if color not in concepts:
            concepts.append(color)

    words = re.findall(r"\b[a-z]{3,}\b", lower)
    for word in words:
        if word not in _STOPWORDS and word not in COLOR_NAMES and word not in concepts:
            already_covered = any(word in c for c in concepts)
            if not already_covered:
                concepts.append(word)

    concepts.append(prompt.strip())

    seen: set[str] = set()
    unique: list[str] = []
    for c in concepts:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique[:20]


# ---------- Model pool ----------
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
        """Acquire a model instance, run inference, return to pool."""
        model = await self._queue.get()
        try:
            results = await asyncio.to_thread(model, image, **kwargs)
            return results
        finally:
            await self._queue.put(model)

    async def run_with_classes(self, image: np.ndarray, classes: list[str], **kwargs) -> list:
        """
        Run YOLOE with custom text-prompted classes.

        Only valid on text-prompted model variants (no -pf suffix).
        Holds the model for the full set_classes + inference cycle before
        releasing so the class state cannot be clobbered by a concurrent call.
        """
        model = await self._queue.get()
        try:
            await asyncio.to_thread(model.set_classes, classes)
            results = await asyncio.to_thread(model, image, **kwargs)
            return results
        finally:
            await self._queue.put(model)

    async def get_names(self) -> dict:
        """Peek at model.names without disrupting the pool."""
        model = await self._queue.get()
        names = model.names
        await self._queue.put(model)
        return names


# Pools initialised in main() before the message loop
pool_seg:     ModelPool | None = None
pool_pose:    ModelPool | None = None
pool_ov:      ModelPool | None = None   # prompt-free, background detection
pool_ov_text: ModelPool | None = None   # text-prompted, query-directed detection


# ---------- Spatial helpers ----------
def _position_label(cx_norm: float) -> str:
    if cx_norm < 0.33:
        return "left"
    elif cx_norm < 0.67:
        return "center"
    return "right"


def _depth_label(area_norm: float) -> str:
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
        kp_data   = (
            keypoints.data.cpu().numpy().tolist()
            if keypoints is not None
            else [None] * len(boxes)
        )
        for i in range(len(boxes)):
            if int(classes[i]) != 0:  # persons only
                continue
            conf = confs[i]
            if conf < CONFIDENCE_THRESHOLD:
                continue
            kp = kp_data[i]
            pose_data = (
                [{"x": pt[0], "y": pt[1], "visibility": int(pt[2])} for pt in kp]
                if kp else None
            )
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
    """Rough seated/standing heuristic from COCO keypoints."""
    try:
        nose    = keypoints[0]
        l_hip   = keypoints[11]
        r_hip   = keypoints[12]
        l_ankle = keypoints[15]
        r_ankle = keypoints[16]

        visible = lambda kp: kp[2] > 0.3

        hips_visible   = visible(l_hip) or visible(r_hip)
        ankles_visible = visible(l_ankle) or visible(r_ankle)

        if not hips_visible:
            return "partial"

        hip_y = (
            (l_hip[1] if visible(l_hip) else 0) +
            (r_hip[1] if visible(r_hip) else 0)
        ) / max((1 if visible(l_hip) else 0) + (1 if visible(r_hip) else 0), 1)

        nose_y = nose[1] if visible(nose) else 0

        if ankles_visible:
            ankle_y = (
                (l_ankle[1] if visible(l_ankle) else 0) +
                (r_ankle[1] if visible(r_ankle) else 0)
            ) / max(
                (1 if visible(l_ankle) else 0) + (1 if visible(r_ankle) else 0), 1
            )
            if nose_y > 0 and ankle_y > hip_y:
                torso_to_leg_ratio = (hip_y - nose_y) / max(ankle_y - hip_y, 1)
                if torso_to_leg_ratio > 1.5:
                    return "seated"
                return "standing"

        return "seated"
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
    Marked source='query' so they appear first in the summary and are
    clearly distinguished from background detections.
    Uses a lower confidence threshold since classes are query-specific.
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
    1. Attach pose keypoints/state to matching seg person detections.
    2. Append unmatched pose detections as standalone entries.
    3. Append open-vocab detections.
    4. Prepend query-directed detections — most relevant to the prompt.
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

    return query_dets + seg_dets + ov_dets


# ---------- Summary builder ----------
def build_detection_summary(detections: list[dict], query_concepts: list[str]) -> str:
    """
    Build an enriched detection summary string.

    Query detections are listed first with [Q] prefix so the VLM knows
    they were specifically sought. Low-confidence non-query detections
    are filtered. Duplicate class names are deduplicated (query detections
    always included regardless of duplication).

    Example:
      "[Q] pink item (center, foreground, mid), person (center, foreground, high, seated),
       laptop (left, midground, high) | searched for: "pink item", "pink object", "pink""
    """
    parts: list[str] = []
    seen_classes: set[str] = set()

    for det in detections:
        cls    = det.get("class", "")
        source = det.get("source", "")
        conf   = det.get("confidence", 0.0)
        conf_l = det.get("conf_label", _confidence_label(conf))
        pos    = det.get("position", "")
        depth  = det.get("depth", "")

        if source != "query" and conf_l == "low":
            continue

        label = f"[Q] {cls}" if source == "query" else cls

        attrs = [pos, depth, conf_l]
        if source != "query":
            state = det.get("pose_state", "")
            if state and state not in ("unknown", "partial"):
                attrs.append(state)

        part = f"{label} ({', '.join(a for a in attrs if a)})"

        key = cls.lower()
        if source == "query":
            parts.append(part)
        elif key not in seen_classes:
            seen_classes.add(key)
            parts.append(part)

    summary = ", ".join(parts) if parts else "no detections"

    if query_concepts:
        searched = ", ".join(f'"{c}"' for c in query_concepts[:3])
        summary += f" | searched for: {searched}"

    return summary


# ---------- Redis / HTTP ----------
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

    query_concepts: list[str] = []
    if prompt and prompt.strip():
        query_concepts = extract_visual_concepts(prompt)
        rlog.info(
            f"Extracted {len(query_concepts)} visual concepts: {query_concepts[:5]}"
        )

    rlog.info(f"Running detection on {image_url} (prompt={'yes' if prompt else 'no'})")

    # Run seg, pose, and prompt-free OV concurrently.
    # Query-directed run uses pool_ov_text (text-prompted variant) separately.
    tasks = [
        pool_seg.run(img_array),
        pool_pose.run(img_array),
        pool_ov.run(img_array),
    ]
    if query_concepts:
        tasks.append(pool_ov_text.run_with_classes(img_array, query_concepts))
    else:
        tasks.append(asyncio.sleep(0))  # placeholder to keep index alignment

    results = await asyncio.gather(*tasks, return_exceptions=True)
    results_seg, results_pose, results_ov, results_query = results

    # --- seg ---
    seg_dets: list[dict] = []
    if isinstance(results_seg, Exception):
        rlog.error(f"Segmentation model failed: {results_seg}")
    else:
        seg_names = await pool_seg.get_names()
        seg_dets  = extract_seg_detections(results_seg, img_w, img_h, seg_names)

    # --- pose ---
    pose_dets: list[dict] = []
    if isinstance(results_pose, Exception):
        rlog.error(f"Pose model failed: {results_pose}")
    else:
        pose_dets = extract_pose_detections(results_pose, img_w, img_h)

    # --- prompt-free OV ---
    ov_dets: list[dict] = []
    if isinstance(results_ov, Exception):
        rlog.error(f"Open-vocab model failed: {results_ov}")
    else:
        ov_names = await pool_ov.get_names()
        ov_dets  = extract_ov_detections(results_ov, img_w, img_h, ov_names)

    # --- query-directed OV (text-prompted model) ---
    query_dets: list[dict] = []
    if query_concepts:
        if isinstance(results_query, Exception):
            rlog.error(f"Query-directed model failed: {results_query}")
        else:
            # After run_with_classes the model's .names reflects the custom
            # classes that were set for this run
            q_names    = await pool_ov_text.get_names()
            query_dets = extract_query_detections(results_query, img_w, img_h, q_names)
            rlog.info(
                f"Query detection: {len(query_dets)} matches "
                f"for concepts {query_concepts[:3]}"
            )

    all_dets = merge_detections(seg_dets, pose_dets, ov_dets, query_dets)
    summary  = build_detection_summary(all_dets, query_concepts)

    rlog.info(f"Detection summary: {summary[:120]}...")

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
    global pool_seg, pool_pose, pool_ov, pool_ov_text

    pool_seg      = ModelPool(SEG_MODEL,      MODEL_POOL_SIZE)
    pool_pose     = ModelPool(POSE_MODEL,     MODEL_POOL_SIZE)
    pool_ov       = ModelPool(OV_MODEL,       MODEL_POOL_SIZE)
    pool_ov_text  = ModelPool(OV_TEXT_MODEL,  MODEL_POOL_SIZE)

    await asyncio.gather(
        pool_seg.init(),
        pool_pose.init(),
        pool_ov.init(),
        pool_ov_text.init(),
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
                        rlog.error(
                            f"Processing failed, routing to DLQ: {e}", exc_info=True
                        )
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
