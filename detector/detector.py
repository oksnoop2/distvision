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
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# ---------- Structured JSON Logging ----------
SERVICE_NAME = "detector"

class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "service": SERVICE_NAME,
            "msg":     record.getMessage(),
        }
        if hasattr(record, "request_id"):
            entry["request_id"] = record.request_id
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logging.basicConfig(handlers=[handler], level=logging.INFO)
log = logging.getLogger(SERVICE_NAME)

# ---------- Configuration ----------
REDIS_HOST       = os.getenv("REDIS_HOST", "redis-service")
IMAGE_SERVER_URL = os.getenv("IMAGE_SERVER_URL", "http://image-server:8000")

SEG_MODEL     = os.getenv("SEG_MODEL",      "yolo26x-seg.pt")
POSE_MODEL    = os.getenv("POSE_MODEL",     "yolo26x-pose.pt")
OV_MODEL      = os.getenv("OV_MODEL",       "yoloe-26x-seg-pf.pt")  # prompt-free
OV_TEXT_MODEL = os.getenv("OV_TEXT_MODEL",  "yoloe-26x-seg.pt")     # text + visual prompted

CONFIDENCE_THRESHOLD       = float(os.getenv("CONFIDENCE_THRESHOLD",       "0.3"))
QUERY_CONFIDENCE_THRESHOLD = float(os.getenv("QUERY_CONFIDENCE_THRESHOLD", "0.2"))
VP_CONFIDENCE_THRESHOLD    = float(os.getenv("VP_CONFIDENCE_THRESHOLD",    "0.25"))
MODEL_POOL_SIZE            = int(os.getenv("MODEL_POOL_SIZE", "1"))

# Streams
#   CAPTURE_STREAM  — background captures and user query captures (camera_agent → detector)
#   VP_STREAM       — visual prompt verification jobs dispatched by embed
#                     for temporal queries; result goes back to a per-request reply stream
CAPTURE_STREAM = "stream:camera:raw"
VP_STREAM      = "stream:detector:vp"
OUTPUT_STREAM  = "stream:embed:input"
DLQ_STREAM     = "stream:dlq:detector"
CONSUMER_GROUP = "detector_workers"
CONSUMER_NAME  = f"detector_{os.getenv('HOSTNAME', 'unknown')}"

MAX_VP_BBOXES   = 8        # max past-frame bboxes to use as visual prompts
MAX_PENDING_AGE = 300_000  # ms before a stale pending message is DLQ'd


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

_TRAILING_VERBS = {"are", "is", "was", "were", "be", "been", "have", "has"}


def extract_visual_concepts(prompt: str) -> list[str]:
    """
    Extract YOLOE text-prompt class strings from a natural language query.

    Priority:
      1. "color + noun" phrases  →  "pink blanket", "pink object", "pink"
      2. Standalone meaningful nouns
      3. Full prompt as holistic fallback

    Trailing auxiliary verbs are stripped from the captured noun so
    "pink things are" becomes "pink things".
    """
    concepts: list[str] = []
    lower = prompt.lower()

    for match in _COLOR_PATTERN.finditer(lower):
        color = match.group(1).strip()
        noun  = match.group(2).strip()
        noun_words = [w for w in noun.split() if w not in _TRAILING_VERBS]
        noun = " ".join(noun_words).strip()
        if not noun:
            continue
        phrase = f"{color} {noun}"
        concepts.append(phrase)
        if f"{color} object" not in concepts:
            concepts.append(f"{color} object")
        if color not in concepts:
            concepts.append(color)

    words = re.findall(r"\b[a-z]{3,}\b", lower)
    for word in words:
        if word not in _STOPWORDS and word not in COLOR_NAMES and word not in concepts:
            if not any(word in c for c in concepts):
                concepts.append(word)

    concepts.append(prompt.strip())

    seen:   set[str]  = set()
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
        model = await self._queue.get()
        try:
            return await asyncio.to_thread(model, image, **kwargs)
        finally:
            await self._queue.put(model)

    async def run_with_classes(self, image: np.ndarray, classes: list[str], **kwargs) -> list:
        """
        Text-prompted YOLOE inference.
        Holds the model for the full set_classes + inference cycle so class
        state cannot be clobbered by a concurrent call.
        """
        model = await self._queue.get()
        try:
            await asyncio.to_thread(model.set_classes, classes)
            return await asyncio.to_thread(model, image, **kwargs)
        finally:
            await self._queue.put(model)

    async def run_with_visual_prompts(
        self,
        current_image: np.ndarray,
        refer_image:   np.ndarray,
        bboxes:        np.ndarray,   # (N, 4) xyxy from past frame
        cls_ids:       np.ndarray,   # (N,)   sequential ints starting at 0
    ) -> list:
        """
        Visual-prompted YOLOE inference (SAVPE module).

        refer_image + bboxes define what to look for — the model finds
        visually similar objects in current_image.

        cls_ids must be sequential starting from 0 (YOLOE requirement).
        Map them back to human class names via past_class_map externally.
        """
        model = await self._queue.get()
        try:
            visual_prompts = {"bboxes": bboxes, "cls": cls_ids}

            def _predict():
                return model.predict(
                    source=current_image,
                    refer_image=refer_image,
                    visual_prompts=visual_prompts,
                    predictor=YOLOEVPSegPredictor,
                    verbose=False,
                )

            return await asyncio.to_thread(_predict)
        finally:
            await self._queue.put(model)

    async def get_names(self) -> dict:
        model = await self._queue.get()
        names = model.names
        await self._queue.put(model)
        return names


# Initialised in main()
pool_seg:     ModelPool | None = None
pool_pose:    ModelPool | None = None
pool_ov:      ModelPool | None = None   # prompt-free background sweep
pool_ov_text: ModelPool | None = None   # text + visual prompted


# ---------- Spatial helpers ----------
def _position_label(cx_norm: float) -> str:
    if cx_norm < 0.33:    return "left"
    elif cx_norm < 0.67:  return "center"
    return "right"

def _depth_label(area_norm: float) -> str:
    if area_norm > 0.15:   return "foreground"
    elif area_norm > 0.04: return "midground"
    return "background"

def _confidence_label(conf: float) -> str:
    if conf >= 0.70:   return "high"
    elif conf >= 0.40: return "mid"
    return "low"

def _bbox_to_spatial(bbox: list[float], img_w: int, img_h: int) -> tuple[str, str]:
    x1, y1, x2, y2 = bbox
    cx_norm   = ((x1 + x2) / 2) / img_w
    area_norm = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
    return _position_label(cx_norm), _depth_label(area_norm)


# ---------- Detection extractors ----------
def extract_seg_detections(results, img_w, img_h, model_names) -> list[dict]:
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for bbox, conf, cls in zip(
            r.boxes.xyxy.cpu().numpy().tolist(),
            r.boxes.conf.cpu().numpy().tolist(),
            r.boxes.cls.cpu().numpy().tolist(),
        ):
            if conf < CONFIDENCE_THRESHOLD:
                continue
            pos, depth = _bbox_to_spatial(bbox, img_w, img_h)
            dets.append({
                "source":     "seg",
                "class":      model_names[int(cls)],
                "confidence": conf,
                "conf_label": _confidence_label(conf),
                "position":   pos,
                "depth":      depth,
                "bbox":       bbox,
                "pose":       None,
            })
    return dets


def extract_pose_detections(results, img_w, img_h) -> list[dict]:
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        keypoints = r.keypoints
        kp_data = (
            keypoints.data.cpu().numpy().tolist()
            if keypoints is not None
            else [None] * len(r.boxes)
        )
        for i, (bbox, conf, cls) in enumerate(zip(
            r.boxes.xyxy.cpu().numpy().tolist(),
            r.boxes.conf.cpu().numpy().tolist(),
            r.boxes.cls.cpu().numpy().tolist(),
        )):
            if int(cls) != 0 or conf < CONFIDENCE_THRESHOLD:
                continue
            kp = kp_data[i]
            pose_data  = (
                [{"x": pt[0], "y": pt[1], "visibility": int(pt[2])} for pt in kp]
                if kp else None
            )
            pose_state = _infer_pose_state(kp) if kp else "unknown"
            pos, depth = _bbox_to_spatial(bbox, img_w, img_h)
            dets.append({
                "bbox":       bbox,
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
    Seated/standing heuristic from COCO keypoints.

    Requires nose, hips, AND ankles all visible above threshold before
    committing to a label. Returns "partial" otherwise so unreliable pose
    labels don't pollute the detection summary.
    """
    try:
        nose    = keypoints[0]
        l_hip   = keypoints[11]
        r_hip   = keypoints[12]
        l_ankle = keypoints[15]
        r_ankle = keypoints[16]

        visible = lambda kp: kp[2] > 0.3

        if not (
            (visible(l_hip) or visible(r_hip)) and
            (visible(l_ankle) or visible(r_ankle)) and
            visible(nose)
        ):
            return "partial"

        hip_y = (
            (l_hip[1] if visible(l_hip) else 0) +
            (r_hip[1] if visible(r_hip) else 0)
        ) / max(int(visible(l_hip)) + int(visible(r_hip)), 1)

        ankle_y = (
            (l_ankle[1] if visible(l_ankle) else 0) +
            (r_ankle[1] if visible(r_ankle) else 0)
        ) / max(int(visible(l_ankle)) + int(visible(r_ankle)), 1)

        torso_to_leg = (hip_y - nose[1]) / max(ankle_y - hip_y, 1)
        return "seated" if torso_to_leg > 1.5 else "standing"

    except (IndexError, ZeroDivisionError):
        return "unknown"


def extract_ov_detections(results, img_w, img_h, model_names) -> list[dict]:
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for bbox, conf, cls in zip(
            r.boxes.xyxy.cpu().numpy().tolist(),
            r.boxes.conf.cpu().numpy().tolist(),
            r.boxes.cls.cpu().numpy().tolist(),
        ):
            if conf < CONFIDENCE_THRESHOLD:
                continue
            pos, depth = _bbox_to_spatial(bbox, img_w, img_h)
            dets.append({
                "source":     "openvocab",
                "class":      model_names[int(cls)],
                "confidence": conf,
                "conf_label": _confidence_label(conf),
                "position":   pos,
                "depth":      depth,
                "bbox":       bbox,
                "pose":       None,
            })
    return dets


def extract_query_detections(results, img_w, img_h, model_names) -> list[dict]:
    """Text-prompted detections. Tagged source='query' → [Q] in summary."""
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for bbox, conf, cls in zip(
            r.boxes.xyxy.cpu().numpy().tolist(),
            r.boxes.conf.cpu().numpy().tolist(),
            r.boxes.cls.cpu().numpy().tolist(),
        ):
            if conf < QUERY_CONFIDENCE_THRESHOLD:
                continue
            pos, depth = _bbox_to_spatial(bbox, img_w, img_h)
            dets.append({
                "source":     "query",
                "class":      model_names[int(cls)],
                "confidence": conf,
                "conf_label": _confidence_label(conf),
                "position":   pos,
                "depth":      depth,
                "bbox":       bbox,
                "pose":       None,
            })
    return dets


def extract_vp_detections(
    results,
    img_w: int,
    img_h: int,
    past_class_map: dict[int, str],
) -> list[dict]:
    """
    Visual-prompt detections. Tagged source='visual_prompt' → [VP] in summary.

    past_class_map maps sequential cls_id (0, 1, 2...) back to the human-
    readable class name from the past frame so the summary says
    "[VP] pink blanket" rather than "[VP] 0".
    """
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for bbox, conf, cls in zip(
            r.boxes.xyxy.cpu().numpy().tolist(),
            r.boxes.conf.cpu().numpy().tolist(),
            r.boxes.cls.cpu().numpy().tolist(),
        ):
            if conf < VP_CONFIDENCE_THRESHOLD:
                continue
            cls_id     = int(cls)
            cls_name   = past_class_map.get(cls_id, f"object_{cls_id}")
            pos, depth = _bbox_to_spatial(bbox, img_w, img_h)
            dets.append({
                "source":     "visual_prompt",
                "class":      cls_name,
                "confidence": conf,
                "conf_label": _confidence_label(conf),
                "position":   pos,
                "depth":      depth,
                "bbox":       bbox,
                "pose":       None,
            })
    return dets


# ---------- IoU ----------
def iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ---------- Merge ----------
def merge_detections(
    seg_dets:   list[dict],
    pose_dets:  list[dict],
    ov_dets:    list[dict],
    query_dets: list[dict],
    vp_dets:    list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """
    Fuse pose onto matching seg person boxes, then order:
      [VP] verified past objects first — concrete temporal evidence
      [Q]  query-directed detections second
      seg + ov background sweep last
    """
    seg_person_idx = [i for i, d in enumerate(seg_dets) if d["class"] == "person"]

    for p in pose_dets:
        best_iou, best_idx = 0.0, -1
        for idx in seg_person_idx:
            v = iou(p["bbox"], seg_dets[idx]["bbox"])
            if v > best_iou:
                best_iou, best_idx = v, idx
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

    return vp_dets + query_dets + seg_dets + ov_dets


# ---------- Summary builder ----------
def build_detection_summary(
    detections:    list[dict],
    query_concepts: list[str],
) -> str:
    """
    [VP] verified past-object detections come first.
    [Q]  query-directed detections come second.
    Low-confidence background detections are filtered.
    Class names are deduplicated for background detections only.
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

        is_special = source in ("query", "visual_prompt")

        if not is_special and conf_l == "low":
            continue

        if source == "visual_prompt":
            label = f"[VP] {cls}"
        elif source == "query":
            label = f"[Q] {cls}"
        else:
            label = cls

        attrs = [pos, depth, conf_l]
        if not is_special:
            state = det.get("pose_state", "")
            if state and state not in ("unknown", "partial"):
                attrs.append(state)

        part = f"{label} ({', '.join(a for a in attrs if a)})"

        key = cls.lower()
        if is_special:
            parts.append(part)
        elif key not in seen_classes:
            seen_classes.add(key)
            parts.append(part)

    summary = ", ".join(parts) if parts else "no detections"

    if query_concepts:
        searched = ", ".join(f'"{c}"' for c in query_concepts[:3])
        summary += f" | searched for: {searched}"

    return summary


# ---------- Image fetch ----------
async def fetch_image(url: str) -> np.ndarray | None:
    try:
        resp = await http_client.get(url)
        resp.raise_for_status()
        return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        log.warning(f"fetch_image failed for {url}: {e}")
        return None


# ---------- Job handlers ----------
async def handle_capture_job(data: dict):
    """
    Background capture or user query capture.

    Always runs: seg + pose + prompt-free OV.
    Also runs:   text-prompted OV when a user prompt is present.
    Publishes to stream:embed:input.
    """
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

    img_array = await fetch_image(image_url)
    if img_array is None:
        rlog.error(f"Could not fetch image: {image_url}")
        return

    img_h, img_w = img_array.shape[:2]

    query_concepts: list[str] = []
    if prompt and prompt.strip():
        query_concepts = extract_visual_concepts(prompt)
        rlog.info(f"Extracted {len(query_concepts)} concepts: {query_concepts[:5]}")

    rlog.info(f"Capture job: {filename} prompt={'yes' if prompt else 'no'}")

    r_seg, r_pose, r_ov, r_query = await asyncio.gather(
        pool_seg.run(img_array),
        pool_pose.run(img_array),
        pool_ov.run(img_array),
        pool_ov_text.run_with_classes(img_array, query_concepts)
        if query_concepts else asyncio.sleep(0),
        return_exceptions=True,
    )

    seg_dets: list[dict] = []
    if isinstance(r_seg, Exception):
        rlog.error(f"Seg failed: {r_seg}")
    else:
        seg_dets = extract_seg_detections(r_seg, img_w, img_h, await pool_seg.get_names())

    pose_dets: list[dict] = []
    if isinstance(r_pose, Exception):
        rlog.error(f"Pose failed: {r_pose}")
    else:
        pose_dets = extract_pose_detections(r_pose, img_w, img_h)

    ov_dets: list[dict] = []
    if isinstance(r_ov, Exception):
        rlog.error(f"OV failed: {r_ov}")
    else:
        ov_dets = extract_ov_detections(r_ov, img_w, img_h, await pool_ov.get_names())

    query_dets: list[dict] = []
    if query_concepts:
        if isinstance(r_query, Exception):
            rlog.error(f"Query-directed failed: {r_query}")
        else:
            query_dets = extract_query_detections(
                r_query, img_w, img_h, await pool_ov_text.get_names()
            )
            rlog.info(f"Query detections: {len(query_dets)} for {query_concepts[:3]}")

    all_dets = merge_detections(seg_dets, pose_dets, ov_dets, query_dets, vp_dets=[])
    summary  = build_detection_summary(all_dets, query_concepts)
    rlog.info(f"Summary: {summary[:120]}...")

    await r.xadd(OUTPUT_STREAM, {"data": json.dumps({
        "id":                request_id,
        "filepath":          filepath,
        "timestamp":         timestamp,
        "prompt":            prompt,
        "detection_json":    json.dumps(all_dets),
        "detection_summary": summary,
    })})
    rlog.info(f"→ embed ({filename})")


async def handle_vp_job(data: dict):
    """
    Visual prompt verification job dispatched by embed for temporal queries.

    Embed retrieves a relevant past frame and its stored detection bboxes,
    then asks detector: are these objects still present in the current frame?

    Message fields:
        id                  — original request_id (for correlation)
        current_image_url   — current frame to run inference on
        past_image_url      — reference frame (bboxes originate here)
        past_detection_json — JSON list of detection dicts from past frame
        reply_to            — stream key for the result
                              e.g. "stream:detector:reply:{request_id}"

    Result published to reply_to:
        id                    — request_id
        vp_detection_json     — JSON list of VP detection dicts
        vp_detection_summary  — summary string with [VP] prefixes
        elapsed_ms            — wall time for this VP pass
    """
    request_id    = data.get("id", str(uuid.uuid4()))
    current_url   = data.get("current_image_url")
    past_url      = data.get("past_image_url")
    past_det_json = data.get("past_detection_json", "[]")
    reply_to      = data.get("reply_to")
    rlog = logging.LoggerAdapter(log, {"request_id": request_id})

    if not current_url or not past_url or not reply_to:
        rlog.warning("VP job missing required fields — skipping")
        return

    t0 = asyncio.get_event_loop().time()

    current_arr, past_arr = await asyncio.gather(
        fetch_image(current_url),
        fetch_image(past_url),
        return_exceptions=True,
    )

    if isinstance(current_arr, Exception) or current_arr is None:
        rlog.error(f"Could not fetch current image: {current_url}")
        return
    if isinstance(past_arr, Exception) or past_arr is None:
        rlog.error(f"Could not fetch past image: {past_url}")
        return

    img_h, img_w = current_arr.shape[:2]

    try:
        past_dets = json.loads(past_det_json)
    except Exception:
        past_dets = []

    # Pick high-confidence past detections, sorted by confidence desc
    candidates = sorted(
        [d for d in past_dets if d.get("confidence", 0) >= CONFIDENCE_THRESHOLD and d.get("class")],
        key=lambda d: d["confidence"],
        reverse=True,
    )[:MAX_VP_BBOXES]

    if not candidates:
        rlog.info("No suitable bboxes from past frame — skipping VP job")
        return

    # Build sequential cls_ids (YOLOE requirement) and name lookup
    past_class_map: dict[int, str] = {}
    bboxes_list:    list[list]     = []
    cls_ids_list:   list[int]      = []

    for i, det in enumerate(candidates):
        bboxes_list.append(det["bbox"])
        cls_ids_list.append(i)
        past_class_map[i] = det["class"]

    bboxes  = np.array(bboxes_list,  dtype=np.float32)
    cls_ids = np.array(cls_ids_list, dtype=np.int32)

    rlog.info(
        f"VP job: checking {len(candidates)} past objects "
        f"{list(past_class_map.values())[:5]} in current frame"
    )

    try:
        results = await pool_ov_text.run_with_visual_prompts(
            current_image=current_arr,
            refer_image=past_arr,
            bboxes=bboxes,
            cls_ids=cls_ids,
        )
    except Exception as e:
        rlog.error(f"VP inference failed: {e}")
        return

    vp_dets  = extract_vp_detections(results, img_w, img_h, past_class_map)
    vp_summary = build_detection_summary(vp_dets, [])
    elapsed_ms = int((asyncio.get_event_loop().time() - t0) * 1000)

    rlog.info(f"VP result: {len(vp_dets)} matches in {elapsed_ms}ms — {vp_summary[:80]}")

    await r.xadd(reply_to, {"data": json.dumps({
        "id":                   request_id,
        "vp_detection_json":    json.dumps(vp_dets),
        "vp_detection_summary": vp_summary,
        "elapsed_ms":           elapsed_ms,
    })})
    # Auto-expire the per-request reply stream after 60s
    await r.expire(reply_to, 60)
    rlog.info(f"→ {reply_to}")


# ---------- Redis / HTTP ----------
r           = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
http_client = httpx.AsyncClient(timeout=30.0)


async def ensure_consumer_groups():
    for stream in (CAPTURE_STREAM, VP_STREAM):
        try:
            await r.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise


async def recover_pending_messages():
    for stream in (CAPTURE_STREAM, VP_STREAM):
        try:
            pending = await r.xpending(stream, CONSUMER_GROUP)
            count   = pending.get("pending", 0)
            if count == 0:
                continue
            log.info(f"Stream {stream}: {count} pending — checking stale")
            claimed = await r.xautoclaim(
                stream, CONSUMER_GROUP, CONSUMER_NAME,
                min_idle_time=MAX_PENDING_AGE, start_id="0-0", count=100,
            )
            messages = claimed[1] if isinstance(claimed, (list, tuple)) else []
            for mid, fields in messages:
                log.warning(f"Moving stale {mid} from {stream} to DLQ")
                await r.xadd(DLQ_STREAM, {
                    "original_id": mid,
                    "stream":      stream,
                    "data":        fields.get("data", ""),
                })
                await r.xack(stream, CONSUMER_GROUP, mid)
            if messages:
                log.info(f"Recovered {len(messages)} stale from {stream}")
        except Exception as e:
            log.warning(f"Pending recovery for {stream} failed (non-fatal): {e}")


# ---------- Main ----------
async def main():
    global pool_seg, pool_pose, pool_ov, pool_ov_text

    pool_seg     = ModelPool(SEG_MODEL,     MODEL_POOL_SIZE)
    pool_pose    = ModelPool(POSE_MODEL,    MODEL_POOL_SIZE)
    pool_ov      = ModelPool(OV_MODEL,      MODEL_POOL_SIZE)
    pool_ov_text = ModelPool(OV_TEXT_MODEL, MODEL_POOL_SIZE)

    await asyncio.gather(
        pool_seg.init(),
        pool_pose.init(),
        pool_ov.init(),
        pool_ov_text.init(),
    )

    await ensure_consumer_groups()
    await recover_pending_messages()
    log.info(
        f"Detector listening on '{CAPTURE_STREAM}' + '{VP_STREAM}' "
        f"(consumer={CONSUMER_NAME})"
    )

    while True:
        try:
            # Single blocking read across both streams — VP jobs get fair
            # scheduling alongside normal captures
            messages = await r.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={CAPTURE_STREAM: ">", VP_STREAM: ">"},
                count=1,
                block=2000,
            )
            if not messages:
                continue

            for stream_name, msg_list in messages:
                for msg_id, fields in msg_list:
                    data       = json.loads(fields["data"])
                    request_id = data.get("id", "unknown")
                    rlog       = logging.LoggerAdapter(log, {"request_id": request_id})

                    try:
                        if stream_name == VP_STREAM:
                            await handle_vp_job(data)
                        else:
                            await handle_capture_job(data)
                        await r.xack(stream_name, CONSUMER_GROUP, msg_id)

                    except Exception as e:
                        rlog.error(
                            f"Job failed on {stream_name}, routing to DLQ: {e}",
                            exc_info=True,
                        )
                        await r.xadd(DLQ_STREAM, {
                            "original_id": msg_id,
                            "stream":      stream_name,
                            "error":       str(e),
                            "data":        fields["data"],
                        })
                        await r.xack(stream_name, CONSUMER_GROUP, msg_id)

        except Exception as e:
            log.error(f"Outer loop error: {e}", exc_info=True)
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(http_client.aclose())
