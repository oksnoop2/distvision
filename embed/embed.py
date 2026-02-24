import json
import asyncio
import logging
import os
import base64
import uuid
import sys
from datetime import datetime, timezone
import httpx
import redis.asyncio as redis
from redis.exceptions import ResponseError
import chromadb

# ---------- Structured JSON Logging ----------
SERVICE_NAME = "embed"

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "service": SERVICE_NAME,
            "msg": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log["request_id"] = record.request_id
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        return json.dumps(log)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logging.basicConfig(handlers=[handler], level=logging.INFO)
log = logging.getLogger(SERVICE_NAME)


# ---------- Environment ----------
REDIS_HOST       = os.getenv("REDIS_HOST", "redis-service")
CHROMA_HOST      = os.getenv("CHROMA_HOST", "chroma-service")
EMBED_URL        = os.getenv("EMBED_LLM_URL", "http://embed-model:8085")
DUCKLING_URL     = os.getenv("DUCKLING_URL", "http://duckling:8000")
IMAGE_SERVER_URL = os.getenv("IMAGE_SERVER_URL", "http://image-server:8000")

log.info(f"EMBED_URL={EMBED_URL} DUCKLING_URL={DUCKLING_URL}")

# ---------- Streams ----------
INPUT_STREAM   = "stream:embed:input"
OUTPUT_STREAM  = "stream:vision:large:input"   # direct — no reranker
STATUS_STREAM  = "stream:interface:status"
DLQ_STREAM     = "stream:dlq:embed"
CONSUMER_GROUP = "embed_worker"
CONSUMER_NAME  = f"worker_{os.getenv('HOSTNAME', 'default')}"

# ---------- Chroma collections ----------
IMAGE_COLLECTION        = os.getenv("IMAGE_COLLECTION", "visual_memory_images")
CONVERSATION_COLLECTION = os.getenv("CONVERSATION_COLLECTION", "conversation_memory")

# ---------- Retrieval settings ----------
RETRIEVAL_COUNT = 10
EXPECTED_DIM    = 2048
RRF_K           = 60       # standard constant for Reciprocal Rank Fusion
MAX_PENDING_AGE = 300_000  # ms — reclaim messages stuck longer than 5 minutes

# ---------- Temporal intent detection ----------
TEMPORAL_SIGNALS = {
    "before", "earlier", "yesterday", "last", "when did", "has", "was",
    "used to", "previously", "ago", "moved", "changed", "still", "anymore",
    "lately", "recently", "today", "this morning", "tonight", "this evening",
    "an hour", "hours ago", "minute", "minutes ago", "earlier today",
    "just now", "a while", "a moment"
}

def needs_history(prompt: str, time_filter: dict | None) -> bool:
    """
    Return True if the prompt implies temporal reasoning or a time filter
    was extracted by Duckling. If False, skip past image retrieval entirely
    and send only the current image — saves embedding queries, ChromaDB
    lookups, and 220 tokens per skipped image in the VLM context.
    """
    if time_filter:
        return True
    lower = prompt.lower()
    return any(signal in lower for signal in TEMPORAL_SIGNALS)


# ---------- RRF merge ----------
def rrf_merge(
    image_ids: list[str],
    image_meta: list[dict],
    image_paths: list[str],
    text_ids: list[str],
    text_meta: list[dict],
    text_paths: list[str],
    k: int = RRF_K,
) -> tuple[list[str], list[dict]]:
    """
    Reciprocal Rank Fusion across two ranked lists.
    Images appearing highly in both the visual-similarity list and the
    query-intent list float to the top. No model required — runs in
    microseconds.

    Returns (merged_paths, merged_metadatas) in descending RRF score order.
    """
    id_to_data: dict[str, tuple[str, dict]] = {}
    for id_, meta, path in zip(image_ids, image_meta, image_paths):
        id_to_data[id_] = (path, meta)
    for id_, meta, path in zip(text_ids, text_meta, text_paths):
        id_to_data[id_] = (path, meta)

    scores: dict[str, float] = {}
    for rank, id_ in enumerate(image_ids):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (rank + k)
    for rank, id_ in enumerate(text_ids):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (rank + k)

    sorted_ids = sorted(scores, key=lambda i: scores[i], reverse=True)

    merged_paths = []
    merged_metas = []
    for id_ in sorted_ids:
        path, meta = id_to_data[id_]
        merged_paths.append(path)
        merged_metas.append(meta)

    return merged_paths, merged_metas


# ---------- Duckling HTTP helper ----------
async def extract_time_filter_http(
    text: str, reference_dt: datetime, http: httpx.AsyncClient
) -> dict | None:
    """
    Call Duckling to parse time expressions.
    Returns a dict with timestamp_epoch conditions, or None if no time
    reference is found.

    Important: we store the raw $gte/$lte values here and let
    build_where_clause split them into separate ChromaDB conditions.
    ChromaDB requires exactly one operator per expression — it cannot
    accept {"$gte": x, "$lte": y} in a single field dict.
    """
    payload = {
        "locale": "en_US",
        "text": text,
        "tz": "UTC",
        "reference_time": reference_dt.isoformat(),
    }
    for attempt in range(3):
        try:
            resp = await http.post(f"{DUCKLING_URL}/parse", data=payload, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            log.warning(f"Duckling attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                return None
            await asyncio.sleep(2 ** attempt)

    for entity in data:
        if entity.get("dim") != "time":
            continue
        value = entity.get("value")
        if not value:
            continue

        if isinstance(value, dict):
            if "from" in value and "to" in value:
                from_val = value["from"].get("value")
                to_val   = value["to"].get("value")
                if from_val and to_val:
                    from_epoch = datetime.fromisoformat(from_val.replace("Z", "+00:00")).timestamp()
                    to_epoch   = datetime.fromisoformat(to_val.replace("Z", "+00:00")).timestamp()
                    # Store as a range dict — build_where_clause will split it
                    return {"timestamp_epoch": {"$gte": from_epoch, "$lte": to_epoch}}

            elif "value" in value:
                exact = value["value"]
                if isinstance(exact, str):
                    epoch = datetime.fromisoformat(exact.replace("Z", "+00:00")).timestamp()
                    return {"timestamp_epoch": {"$eq": epoch}}

        elif isinstance(value, str):
            epoch = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            return {"timestamp_epoch": {"$eq": epoch}}

    return None


# ---------- Clients ----------
r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

# Single shared HTTP client — avoids rebuilding connection pools per call
http_client = httpx.AsyncClient(timeout=30.0)

chroma_client    = None
image_collection = None
conv_collection  = None


async def init_chroma():
    global chroma_client, image_collection, conv_collection
    log.info(f"Connecting to ChromaDB at {CHROMA_HOST}")
    chroma_client = await chromadb.AsyncHttpClient(host=CHROMA_HOST, port=8000)

    image_collection = await chroma_client.get_or_create_collection(
        name=IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(f"Image collection '{IMAGE_COLLECTION}' ready")

    conv_collection = await chroma_client.get_or_create_collection(
        name=CONVERSATION_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(f"Conversation collection '{CONVERSATION_COLLECTION}' ready")


# ---------- Embedding helpers ----------
async def get_image_embedding(image_url: str, text: str | None = None) -> list[float]:
    resp = await http_client.get(image_url, timeout=10)
    resp.raise_for_status()
    img_base64 = base64.b64encode(resp.content).decode("utf-8")

    prompt = (
        f"<image>\nRepresent this image and text for retrieval: {text}"
        if text
        else "<image>\nRepresent this image."
    )
    payload = {
        "content": prompt,
        "image_data": [{"data": img_base64, "id": 0}],
    }
    resp = await http_client.post(f"{EMBED_URL}/embedding", json=payload, timeout=30)
    resp.raise_for_status()
    return _extract_embedding(resp.json())


async def get_text_embedding(text: str) -> list[float]:
    payload = {"content": text}
    resp = await http_client.post(f"{EMBED_URL}/embedding", json=payload, timeout=30)
    resp.raise_for_status()
    return _extract_embedding(resp.json())


def _extract_embedding(data) -> list[float]:
    embedding = None

    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "embedding" in first:
            embedding = first["embedding"]
        elif isinstance(first, list):
            embedding = first
        elif len(data) == EXPECTED_DIM:
            embedding = data
        else:
            raise ValueError(f"Unexpected list format: {type(first)}")
    elif isinstance(data, dict) and "embedding" in data:
        embedding = data["embedding"]

    if embedding is None:
        raise ValueError(f"Could not extract embedding from: {data}")

    # Some models return per-token embeddings — take the last token
    if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
        embedding = embedding[-1]
        log.info(f"Extracted last-token embedding, length {len(embedding)}")

    if not isinstance(embedding, list):
        raise ValueError(f"Embedding is not a list: {type(embedding)}")

    if len(embedding) != EXPECTED_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: got {len(embedding)}, expected {EXPECTED_DIM}. "
            f"Check that the embed model has not changed."
        )

    return embedding


# ---------- ChromaDB where-clause builder ----------
def build_where_clause(exclude_timestamp: str | None, time_filter: dict | None) -> dict | None:
    """
    Assemble a ChromaDB where clause from optional exclusion and time filter.

    Critical constraint: ChromaDB requires exactly one operator per field
    expression. A range like {"$gte": x, "$lte": y} is NOT valid as a single
    expression — it must be split into two separate conditions joined by $and.

    Valid:   {"timestamp_epoch": {"$gte": x}}
    Valid:   {"$and": [{"timestamp_epoch": {"$gte": x}}, {"timestamp_epoch": {"$lte": y}}]}
    Invalid: {"timestamp_epoch": {"$gte": x, "$lte": y}}  ← ChromaDB rejects this
    """
    conditions = []

    if exclude_timestamp:
        conditions.append({"timestamp": {"$ne": exclude_timestamp}})

    if time_filter:
        ts_cond = time_filter.get("timestamp_epoch")
        if isinstance(ts_cond, dict):
            # Split multi-operator range into individual conditions
            if "$gte" in ts_cond:
                conditions.append({"timestamp_epoch": {"$gte": ts_cond["$gte"]}})
            if "$lte" in ts_cond:
                conditions.append({"timestamp_epoch": {"$lte": ts_cond["$lte"]}})
            if "$eq" in ts_cond:
                conditions.append({"timestamp_epoch": {"$eq": ts_cond["$eq"]}})
        elif ts_cond is not None:
            # Bare value — shouldn't reach here given extract_time_filter_http
            # always wraps in an operator dict, but handle defensively
            conditions.append({"timestamp_epoch": {"$eq": float(ts_cond)}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------- Dead letter / pending message recovery ----------
async def recover_pending_messages():
    """
    On startup, claim any messages left pending by a previous worker that
    crashed. Messages pending longer than MAX_PENDING_AGE go to the DLQ
    for inspection rather than being retried blindly.
    """
    try:
        pending = await r.xpending(INPUT_STREAM, CONSUMER_GROUP)
        count = pending.get("pending", 0)
        if count == 0:
            return
        log.info(f"Found {count} pending messages — checking for stale entries")

        claimed = await r.xautoclaim(
            INPUT_STREAM,
            CONSUMER_GROUP,
            CONSUMER_NAME,
            min_idle_time=MAX_PENDING_AGE,
            start_id="0-0",
            count=100,
        )
        messages = claimed[1] if isinstance(claimed, (list, tuple)) else []
        if not messages:
            return

        for mid, fields in messages:
            log.warning(f"Moving stale pending message {mid} to DLQ",
                        extra={"request_id": None})
            await r.xadd(DLQ_STREAM, {"original_id": mid, "data": fields.get("data", "")})
            await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)

        log.info(f"Recovered {len(messages)} stale messages → DLQ")
    except Exception as e:
        log.warning(f"Pending recovery failed (non-fatal): {e}")


# ---------- Message handlers ----------
async def handle_conversation(data: dict):
    """Store a conversation exchange in the conversation ChromaDB collection."""
    user_text      = data["user"]
    assistant_text = data["assistant"]
    timestamp      = data["timestamp"]
    combined       = f"User: {user_text}\nAssistant: {assistant_text}"

    embedding = await get_text_embedding(combined)

    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        timestamp_epoch = dt.timestamp()
    except Exception:
        timestamp_epoch = 0.0

    memory_id = f"conv_{timestamp}_{uuid.uuid4().hex[:8]}"
    metadata  = {
        "timestamp":       timestamp,
        "timestamp_epoch": timestamp_epoch,
        "user":            user_text,
        "assistant":       assistant_text,
        "image_url":       data.get("image_url", ""),
    }
    await conv_collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[combined],
    )
    log.info(f"Stored conversation {memory_id}")


async def handle_image(data: dict, rlog: logging.LoggerAdapter):
    """
    Process an image message from the detector.

    For background captures (no prompt): index and exit.
    For user requests (has prompt):
      1. Determine whether historical context is needed.
      2. If yes, run dual-embedding retrieval with RRF merge.
      3. Forward to vision_large with appropriately sized context.
    """
    request_id        = data.get("id")
    prompt            = data.get("prompt")
    filepath          = data.get("filepath")
    timestamp         = data.get("timestamp")
    detection_summary = data.get("detection_summary", "")
    detection_json    = data.get("detection_json", "")

    if not request_id or not filepath:
        rlog.warning(f"Missing id or filepath, skipping: {data}")
        return

    filename  = os.path.basename(filepath)
    image_url = f"{IMAGE_SERVER_URL}/{filename}"

    # Build document text for storage
    if prompt and detection_summary:
        document_text  = f"{prompt}\nObjects: {detection_summary}"
        embedding_text = document_text
    elif prompt:
        document_text  = prompt
        embedding_text = prompt
    elif detection_summary:
        document_text  = detection_summary
        embedding_text = detection_summary
    else:
        document_text  = ""
        embedding_text = None

    # Get image embedding
    image_embedding = await get_image_embedding(image_url, text=embedding_text)

    # Parse timestamp to epoch
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        timestamp_epoch = dt.timestamp()
    except Exception as e:
        rlog.warning(f"Could not parse timestamp {timestamp}: {e}")
        timestamp_epoch = 0.0

    # Build metadata
    metadata = {
        "timestamp":       timestamp,
        "timestamp_epoch": timestamp_epoch,
        "filepath":        filepath,
        "has_prompt":      bool(prompt),
    }
    if detection_json:
        metadata["detection_data"] = detection_json
    if detection_summary:
        metadata["summary"] = detection_summary

    memory_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

    # Store in ChromaDB — use [""] not None when no document text
    await image_collection.add(
        ids=[memory_id],
        embeddings=[image_embedding],
        metadatas=[metadata],
        documents=[document_text if document_text else ""],
    )
    rlog.info(
        f"Indexed image {memory_id}"
        + (f" doc='{document_text[:60]}...'" if document_text else "")
    )

    # Background capture — nothing more to do
    if not prompt:
        rlog.info(f"Background capture indexed (id: {request_id})")
        return

    # ---------- User request: retrieval ----------
    now         = datetime.now(timezone.utc)
    time_filter = await extract_time_filter_http(prompt, now, http_client)
    if time_filter:
        rlog.info(f"Time filter applied: {time_filter}")

    where_clause = build_where_clause(
        exclude_timestamp=timestamp,
        time_filter=time_filter,
    )

    past_image_urls = []
    stripped_metas  = []
    text_embedding  = None   # may be computed below, reused for conv retrieval

    if needs_history(prompt, time_filter):
        rlog.info("Temporal query detected — running dual-embedding retrieval")

        # Query 1: visual similarity (what does this scene look like?)
        img_results = await image_collection.query(
            query_embeddings=[image_embedding],
            n_results=RETRIEVAL_COUNT,
            where=where_clause,
        )
        img_ids   = img_results.get("ids", [[]])[0]
        img_metas = img_results.get("metadatas", [[]])[0]
        img_paths = [
            f"{IMAGE_SERVER_URL}/{os.path.basename(m['filepath'])}"
            for m in img_metas
        ]

        # Query 2: query intent (what is the user asking about?)
        text_embedding = await get_text_embedding(prompt)
        txt_results = await image_collection.query(
            query_embeddings=[text_embedding],
            n_results=RETRIEVAL_COUNT,
            where=where_clause,
        )
        txt_ids   = txt_results.get("ids", [[]])[0]
        txt_metas = txt_results.get("metadatas", [[]])[0]
        txt_paths = [
            f"{IMAGE_SERVER_URL}/{os.path.basename(m['filepath'])}"
            for m in txt_metas
        ]

        # Merge with Reciprocal Rank Fusion
        merged_paths, merged_metas = rrf_merge(
            img_ids, img_metas, img_paths,
            txt_ids, txt_metas, txt_paths,
        )

        past_image_urls = merged_paths
        stripped_metas  = [
            {"timestamp": m.get("timestamp"), "summary": m.get("summary", "")}
            for m in merged_metas
        ]

        rlog.info(
            f"RRF retrieval: {len(img_ids)} visual + {len(txt_ids)} text "
            f"→ {len(merged_paths)} merged"
        )

        # Send image preview to interface
        status_msg = {
            "type":              "image_update",
            "request_id":        request_id,
            "container":         "embed",
            "current_image_url": image_url,
            "past_image_urls":   past_image_urls[:3],
            "timestamp":         timestamp,
        }
        await r.xadd(STATUS_STREAM, {"data": json.dumps(status_msg)})

    else:
        rlog.info("Present-moment query — skipping past image retrieval")

    # ---------- Conversation retrieval ----------
    # Reuse text_embedding if already computed above, otherwise compute now
    if text_embedding is None:
        text_embedding = await get_text_embedding(prompt)

    # Apply same time filter to conversations so temporal context is consistent
    conv_where = build_where_clause(
        exclude_timestamp=None,
        time_filter=time_filter,
    )

    conv_results = await conv_collection.query(
        query_embeddings=[text_embedding],
        n_results=5,
        where=conv_where,
    )
    past_conversations = [
        {
            "user":      m.get("user"),
            "assistant": m.get("assistant"),
            "timestamp": m.get("timestamp"),
        }
        for m in conv_results.get("metadatas", [[]])[0]
    ]

    # ---------- Forward to vision_large ----------
    out = {
        "id":                 request_id,
        "prompt":             prompt,
        "current_image_url":  image_url,
        "past_image_urls":    past_image_urls,
        "past_metadatas":     stripped_metas,
        "past_conversations": past_conversations,
        "timestamp":          timestamp,
    }
    await r.xadd(OUTPUT_STREAM, {"data": json.dumps(out)})
    rlog.info(
        f"→ vision_large | past_images={len(past_image_urls)} "
        f"past_conversations={len(past_conversations)}"
    )


# ---------- Main loop ----------
async def main():
    await init_chroma()

    try:
        await r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    await recover_pending_messages()
    log.info(f"Embed worker listening (consumer={CONSUMER_NAME})")

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

            mid, mdata = messages[0][1][0]
            data       = json.loads(mdata["data"])
            request_id = data.get("id", "unknown")

            rlog = logging.LoggerAdapter(log, {"request_id": request_id})

            try:
                if data.get("type") == "conversation":
                    await handle_conversation(data)
                else:
                    await handle_image(data, rlog)

                await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)

            except Exception as e:
                rlog.error(f"Processing failed, sending to DLQ: {e}", exc_info=True)
                await r.xadd(
                    DLQ_STREAM,
                    {"original_id": mid, "error": str(e), "data": mdata["data"]},
                )
                await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)

        except Exception as e:
            log.error(f"Outer loop error: {e}", exc_info=True)
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(http_client.aclose())
