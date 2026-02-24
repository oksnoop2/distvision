import json
import asyncio
import logging
import os
import re
import sys
import uuid
import base64
import cv2
import numpy as np
import httpx
import humanize
import redis.asyncio as redis
from redis.exceptions import ResponseError
from openai import AsyncOpenAI
from datetime import datetime, timezone

# ---------- Structured JSON Logging ----------
SERVICE_NAME = "vision_large"

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
LLM_URL          = os.getenv("LLM_URL", "http://vision-large-model:8083/v1")
IMAGE_SERVER_URL = os.getenv("IMAGE_SERVER_URL", "http://image-server:8000")
MAX_PAST_IMAGES  = int(os.getenv("MAX_PAST_IMAGES", "3"))
MAX_HISTORY      = int(os.getenv("MAX_HISTORY", "20"))

INPUT_STREAM   = "stream:vision:large:input"
OUTPUT_STREAM  = "stream:interface:output"
EMBED_STREAM   = "stream:embed:input"
STATUS_STREAM  = "stream:interface:status"
DLQ_STREAM     = "stream:dlq:vision_large"
CONSUMER_GROUP = "vision:large"
CONSUMER_NAME  = f"worker_{os.getenv('HOSTNAME', 'default')}"

MAX_PENDING_AGE = 300_000  # ms

# ---------- Clients ----------
r          = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
llm_client = AsyncOpenAI(base_url=LLM_URL, api_key="sk-no-key", timeout=300.0)
http_client = httpx.AsyncClient(timeout=30.0)


# ---------- Helpers ----------
def relative_time(iso_timestamp: str, reference: datetime) -> str:
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        delta = reference - dt
        return humanize.naturaltime(delta)
    except Exception:
        return iso_timestamp


async def process_image(image_url: str) -> str | None:
    """
    Download, resize to max 640px, encode as base64 JPEG data URL.
    Uses the shared http_client — avoids rebuilding connection pools per call.
    """
    try:
        resp = await http_client.get(image_url, timeout=10)
        resp.raise_for_status()
        img_array = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            log.warning(f"Failed to decode image: {image_url}")
            return None

        h, w  = img.shape[:2]
        scale = 640 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

        ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            log.warning(f"Failed to encode image: {image_url}")
            return None

        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        log.warning(f"process_image failed for {image_url}: {e}")
        return None


async def get_recent_history(request_id: str, max_turns: int) -> list[dict]:
    try:
        raw     = await r.lrange("chat:history", 0, max_turns - 1)
        history = [json.loads(m) for m in raw]
        history.reverse()
        return [m for m in history if m.get("id") != request_id]
    except Exception as e:
        log.warning(f"get_recent_history failed: {e}")
        return []


async def store_assistant_message(request_id: str, content: str, timestamp: str):
    msg = {"id": request_id, "role": "assistant",
           "content": content, "timestamp": timestamp}
    await r.lpush("chat:history", json.dumps(msg))
    await r.ltrim("chat:history", 0, 99)


async def send_conversation_to_embed(user: str, assistant: str,
                                     timestamp: str, image_url: str = ""):
    payload = {
        "id":        str(uuid.uuid4()),
        "type":      "conversation",
        "user":      user,
        "assistant": assistant,
        "timestamp": timestamp,
        "image_url": image_url,
    }
    await r.xadd(EMBED_STREAM, {"data": json.dumps(payload)})


def extract_thinking_and_answer(full_response: str) -> tuple[str | None, str]:
    """
    Split <think>...</think> from the final answer.
    Returns (thinking, answer). thinking is None if no tags present.
    """
    match = re.search(r"<think>(.*?)</think>(.*)", full_response, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, full_response.strip()


# ---------- Prompt builder ----------
def build_messages(
    prompt: str,
    timestamp: str,
    now: datetime,
    current_image_data: str | None,
    past_images: list[tuple[str, dict]],   # [(data_url, metadata), ...]
    past_conversations: list[dict],
    recent_history: list[dict],
    detection_summary: str,
) -> list[dict]:
    """
    Build the messages payload for the VLM.

    Design decisions:
    - Images come before the question so the VLM encodes them into its
      attention context before reading the query.
    - Detection summary is framed as supplementary hints, not ground truth.
      The VLM is explicitly instructed to trust its own visual observations
      over the summary. This prevents it from answering "0 pink items"
      because the detector missed them.
    - Past conversations are injected as assistant/user turns in the correct
      chat history position so the model has genuine conversational context.
    - Chat history sits between the system message and the current user turn
      so it looks like a real ongoing conversation.
    - The current user turn is structured: images first → detection hint →
      question last. This order maximises the chance the model looks at the
      image before reading the question.
    """
    current_time_str = now.strftime("%B %d, %Y at %I:%M %p UTC")

    system_content = f"""\
You are a visual AI assistant connected to a live camera system.

VISUAL AUTHORITY:
- Your PRIMARY source of truth is what you can directly see in the images.
- Detection summaries are SUPPLEMENTARY HINTS generated by a separate model.
  They are often incomplete — they miss objects identified by color, texture,
  or context. If you can see something the summary does not mention, trust
  your eyes. Never say something is absent just because the detector missed it.

ANSWERING:
- Answer based on direct visual observation first, detection hints second.
- For counting questions (how many, are there any), look carefully at the image
  yourself. Do not defer to the detection summary for the final count.
- For temporal questions, compare images using their timestamps.
- If you genuinely cannot determine something from the images, say so briefly.
- Keep answers concise and direct. Do not narrate your reasoning process in the
  final answer unless asked.

Current date and time: {current_time_str}"""

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Inject relevant past conversations as history turns
    # These are semantically retrieved exchanges, not chronological chat log
    if past_conversations:
        for conv in past_conversations:
            u = conv.get("user", "")
            a = conv.get("assistant", "")
            ts = conv.get("timestamp", "")
            rel = relative_time(ts, now) if ts else ""
            label = f" [{rel}]" if rel else ""
            if u:
                messages.append({"role": "user",      "content": f"{u}{label}"})
            if a:
                messages.append({"role": "assistant", "content": a})

    # Inject recent chat history (maintains conversational continuity)
    for msg in recent_history:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Build the current user turn
    user_content: list[dict] = []

    # Past images first (oldest context)
    for data_url, meta in past_images:
        ts  = meta.get("timestamp", "")
        rel = relative_time(ts, now) if ts else "unknown time"
        summary = meta.get("summary", "")
        label = f"Past image ({rel})"
        if summary:
            label += f" — detector hints: {summary}"
        user_content.append({"type": "text", "text": label + ":"})
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})

    # Current image
    if current_image_data:
        rel_current = relative_time(timestamp, now) if timestamp else "just now"
        current_label = f"Current image ({rel_current})"
        if detection_summary:
            current_label += (
                f"\nDetector hints (incomplete — trust your visual observation): "
                f"{detection_summary}"
            )
        user_content.append({"type": "text", "text": current_label + ":"})
        user_content.append({"type": "image_url",
                              "image_url": {"url": current_image_data}})

    # Question last — model has seen all images before reading the query
    user_content.append({"type": "text", "text": prompt})

    messages.append({"role": "user", "content": user_content})
    return messages


# ---------- Pending message recovery ----------
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
            await r.xadd(DLQ_STREAM,
                         {"original_id": mid, "data": fields.get("data", "")})
            await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)
        if messages:
            log.info(f"Recovered {len(messages)} stale messages → DLQ")
    except Exception as e:
        log.warning(f"Pending recovery failed (non-fatal): {e}")


# ---------- Main loop ----------
async def main():
    try:
        await r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    await recover_pending_messages()
    log.info(f"Vision Large listening (consumer={CONSUMER_NAME})")

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
            rlog       = logging.LoggerAdapter(log, {"request_id": request_id})

            try:
                await process_message(data, rlog)
                await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)
            except Exception as e:
                rlog.error(f"Processing failed, routing to DLQ: {e}", exc_info=True)
                await r.xadd(DLQ_STREAM,
                             {"original_id": mid, "error": str(e),
                              "data": mdata["data"]})
                await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)

        except Exception as e:
            log.error(f"Outer loop error: {e}", exc_info=True)
            await asyncio.sleep(1)


async def process_message(data: dict, rlog: logging.LoggerAdapter):
    request_id  = data.get("id")
    prompt      = data.get("prompt")
    timestamp   = data.get("timestamp")

    if not request_id or not prompt:
        rlog.warning(f"Missing id or prompt — skipping: {data}")
        return

    rlog.info(f"Processing prompt='{prompt[:60]}...' " if len(prompt) > 60
              else f"Processing prompt='{prompt}'")

    # Status update to interface
    await r.xadd(STATUS_STREAM, {"data": json.dumps({
        "type":       "status",
        "request_id": request_id,
        "container":  SERVICE_NAME,
        "status":     "processing",
        "timestamp":  timestamp,
    })})

    current_image_url  = data.get("current_image_url") or data.get("filepath")
    past_image_urls    = data.get("past_image_urls", [])[:MAX_PAST_IMAGES]
    past_metadatas     = data.get("past_metadatas",  [])[:MAX_PAST_IMAGES]
    past_conversations = data.get("past_conversations", [])
    detection_summary  = data.get("detection_summary", "")

    now = datetime.now(timezone.utc)

    # Fetch and encode all images concurrently
    all_urls = ([current_image_url] if current_image_url else []) + past_image_urls
    all_data = await asyncio.gather(*[process_image(u) for u in all_urls],
                                    return_exceptions=True)

    # Split results back out
    current_image_data: str | None = None
    if current_image_url:
        result = all_data[0]
        if isinstance(result, Exception):
            rlog.warning(f"Failed to fetch current image: {result}")
        else:
            current_image_data = result
        past_results = all_data[1:]
    else:
        past_results = all_data

    past_images: list[tuple[str, dict]] = []
    for url, result, meta in zip(past_image_urls, past_results, past_metadatas):
        if isinstance(result, Exception):
            rlog.warning(f"Failed to fetch past image {url}: {result}")
        elif result:
            past_images.append((result, meta))

    rlog.info(
        f"Images ready: current={'yes' if current_image_data else 'no'} "
        f"past={len(past_images)}"
    )

    # Fetch chat history
    recent_history = await get_recent_history(request_id, MAX_HISTORY)

    # Build messages
    messages_payload = build_messages(
        prompt=prompt,
        timestamp=timestamp or "",
        now=now,
        current_image_data=current_image_data,
        past_images=past_images,
        past_conversations=past_conversations,
        recent_history=recent_history,
        detection_summary=detection_summary,
    )

    total_images = (1 if current_image_data else 0) + len(past_images)
    rlog.info(
        f"Sending to LLM: {total_images} image(s), "
        f"{len(recent_history)} history turns, "
        f"{len(past_conversations)} retrieved conversations"
    )

    # Call LLM
    resp = await llm_client.chat.completions.create(
        model="Qwen3-VL",
        messages=messages_payload,
        max_tokens=2000,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=0.0,
        stream=False,
    )

    full_response = resp.choices[0].message.content or ""
    thinking, answer = extract_thinking_and_answer(full_response)

    if not answer:
        rlog.warning("Empty answer after thinking extraction — sending error to interface")
        answer = "I wasn't able to generate a response. Please try again."

    if thinking:
        await r.xadd(STATUS_STREAM, {"data": json.dumps({
            "type":       "thinking",
            "request_id": request_id,
            "container":  SERVICE_NAME,
            "content":    thinking,
            "timestamp":  timestamp,
        })})
        rlog.info(f"Thinking block: {len(thinking)} chars")

    rlog.info(f"Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")

    # Persist to chat history and long-term memory
    response_ts = datetime.now(timezone.utc).isoformat()
    await store_assistant_message(request_id, answer, response_ts)
    await send_conversation_to_embed(
        user=prompt,
        assistant=answer,
        timestamp=timestamp or response_ts,
        image_url=current_image_url or "",
    )

    # Forward answer to interface
    await r.xadd(OUTPUT_STREAM, {"data": json.dumps({
        "id":        request_id,
        "response":  answer,
        "timestamp": timestamp,
    })})
    rlog.info("→ interface")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(http_client.aclose())
