import json
import asyncio
import os
import redis.asyncio as redis
from redis.exceptions import ResponseError
from openai import AsyncOpenAI
import cv2
import numpy as np
import base64
import httpx
import uuid
from datetime import datetime, timezone

REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
LLM_URL = os.getenv("LLM_URL", "http://vision-large-model:8083/v1")
IMAGE_SERVER_URL = os.getenv("IMAGE_SERVER_URL", "http://image-server:8000")
MAX_PAST_IMAGES = int(os.getenv("MAX_PAST_IMAGES", "3"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
EMBED_STREAM = "stream:embed:input"
STATUS_STREAM = "stream:interface:status"

INPUT_STREAM = "stream:vision:large:input"
OUTPUT_STREAM = "stream:interface:output"
CONSUMER_GROUP = "vision:large"
CONSUMER_NAME = "worker"

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
client = AsyncOpenAI(base_url=LLM_URL, api_key="sk-no-key", timeout=300.0)

async def process_image(image_url: str) -> str | None:
    """Download, resize, and encode image to base64 data URL. Return None on failure."""
    try:
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(image_url, timeout=10)
            resp.raise_for_status()
            img_bytes = resp.content

        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to decode image: {image_url}")
            return None

        max_size = 640
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = img

        ret, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            print(f"Failed to encode resized image: {image_url}")
            return None

        base64_data = base64.b64encode(buf).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None

async def get_recent_history(request_id: str, max_turns: int = 20) -> list[dict]:
    """Fetch last `max_turns` messages from Redis chat history, excluding the current request."""
    try:
        raw = await r.lrange("chat:history", 0, max_turns - 1)
        history = [json.loads(msg) for msg in raw]
        history.reverse()
        history = [msg for msg in history if msg.get("id") != request_id]
        return history
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return []

async def store_assistant_message(request_id: str, content: str, timestamp: str):
    """Store assistant response in Redis chat history."""
    msg = {
        "id": request_id,
        "role": "assistant",
        "content": content,
        "timestamp": timestamp
    }
    await r.lpush("chat:history", json.dumps(msg))
    await r.ltrim("chat:history", 0, 99)

async def send_conversation_to_embed(user: str, assistant: str, timestamp: str, image_url: str = ""):
    """Send a conversation exchange to embed:input for long‑term storage."""
    payload = {
        "id": str(uuid.uuid4()),
        "type": "conversation",
        "user": user,
        "assistant": assistant,
        "timestamp": timestamp,
        "image_url": image_url
    }
    await r.xadd(EMBED_STREAM, {"data": json.dumps(payload)})
    print("Stored conversation exchange in embed stream")

async def main():
    try:
        await r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise e

    print("Vision Large listening...")

    while True:
        try:
            messages = await r.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={INPUT_STREAM: ">"},
                count=1,
                block=2000
            )
            if not messages:
                continue

            mid, mdata = messages[0][1][0]
            data = json.loads(mdata["data"])

            print("VL INPUT:", data)

            request_id = data.get("id")
            prompt = data.get("prompt")
            timestamp = data.get("timestamp")
            context = data.get("context_text", "")

            if not request_id or not prompt:
                print(f"⚠️ Missing id or prompt in message: {data}")
                await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)
                continue

            # --- Send processing status to interface ---
            status_msg = {
                "type": "status",
                "request_id": request_id,
                "container": "vision_large",
                "status": "processing",
                "timestamp": timestamp
            }
            await r.xadd(STATUS_STREAM, {"data": json.dumps(status_msg)})

            # --- Retrieve data from embed worker ---
            current_image_url = data.get("current_image_url") or data.get("filepath")
            past_image_urls = data.get("past_image_urls", [])
            past_metadatas = data.get("past_metadatas", [])
            past_conversations = data.get("past_conversations", [])

            # Limit number of past images
            if past_image_urls:
                past_image_urls = past_image_urls[:MAX_PAST_IMAGES]
                past_metadatas = past_metadatas[:MAX_PAST_IMAGES]

            # --- Build system message with enhanced instructions and current time ---
            current_time = datetime.now(timezone.utc).isoformat()
            system_content = (
                "You are a helpful AI assistant. You will be shown one current image and several past images, "
                "each with a timestamp. Answer the user's question covering any of the six Ws (What, Why, When, Where, How, Who) "
                "based on the visual information and the temporal context. "
                "If a question refers to 'today', 'now', or similar, use the current time provided below to determine if an image timestamp falls within today. "
                "If the answer cannot be determined from the images, say 'I don't know based on the available images.' "
                "Keep responses concise and in English.\n\n"
                f"Current date and time (for reference): {current_time}."
            )
            if context:
                system_content += f"\n\n{context}"

            messages_payload = [{"role": "system", "content": system_content}]

            # --- Add retrieved past conversations as additional system context ---
            if past_conversations:
                conv_text = "Relevant past exchanges:\n"
                for conv in past_conversations:
                    conv_text += f"User: {conv['user']}\nAssistant: {conv['assistant']}\n---\n"
                messages_payload.append({"role": "system", "content": conv_text})

            # --- Add recent chat history (global) ---
            history = await get_recent_history(request_id, MAX_HISTORY)
            for msg in history:
                if msg["role"] in ("user", "assistant"):
                    messages_payload.append({"role": msg["role"], "content": msg["content"]})

            # --- Concurrently process all images ---
            image_urls = []
            if current_image_url:
                image_urls.append(current_image_url)
            image_urls.extend(past_image_urls)

            image_results = await asyncio.gather(*[process_image(url) for url in image_urls])

            # Build user_content in image‑first order
            user_content = []

            if current_image_url and image_results[0]:
                user_content.append({
                    "type": "text",
                    "text": f"Current image (timestamp: {timestamp}):"
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_results[0]}
                })
            elif current_image_url:
                print("Warning: Could not process current image")

            for idx, (past_url, meta, data_url) in enumerate(zip(past_image_urls, past_metadatas, image_results[1:])):
                if data_url:
                    past_ts = meta.get("timestamp", "unknown")
                    user_content.append({
                        "type": "text",
                        "text": f"Past image {idx+1} (timestamp: {past_ts}):"
                    })
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
                else:
                    print(f"Warning: Could not process past image {idx+1}")

            user_content.append({"type": "text", "text": prompt})
            messages_payload.append({"role": "user", "content": user_content})

            # --- Call LLM asynchronously ---
            resp = await client.chat.completions.create(
                model="Qwen3-VL",
                messages=messages_payload,
                max_tokens=1024,
                temperature=1.0,
                top_p=0.95,
                presence_penalty=0.0,
                stream=False
            )

            answer = resp.choices[0].message.content

            # --- Store assistant message ---
            await store_assistant_message(request_id, answer, datetime.now(timezone.utc).isoformat())

            # --- Send exchange to embed for long‑term memory ---
            await send_conversation_to_embed(
                user=prompt,
                assistant=answer,
                timestamp=timestamp,
                image_url=current_image_url or ""
            )

            # --- Forward response to interface ---
            out = {
                "id": request_id,
                "response": answer,
                "timestamp": timestamp,
            }
            await r.xadd(OUTPUT_STREAM, {"data": json.dumps(out)})
            print(f"→ Interface (id: {request_id})")

            await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)

        except Exception as e:
            print(f"Vision Large Error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
