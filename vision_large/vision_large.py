import json
import asyncio
import os
import redis.asyncio as redis
from redis.exceptions import ResponseError
from openai import OpenAI
import cv2
import numpy as np
import base64
import httpx

REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
LLM_URL = os.getenv("LLM_URL", "http://vision-large-model:8083/v1")
IMAGE_SERVER_URL = os.getenv("IMAGE_SERVER_URL", "http://image-server:8000")

INPUT_STREAM = "stream:vision:large:input"
OUTPUT_STREAM = "stream:interface:output"
CONSUMER_GROUP = "vision:large"
CONSUMER_NAME = "worker"

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
client = OpenAI(base_url=LLM_URL, api_key="sk-no-key")

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
            context = data.get("context_text", "")
            timestamp = data.get("timestamp")

            if not request_id or not prompt:
                print(f"⚠️ Missing id or prompt in message: {data}")
                await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)
                continue

            messages_payload = []

            if context:
                messages_payload.append({
                    "role": "system",
                    "content": f"You are a helpful AI assistant. Respond in English only. Keep your responses concise.\n\n{context}"
                })
            else:
                messages_payload.append({
                    "role": "system",
                    "content": "You are a helpful AI assistant. Respond in English only. Keep your responses concise."
                })

            # --- Image handling ---
            current_image_url = data.get("current_image_url") or data.get("filepath")
            if current_image_url:
                # current_image_url is already a full URL from the embed worker
                url = current_image_url
                print(f"Processing Image: {url}")

                # Download the full image
                async with httpx.AsyncClient() as http_client:
                    resp = await http_client.get(url, timeout=10)
                    resp.raise_for_status()
                    img_bytes = resp.content

                # Decode with OpenCV
                img_array = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Failed to decode image")

                # Resize while preserving aspect ratio (max dimension 640px)
                max_size = 640
                h, w = img.shape[:2]
                scale = max_size / max(h, w)
                if scale < 1:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    resized = img  # already smaller than max_size

                # Encode back to JPEG (quality 85 saves bandwidth)
                ret, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    raise ValueError("Failed to encode resized image")
                base64_data = base64.b64encode(buf).decode('utf-8')
                data_url = f"data:image/jpeg;base64,{base64_data}"

                # Build user message with base64 image
                user_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            else:
                # Text‑only query
                user_content = prompt

            messages_payload.append({"role": "user", "content": user_content})

            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model="Qwen3-VL",
                messages=messages_payload,
                max_tokens=1024,
                temperature=1.0,
                top_p=0.95,
                presence_penalty=0.0,
                stream=False
            )

            answer = resp.choices[0].message.content

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
