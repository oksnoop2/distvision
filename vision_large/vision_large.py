import json
import asyncio
import os
import redis.asyncio as redis
from redis.exceptions import ResponseError
from openai import OpenAI

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

            # Extract fields (all services now use "id")
            request_id = data.get("id")
            filepath = data.get("filepath")
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

            if filepath:
                filename = os.path.basename(filepath)
                url = f"{IMAGE_SERVER_URL}/{filename}"
                print(f"Processing Image: {url}")
                messages_payload.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": url}}
                    ]
                })
            else:
                print(f"Processing Text Query: {prompt}")
                messages_payload.append({
                    "role": "user",
                    "content": prompt  # simple string for text-only
                })

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
                "id": request_id,                # use consistent "id"
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
