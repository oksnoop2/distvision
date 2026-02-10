import json
import asyncio
import os
import redis.asyncio as redis
from redis.exceptions import ResponseError # <--- FIX
from openai import OpenAI

REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
LLM_URL = os.getenv("LLM_URL", "http://vision-small-model:8082/v1")
IMAGE_SERVER_URL = os.getenv("IMAGE_SERVER_URL", "http://image-server:8000")

INPUT_STREAM = "stream:vision:small:input"
OUTPUT_STREAM = "stream:embed:input"
CONSUMER_GROUP = "vision:small"
CONSUMER_NAME = "worker"

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
client = OpenAI(base_url=LLM_URL, api_key="sk-no-key")

async def main():
    try:
        await r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
    except ResponseError as e: # <--- FIX
        if "BUSYGROUP" not in str(e): raise e

    print("Vision Small listening...")
    while True:
        try:
            messages = await r.xreadgroup(
                groupname=CONSUMER_GROUP, consumername=CONSUMER_NAME,
                streams={INPUT_STREAM: ">"}, count=1, block=2000
            )
            if not messages: continue

            mid, mdata = messages[0][1][0]
            data = json.loads(mdata["data"])
            filepath = data.get("filepath")

            if filepath:
                filename = os.path.basename(filepath)
                url = f"{IMAGE_SERVER_URL}/{filename}"
                print(f"Processing {url}")

                try:
                    resp = await asyncio.to_thread(
                        client.chat.completions.create,
                        model="Qwen3-VL",  # Use actual model name
                        messages=[{
                        "role": "user",
                        "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": url}}
                    ]
                }],
                max_tokens=200,
                temperature=0.7,       # Instruct model: 0.7
                top_p=0.8,            # Instruct model: 0.8
                presence_penalty=1.5, # Instruct model: 1.5
                #top_k=20,
                stream=False
            )
                    desc = resp.choices[0].message.content

                    out = {
                        "id": data.get("id"),
                        "description": desc,
                        "filepath": filepath,
                        "timestamp": data.get("timestamp"),
                        "task": "index"
                    }
                    await r.xadd(OUTPUT_STREAM, {"data": json.dumps(out)})
                    print(" -> Embed")
                except Exception as e:
                    print(f"Inference Error: {e}")

            await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)
        except Exception as e:
            print(f"VS Error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
