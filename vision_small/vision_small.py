import json
import asyncio
import os
import base64
import requests
from io import BytesIO
from PIL import Image

import redis.asyncio as redis
from redis.exceptions import ResponseError
from openai import OpenAI

REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
LLM_URL = os.getenv("LLM_URL", "http://vision-small-model:8082/v1")
IMAGE_SERVER_URL = os.getenv("IMAGE_SERVER_URL", "http://image-server:8000")

INPUT_STREAM = "stream:vision:small:input"
OUTPUT_STREAM = "stream:embed:input"
CONSUMER_GROUP = "vision:small"
CONSUMER_NAME = "worker"

MAX_DIM = 768  # Longest side target for resized images

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
client = OpenAI(base_url=LLM_URL, api_key="sk-no-key")


def load_resize_encode_image(url: str) -> str:
    """
    Download image, fully decode, resize if necessary,
    and return base64-encoded JPEG.
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert("RGB")

    w, h = img.size
    max_side = max(w, h)

    if max_side > MAX_DIM:
        scale = MAX_DIM / max_side
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Force full decode and detach
    img = img.copy()

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return encoded


def extract_focus(question: str) -> str:
    """
    Use the vision-small model (text-only) to extract key visual elements
    from the user's question. Returns a comma-separated string of phrases.
    If extraction fails, returns an empty string.
    """
    instruction = (
        "You are a focus extraction assistant. "
        "Extract the key visual elements from the user's question that should be focused on when describing an image. "
        "Output only a short comma-separated list of phrases. Do not add any explanation.\n\n"
        f"User question: {question}"
    )
    try:
        resp = client.chat.completions.create(
            model="Qwen3-VL",
            messages=[{"role": "user", "content": instruction}],
            max_tokens=50,
            temperature=0.1,
            top_p=0.5,
            presence_penalty=0.0,
            stream=False
        )
        focus_text = resp.choices[0].message.content.strip()
        # Basic cleanup: remove extra spaces, ensure commas separate phrases
        focus_phrases = ", ".join([p.strip() for p in focus_text.split(",") if p.strip()])
        return focus_phrases
    except Exception as e:
        print(f"Focus extraction error: {e}")
        return ""


async def main():
    try:
        await r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise e

    print("Vision Small listening...")

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

            request_id = data.get("id")
            original_prompt = data.get("prompt")          # may be None or empty
            filepath = data.get("filepath")
            timestamp = data.get("timestamp")

            if not request_id:
                print(f"Missing id in message: {data}")
                await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)
                continue

            # Determine if this is a user request (has a non-empty prompt) or background capture
            is_user_request = bool(original_prompt and original_prompt.strip())

            # Prepare base output (will be enriched later)
            out = {
                "id": request_id,
                "prompt": original_prompt,   # keep original for downstream
                "timestamp": timestamp,
            }

            if filepath:
                filename = os.path.basename(filepath)
                url = f"{IMAGE_SERVER_URL}/{filename}"
                print(f"Processing image: {url}")

                try:
                    encoded_image = await asyncio.to_thread(
                        load_resize_encode_image, url
                    )

                    # Step 1: Focus extraction (only for user requests)
                    focus_phrases = ""
                    if is_user_request:
                        focus_phrases = await asyncio.to_thread(extract_focus, original_prompt)
                        if focus_phrases:
                            print(f"Focus phrases: {focus_phrases}")

                    # Step 2: Build description prompt
                    if is_user_request and focus_phrases:
                        description_prompt = (
                            f"Describe this image, focusing on: {focus_phrases}. "
                            f"Original question: {original_prompt}"
                        )
                    else:
                        # Fallback to original prompt if no focus, or use default for background
                        description_prompt = original_prompt if original_prompt else "Describe this image in detail."

                    # Step 3: Generate focused description
                    resp = await asyncio.to_thread(
                        client.chat.completions.create,
                        model="Qwen3-VL",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": description_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encoded_image}"
                                    }
                                }
                            ]
                        }],
                        max_tokens=200,
                        temperature=0.7,
                        top_p=0.8,
                        presence_penalty=1.5,
                        stream=False
                    )

                    description = resp.choices[0].message.content
                    out["description"] = description
                    out["filepath"] = filepath

                    print(f"Generated description for {request_id}")

                except Exception as e:
                    print(f"Inference Error: {e}")
                    out["filepath"] = filepath
                    print(f"Forwarding without description due to error")

            else:
                print(f"No image for {request_id}, forwarding text only")

            await r.xadd(OUTPUT_STREAM, {"data": json.dumps(out)})
            print(f"→ Embed (id: {request_id})")

            await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)

        except Exception as e:
            print(f"VS Error: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
