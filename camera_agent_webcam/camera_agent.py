import cv2
import time
import os
import json
import redis.asyncio as redis
from redis.exceptions import ResponseError # <--- FIX
import asyncio
import uuid
from datetime import datetime

# Config
NODE_NAME = os.getenv("NODE_NAME", "unknown_node")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
IMAGE_DIR = os.getenv("IMAGE_DIR", "/app/images")
BACKGROUND_INTERVAL = int(os.getenv("BACKGROUND_INTERVAL", "60"))

# Streams
CAMERA_REQUEST_STREAM = "stream:camera:requests"
VISION_SMALL_STREAM = "stream:vision:small:input"
VISION_LARGE_STREAM = "stream:vision:large:input"
CAMERA_CONSUMER_GROUP = "camera:agents"
CAMERA_CONSUMER_NAME = f"camera:{NODE_NAME}"

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

async def capture_image(filename_prefix):
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        print(f"[{NODE_NAME}] Error: Camera {CAMERA_INDEX} not found.")
        return None
    cam.read()
    ret, frame = cam.read()
    cam.release()

    if ret:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{NODE_NAME}_{timestamp}.jpg"
        filepath = os.path.join(IMAGE_DIR, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"[{NODE_NAME}] Saved {filepath}")
        return filepath
    return None

async def ensure_consumer_group(stream_name, group_name):
    try:
        await r.xgroup_create(name=stream_name, groupname=group_name, id="0", mkstream=True)
        print(f"[{NODE_NAME}] ✓ Group '{group_name}' ready")
    except ResponseError as e: # <--- FIX
        if "BUSYGROUP" in str(e):
            pass
        else:
            raise e

async def background_loop():
    print(f"[{NODE_NAME}] Starting background loop ({BACKGROUND_INTERVAL}s)")
    while True:
        filepath = await capture_image("bg")
        if filepath:
            msg = {
                "id": str(uuid.uuid4()),
                "prompt": "Describe the environment in detail.",
                "filepath": filepath,
                "type": "background_indexing",
                "timestamp": datetime.now().isoformat(),
                "camera_node": NODE_NAME
            }
            await r.xadd(VISION_SMALL_STREAM, {"data": json.dumps(msg)}, maxlen=1000)
            print(f"[{NODE_NAME}] → Sent to {VISION_SMALL_STREAM}")
        await asyncio.sleep(BACKGROUND_INTERVAL)

async def request_listener():
    await ensure_consumer_group(CAMERA_REQUEST_STREAM, CAMERA_CONSUMER_GROUP)
    print(f"[{NODE_NAME}] Listening for requests...")

    while True:
        try:
            messages = await r.xreadgroup(
                groupname=CAMERA_CONSUMER_GROUP,
                consumername=CAMERA_CONSUMER_NAME,
                streams={CAMERA_REQUEST_STREAM: ">"},
                count=1, block=2000
            )
            if not messages: continue

            mid, mdata = messages[0][1][0]
            data = json.loads(mdata["data"])
            req_id = data.get("req_id")
            prompt = data.get("prompt")

            print(f"[{NODE_NAME}] Handling Request {req_id}")
            filepath = await capture_image(f"req_{req_id}")

            if filepath:
                resp = {
                    "req_id": req_id,
                    "filepath": filepath,
                    "prompt": prompt,
                    "status": "image_ready",
                    "timestamp": datetime.now().isoformat()
                }
                await r.xadd(VISION_LARGE_STREAM, {"data": json.dumps(resp)}, maxlen=1000)
                print(f"[{NODE_NAME}] → Sent to {VISION_LARGE_STREAM}")

            await r.xack(CAMERA_REQUEST_STREAM, CAMERA_CONSUMER_GROUP, mid)
        except Exception as e:
            print(f"[{NODE_NAME}] Listener error: {e}")
            await asyncio.sleep(1)

async def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    await asyncio.gather(background_loop(), request_listener())

if __name__ == "__main__":
    asyncio.run(main())
