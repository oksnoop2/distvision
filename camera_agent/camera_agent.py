import cv2
import time
import os
import json
import redis.asyncio as redis
from redis.exceptions import ResponseError
import asyncio
import uuid
from datetime import datetime

# Config
NODE_NAME = os.getenv("NODE_NAME", "unknown_node")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
IMAGE_DIR = os.getenv("IMAGE_DIR", "/app/images")
BACKGROUND_INTERVAL = int(os.getenv("BACKGROUND_INTERVAL", "60"))

# Thingino RTSP Configuration
RTSP_URLS = os.getenv("RTSP_URLS", "rtsp://thingino:thingino@192.168.1.11/ch1,rtsp://thingino:thingino@192.168.1.11/ch0").split(",")
RTSP_CROP_TOP = int(os.getenv("RTSP_CROP_TOP", "55"))
RTSP_THROWAWAY_FRAMES = int(os.getenv("RTSP_THROWAWAY_FRAMES", "5"))
RTSP_TIMEOUT_MS = int(os.getenv("RTSP_TIMEOUT_MS", "10000"))
ENABLE_WEBCAM_FALLBACK = os.getenv("ENABLE_WEBCAM_FALLBACK", "true").lower() == "true"

# Streams
CAMERA_REQUEST_STREAM = "stream:camera:requests"
CAMERA_RAW_STREAM = "stream:camera:raw"          # new raw stream for detector
# EMBED_STREAM is no longer used directly; kept for reference
EMBED_STREAM = "stream:embed:input"
CAMERA_CONSUMER_GROUP = "camera:agents"
CAMERA_CONSUMER_NAME = f"camera:{NODE_NAME}"

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

async def capture_image(filename_prefix):


    # Try each RTSP URL in order
    for rtsp_url in RTSP_URLS:
        if not rtsp_url.strip():
            continue

        rtsp_url = rtsp_url.strip()
        print(f"[{NODE_NAME}] Attempting RTSP capture from {rtsp_url}")

        try:
            # Open RTSP stream - try different approaches
            cap = cv2.VideoCapture(rtsp_url)

            # Check if we can set RTSP transport properties
            try:
                # Method 1: Try to set RTSP transport to TCP (if available)
                cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, 1)  # TCP transport
                print(f"[{NODE_NAME}] Using TCP transport for RTSP")
            except AttributeError:
                print(f"[{NODE_NAME}] Note: CAP_PROP_RTSP_TRANSPORT not available")
                # Continue anyway - the stream might work without it

            try:
                # Try to set buffer size (if available)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except AttributeError:
                pass

            if not cap.isOpened():
                print(f"[{NODE_NAME}] Failed to open RTSP stream: {rtsp_url}")
                continue

            # Allow stream to stabilize by reading throwaway frames
            for i in range(RTSP_THROWAWAY_FRAMES):
                cap.grab()
                if i == 0:  # Small delay after first grab
                    await asyncio.sleep(0.1)

            # Read the actual frame
            ret, frame = cap.read()
            cap.release()  # Release immediately after reading

            if not ret:
                print(f"[{NODE_NAME}] Failed to read frame from RTSP: {rtsp_url}")
                continue


            if frame is not None and frame.size > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename_prefix}_{NODE_NAME}_{timestamp}.jpg"
                filepath = os.path.join(IMAGE_DIR, filename)

                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                print(f"[{NODE_NAME}] Saved {filepath} (Size: {frame.shape[1]}x{frame.shape[0]})")
                return filepath

        except Exception as e:
            print(f"[{NODE_NAME}] RTSP error with {rtsp_url}: {e}")
            continue

    # RTSP failed, try webcam fallback if enabled
    if ENABLE_WEBCAM_FALLBACK:
        print(f"[{NODE_NAME}] RTSP failed, falling back to webcam index {CAMERA_INDEX}")
        return await capture_webcam_image(filename_prefix)

    print(f"[{NODE_NAME}] All capture methods failed")
    return None

async def capture_webcam_image(filename_prefix):
    """Fallback to webcam capture (original logic)"""
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        print(f"[{NODE_NAME}] Error: Webcam {CAMERA_INDEX} not found.")
        return None

    # Clear initial buffer
    cam.read()
    ret, frame = cam.read()
    cam.release()

    if ret:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{NODE_NAME}_{timestamp}.jpg"
        filepath = os.path.join(IMAGE_DIR, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"[{NODE_NAME}] Saved webcam image: {filepath}")
        return filepath

    return None

async def ensure_consumer_group(stream_name, group_name):
    try:
        await r.xgroup_create(name=stream_name, groupname=group_name, id="0", mkstream=True)
        print(f"[{NODE_NAME}] ✓ Group '{group_name}' ready")
    except ResponseError as e:
        if "BUSYGROUP" in str(e):
            pass
        else:
            raise e

async def background_loop():
    print(f"[{NODE_NAME}] Starting background loop ({BACKGROUND_INTERVAL}s)")
    print(f"[{NODE_NAME}] RTSP URLs: {RTSP_URLS}")
    print(f"[{NODE_NAME}] Webcam fallback: {ENABLE_WEBCAM_FALLBACK}")

    while True:
        filepath = await capture_image("bg")
        if filepath:
            msg = {
                "id": str(uuid.uuid4()),
                "filepath": filepath,
                "timestamp": datetime.now().isoformat(),
                "camera_node": NODE_NAME          # optional – keep for debugging, or remove
            }
            # Publish to raw stream for detector
            await r.xadd(CAMERA_RAW_STREAM, {"data": json.dumps(msg)}, maxlen=1000)
            print(f"[{NODE_NAME}] → Sent to {CAMERA_RAW_STREAM}")
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
            if not messages:
                continue

            mid, mdata = messages[0][1][0]
            data = json.loads(mdata["data"])
            request_id = data.get("id")
            prompt = data.get("prompt", "")
            original_timestamp = data.get("timestamp")  # preserve original timestamp

            print(f"[{NODE_NAME}] Handling Request {request_id}")
            filepath = await capture_image(f"req_{request_id}")

            # Always forward the request, even if capture failed
            resp = {
                "id": request_id,
                "prompt": prompt or "",   # ensure it's a string
                "timestamp": original_timestamp,
            }

            if filepath:
                resp["filepath"] = filepath

            # Publish to raw stream for detector
            await r.xadd(CAMERA_RAW_STREAM, {"data": json.dumps(resp)}, maxlen=1000)
            print(f"[{NODE_NAME}] → Sent to {CAMERA_RAW_STREAM}" + (" (with image)" if filepath else " (text only)"))

            await r.xack(CAMERA_REQUEST_STREAM, CAMERA_CONSUMER_GROUP, mid)
        except Exception as e:
            print(f"[{NODE_NAME}] Listener error: {e}")
            await asyncio.sleep(1)

async def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Initial camera test on startup
    print(f"[{NODE_NAME}] Testing camera connection...")
    test_file = await capture_image("test_init")
    if test_file:
        print(f"[{NODE_NAME}] Camera test successful: {test_file}")
    else:
        print(f"[{NODE_NAME}] WARNING: Camera test failed, system will still run but may use webcam")

    await asyncio.gather(background_loop(), request_listener())

if __name__ == "__main__":
    asyncio.run(main())
