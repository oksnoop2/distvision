import json
import asyncio
import os
import redis.asyncio as redis
from redis.exceptions import ResponseError
from openai import OpenAI

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
# Point to the router-model container defined in compose.yaml
LLM_URL = os.getenv("LLM_URL", "http://router-model:8081/v1")

# Constants for Redis Streams
INTERFACE_INPUT_STREAM = "stream:interface:input"
CAMERA_COMMAND_STREAM = "stream:camera:requests"
EMBED_QUEUE_STREAM = "stream:embed:input"

ROUTER_CONSUMER_GROUP = "router_group"
ROUTER_CONSUMER_NAME = "router_consumer_1"

# Setup Redis
r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

# Setup OpenAI Client (to talk to the model server)
client = OpenAI(base_url=LLM_URL, api_key="sk-no-key-required")

async def needs_eyes(user_prompt):
    """
    Asks the model if the prompt requires visual input.
    Returns: True (YES) or False (NO)
    """
    sys_msg = (
        "You are a routing system. Does the user prompt require looking at a LIVE camera feed RIGHT NOW? "
        "Examples: 'What is this?' -> YES. 'Where are my keys?' -> NO. 'Look at the table' -> YES.\n"
        "Respond ONLY with 'YES' or 'NO'."
    )

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="router",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_prompt}
            ],
                max_tokens=5,
                temperature=0.1,  # Low temperature for deterministic routing
                top_p=0.9,
        stream=False
    )
        decision = response.choices[0].message.content.strip().upper()
        return "YES" in decision
    except Exception as e:
        print(f" [!] Inference Error: {e}")
        # Default to False on error to be safe
        return False

async def ensure_consumer_group():
    try:
        await r.xgroup_create(
            name=INTERFACE_INPUT_STREAM,
            groupname=ROUTER_CONSUMER_GROUP,
            id="0",
            mkstream=True
        )
        print(f"✓ Consumer Group Ready: {ROUTER_CONSUMER_GROUP}")
    except ResponseError as e:
        if "BUSYGROUP" in str(e):
            pass
        else:
            raise e

async def process_incoming():
    print(f"--- [Router] Listening on {INTERFACE_INPUT_STREAM} ---")
    await ensure_consumer_group()

    last_id = ">"

    while True:
        try:
            # Block for 2 seconds
            messages = await r.xreadgroup(
                groupname=ROUTER_CONSUMER_GROUP,
                consumername=ROUTER_CONSUMER_NAME,
                streams={INTERFACE_INPUT_STREAM: last_id},
                count=1,
                block=2000
            )

            if not messages:
                continue

            stream_name, message_list = messages[0]
            message_id, message_data = message_list[0]

            # Parse Data
            raw_msg = message_data.get("data")
            if not raw_msg:
                # Handle edge case where data might not be wrapped
                # (though interface.py wraps it in "data")
                raw_msg = json.dumps(message_data)

            try:
                if isinstance(raw_msg, str):
                    data = json.loads(raw_msg)
                else:
                    data = raw_msg

                req_id = data.get("id") or data.get("req_id")
                prompt = data.get("prompt")

                print(f" [?] Analyzing: {prompt[:30]}...")

                # --- INFERENCE ---
                is_visual_request = await needs_eyes(prompt)

                if is_visual_request:
                    # --- ROUTE TO CAMERA ---
                    cmd_payload = {
                        "req_id": req_id,
                        "prompt": prompt,
                        "cmd": "capture"
                    }
                    # Wrap in "data" to match other services
                    await r.xadd(CAMERA_COMMAND_STREAM, {"data": json.dumps(cmd_payload)})
                    print(f"     -> [EYES] Added to {CAMERA_COMMAND_STREAM}")

                else:
                    # --- ROUTE TO EMBED (MEMORY) ---
                    embed_payload = {
                        "req_id": req_id,
                        "prompt": prompt,
                        "task": "retrieve_and_generate"
                    }
                    await r.xadd(EMBED_QUEUE_STREAM, {"data": json.dumps(embed_payload)})
                    print(f"     -> [BRAIN] Added to {EMBED_QUEUE_STREAM}")

                # ACK
                await r.xack(INTERFACE_INPUT_STREAM, ROUTER_CONSUMER_GROUP, message_id)

            except Exception as e:
                print(f" [!] Processing Error {message_id}: {e}")
                await r.xack(INTERFACE_INPUT_STREAM, ROUTER_CONSUMER_GROUP, message_id)

        except Exception as e:
            print(f" [!] Connection Loop Error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(process_incoming())
