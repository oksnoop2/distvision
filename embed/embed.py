import json
import asyncio
import os
import base64
import uuid
from datetime import datetime
import httpx
import redis.asyncio as redis
from redis.exceptions import ResponseError
import chromadb

# ---------- Environment ----------
REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma-service")
EMBED_URL = os.getenv("EMBED_LLM_URL", "http://embed-model:8085")
print(f"Using EMBED_URL: {EMBED_URL}")
IMAGE_SERVER_URL = os.getenv("IMAGE_SERVER_URL", "http://image-server:8000")

# ---------- Streams ----------
INPUT_STREAM = "stream:embed:input"
OUTPUT_STREAM = "stream:rerank:input"
STATUS_STREAM = "stream:interface:status"          # for real‑time previews
CONSUMER_GROUP = "embed_worker"
CONSUMER_NAME = "worker_1"

# Chroma collections
IMAGE_COLLECTION = os.getenv("IMAGE_COLLECTION", "visual_memory_images")
CONVERSATION_COLLECTION = os.getenv("CONVERSATION_COLLECTION", "conversation_memory")

# Retrieval settings
RETRIEVAL_COUNT = 10
EXPECTED_DIM = 2048  # Qwen3-VL-Embedding dimension

# ---------- Clients ----------
r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
chroma_client = None
image_collection = None
conv_collection = None

async def init_chroma():
    global chroma_client, image_collection, conv_collection
    print(f"Connecting to ChromaDB at {CHROMA_HOST}...")
    chroma_client = await chromadb.AsyncHttpClient(host=CHROMA_HOST, port=8000)

    # Image collection
    image_collection = await chroma_client.get_or_create_collection(
        name=IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Image collection '{IMAGE_COLLECTION}' ready.")

    # Conversation collection
    conv_collection = await chroma_client.get_or_create_collection(
        name=CONVERSATION_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Conversation collection '{CONVERSATION_COLLECTION}' ready.")

async def get_image_embedding(image_url: str, text: str = None) -> list[float]:
    """Compute embedding for an image (with optional text)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(image_url, timeout=10)
        resp.raise_for_status()
        img_base64 = base64.b64encode(resp.content).decode("utf-8")

    # Build prompt
    if text:
        prompt = f"<image>\nRepresent this image and text for retrieval: {text}"
    else:
        prompt = "<image>\nRepresent this image."

    # Call embedding server
    async with httpx.AsyncClient() as client:
        payload = {
            "content": prompt,
            "image_data": [{"data": img_base64, "id": 0}]
        }
        try:
            resp = await client.post(f"{EMBED_URL}/embedding", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return _extract_embedding(data)
        except Exception as e:
            print(f"Error getting image embedding: {e}")
            if 'resp' in locals():
                print(f"Response text: {resp.text}")
            raise

async def get_text_embedding(text: str) -> list[float]:
    """Compute embedding for pure text (no image)."""
    async with httpx.AsyncClient() as client:
        payload = {"content": text}
        try:
            resp = await client.post(f"{EMBED_URL}/embedding", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return _extract_embedding(data)
        except Exception as e:
            print(f"Error getting text embedding: {e}")
            if 'resp' in locals():
                print(f"Response text: {resp.text}")
            raise

def _extract_embedding(data) -> list[float]:
    """Common embedding extraction logic (shared by image and text)."""
    embedding = None

    # Case 1: list of results (some servers return list directly)
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, dict) and "embedding" in first:
            embedding = first["embedding"]
        elif isinstance(first, list):
            embedding = first
        else:
            # Fallback: maybe the whole list is the embedding (if length matches)
            if len(data) == EXPECTED_DIM:
                embedding = data
            else:
                raise ValueError(f"Unexpected list format: {type(first)}")

    # Case 2: direct dict with embedding (llama.cpp server style)
    elif isinstance(data, dict) and "embedding" in data:
        embedding = data["embedding"]

    if embedding is None:
        raise ValueError(f"Could not extract embedding from response: {data}")

    # If embedding is a list of lists (multiple token embeddings), take the last one.
    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        embedding = embedding[-1]
        print(f"Extracted last token embedding, length {len(embedding)}")

    if not isinstance(embedding, list):
        raise ValueError(f"Embedding is not a list: {type(embedding)}")

    if len(embedding) != EXPECTED_DIM:
        print(f"⚠️ Warning: embedding dimension is {len(embedding)}, expected {EXPECTED_DIM}")

    return embedding

async def main():
    await init_chroma()

    try:
        await r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise e

    print("Embed (Qwen) listening...")

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

            # ---------- Handle conversation storage ----------
            if data.get("type") == "conversation":
                # Expected fields: user, assistant, timestamp, (optional image_url)
                user_text = data["user"]
                assistant_text = data["assistant"]
                timestamp = data["timestamp"]
                combined = f"User: {user_text}\nAssistant: {assistant_text}"

                embedding = await get_text_embedding(combined)

                # Store in conversation collection
                memory_id = f"conv_{timestamp}_{uuid.uuid4().hex[:8]}"
                metadata = {
                    "timestamp": timestamp,
                    "user": user_text,
                    "assistant": assistant_text,
                    "image_url": data.get("image_url", "")
                }
                await conv_collection.add(
                    ids=[memory_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[combined]
                )
                print(f"Stored conversation {memory_id}")

                await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)
                continue  # done with this message

            # ---------- Handle image (background capture or user request) ----------
            request_id = data.get("id")
            prompt = data.get("prompt")
            filepath = data.get("filepath")
            timestamp = data.get("timestamp")

            if not request_id or not filepath:
                print(f"Missing id or filepath in message: {data}")
                await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)
                continue

            filename = os.path.basename(filepath)
            image_url = f"{IMAGE_SERVER_URL}/{filename}"

            # Compute image embedding
            embedding = await get_image_embedding(image_url, text=prompt if prompt else None)

            # Store in image collection
            memory_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
            metadata = {
                "timestamp": timestamp,
                "filepath": filepath,
                "has_prompt": bool(prompt)
            }
            await image_collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[prompt] if prompt else None
            )
            print(f"Indexed image {memory_id}")

            # If user request, retrieve relevant past images AND past conversations
            if prompt:
                # --- Image retrieval (existing) ---
                results = await image_collection.query(
                    query_embeddings=[embedding],
                    n_results=RETRIEVAL_COUNT,
                    where={"timestamp": {"$ne": timestamp}} if timestamp else None
                )
                past_image_urls = []
                metadatas = results.get("metadatas", [[]])[0]
                for meta in metadatas:
                    past_path = meta.get("filepath")
                    if past_path:
                        past_filename = os.path.basename(past_path)
                        past_image_urls.append(f"{IMAGE_SERVER_URL}/{past_filename}")

                # --- Send image preview to interface (NEW) ---
                status_msg = {
                    "type": "image_update",
                    "request_id": request_id,
                    "container": "embed",
                    "current_image_url": image_url,
                    "past_image_urls": past_image_urls[:3],   # limit to 3 for display
                    "timestamp": timestamp
                }
                await r.xadd(STATUS_STREAM, {"data": json.dumps(status_msg)})
                print(f"Sent status update for {request_id}")

                # --- Conversation retrieval (new) ---
                # Compute text embedding of the user prompt
                text_embedding = await get_text_embedding(prompt)
                conv_results = await conv_collection.query(
                    query_embeddings=[text_embedding],
                    n_results=5,  # retrieve up to 5 relevant past exchanges
                )
                past_conversations = []
                conv_metadatas = conv_results.get("metadatas", [[]])[0]
                for meta in conv_metadatas:
                    past_conversations.append({
                        "user": meta.get("user"),
                        "assistant": meta.get("assistant"),
                        "timestamp": meta.get("timestamp")
                    })

                # Build output message for reranker
                out = {
                    "id": request_id,
                    "prompt": prompt,
                    "current_image_url": image_url,
                    "past_image_urls": past_image_urls,
                    "past_metadatas": metadatas,
                    "past_conversations": past_conversations,
                    "timestamp": timestamp
                }

                await r.xadd(OUTPUT_STREAM, {"data": json.dumps(out)})
                print(f"→ Reranker (id: {request_id}) with {len(past_image_urls)} past images and {len(past_conversations)} past conversations")
            else:
                print(f"Background capture indexed (id: {request_id})")

            await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)

        except Exception as e:
            print(f"Embed Error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
