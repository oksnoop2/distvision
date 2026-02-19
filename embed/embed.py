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
OUTPUT_STREAM = "stream:vision:large:input"
CONSUMER_GROUP = "embed_worker"
CONSUMER_NAME = "worker_1"

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "visual_memory_images")
RETRIEVAL_COUNT = 10
EXPECTED_DIM = 2048  # Qwen3-VL-Embedding dimension

# ---------- Clients ----------
r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
chroma_client = None
collection = None

async def init_chroma():
    global chroma_client, collection
    print(f"Connecting to ChromaDB at {CHROMA_HOST}...")
    chroma_client = await chromadb.AsyncHttpClient(host=CHROMA_HOST, port=8000)
    collection = await chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Chroma collection '{CHROMA_COLLECTION}' ready.")

async def get_image_embedding(image_url: str, text: str = None) -> list[float]:
    # Download image
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

            # ---- Extract embedding ----
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

            # --- New logic for token embeddings ---
            # If embedding is a list of lists (multiple token embeddings), take the last one.
            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                # We have a list of token embeddings; the last token corresponds to [EOS]
                embedding = embedding[-1]
                print(f"Extracted last token embedding, length {len(embedding)}")

            # Ensure it's a flat list of floats
            if not isinstance(embedding, list):
                raise ValueError(f"Embedding is not a list: {type(embedding)}")

            # Verify dimension
            if len(embedding) != EXPECTED_DIM:
                print(f"⚠️ Warning: embedding dimension is {len(embedding)}, expected {EXPECTED_DIM}")

            return embedding

        except Exception as e:
            print(f"Error getting embedding: {e}")
            if 'resp' in locals():
                print(f"Response text: {resp.text}")
            raise

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

            # Get embedding
            embedding = await get_image_embedding(image_url, text=prompt if prompt else None)

            # Store in ChromaDB
            memory_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
            metadata = {
                "timestamp": timestamp,
                "filepath": filepath,
                "has_prompt": bool(prompt)
            }
            await collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[prompt] if prompt else None
            )
            print(f"Indexed image {memory_id}")

            # If user request, retrieve and forward
            if prompt:
                query_embedding = embedding
                results = await collection.query(
                    query_embeddings=[query_embedding],
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

                out = {
                    "id": request_id,
                    "prompt": prompt,
                    "current_image_url": image_url,
                    "past_image_urls": past_image_urls,
                    "timestamp": timestamp
                }

                await r.xadd(OUTPUT_STREAM, {"data": json.dumps(out)})
                print(f"→ Vision Large (id: {request_id}) with {len(past_image_urls)} past images")
            else:
                print(f"Background capture indexed (id: {request_id})")

            await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)

        except Exception as e:
            print(f"Embed Error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
