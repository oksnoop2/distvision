import json
import asyncio
import os
import uuid
import redis.asyncio as redis
from redis.exceptions import ResponseError
import chromadb
from openai import OpenAI

# ---------- Environment ----------
REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma-service")
LLM_URL = os.getenv("LLM_URL", "http://embed-model:8084/v1")

# ---------- Streams & Constants ----------
INPUT_STREAM = "stream:embed:input"
OUTPUT_STREAM = "stream:vision:large:input"
CONSUMER_GROUP = "embed_worker"
CONSUMER_NAME = "worker_1"
CHROMA_COLLECTION = "visual_memory"
RETRIEVAL_COUNT = 10

# ---------- Clients ----------
r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
client = OpenAI(base_url=LLM_URL, api_key="sk-no-key")

def get_vec(text: str, query: bool = False):
    prefix = "search_query: " if query else "search_document: "
    resp = client.embeddings.create(
        input=[prefix + text],
        model="nomic"
    )
    return resp.data[0].embedding

async def main():
    try:
        await r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise e

    print(f"Connecting to ChromaDB at {CHROMA_HOST}...")
    chroma_client = await chromadb.AsyncHttpClient(host=CHROMA_HOST, port=8000)
    collection = await chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)
    print("Embed listening...")

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
            prompt = data.get("prompt")           # may be None for background
            description = data.get("description") # always present when image was processed
            filepath = data.get("filepath")
            timestamp = data.get("timestamp")

            # Only require ID; prompt can be absent
            if not request_id:
                print(f"⚠️ Missing id in message: {data}")
                await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)
                continue

            # ---------- 1. Always index the description if it exists ----------
            if description:
                vec = await asyncio.to_thread(get_vec, description, False)
                memory_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
                await collection.add(
                    ids=[memory_id],
                    embeddings=[vec],
                    documents=[description],
                    metadatas=[{
                        "timestamp": timestamp,
                        "filepath": filepath
                    }]
                )
                print(f"Indexed memory {memory_id}")

            # ---------- 2. Only do retrieval & answer for user requests (prompt exists) ----------
            if prompt:
                # Use description if available, otherwise fallback to prompt for query embedding
                query_text = description if description else prompt
                vec_query = await asyncio.to_thread(get_vec, query_text, True)
                results = await collection.query(
                    query_embeddings=[vec_query],
                    n_results=RETRIEVAL_COUNT
                )

                metadatas = results.get("metadatas", [[]])[0]
                documents = results.get("documents", [[]])[0]

                context_parts = []
                if description:
                    context_parts.append(f"[CURRENT] {description}")

                for idx, (meta, doc) in enumerate(zip(metadatas, documents), start=1):
                    ts = meta.get("timestamp", "unknown")
                    context_parts.append(f"[{idx}] {ts}: {doc}")

                full_context = "\n".join(context_parts) if context_parts else "(No context available)"

                msg = {
                    "id": request_id,
                    "prompt": prompt,
                    "full_context": full_context,
                    "timestamp": timestamp
                }
                if filepath:
                    msg["filepath"] = filepath

                await r.xadd(OUTPUT_STREAM, {"data": json.dumps(msg)})
                print(f"→ Vision Large (id: {request_id})")
            else:
                # Background capture: indexed but no answer needed
                print(f"Background capture indexed (id: {request_id}) – skipping vision_large")

            # ---------- Acknowledge ----------
            await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)

        except Exception as e:
            print(f"Embed Error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
