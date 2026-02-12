import json
import asyncio
import os
import redis.asyncio as redis
from redis.exceptions import ResponseError
import chromadb
from openai import OpenAI
import httpx

# ---------- Environment ----------
REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma-service")
LLM_URL = os.getenv("LLM_URL", "http://embed-model:8084/v1")
IMAGE_SERVER_URL = os.getenv("IMAGE_SERVER_URL", "http://image-server:8000")

# ---------- Streams & Constants ----------
INPUT_STREAM = "stream:embed:input"
OUTPUT_STREAM = "stream:vision:large:input"
CONSUMER_GROUP = "embed_worker"
CONSUMER_NAME = "worker_1"
CHROMA_COLLECTION = "visual_memory"

# ---------- Clients ----------
r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
client = OpenAI(base_url=LLM_URL, api_key="sk-no-key")


def get_vec(text: str, query: bool = False):
    """Generate embedding using nomic model."""
    prefix = "search_query: " if query else "search_document: "
    resp = client.embeddings.create(
        input=[prefix + text],
        model="nomic"
    )
    return resp.data[0].embedding


async def image_exists(filepath: str) -> bool:
    """Check if image file is accessible on the image server (HEAD request)."""
    filename = os.path.basename(filepath)
    url = f"{IMAGE_SERVER_URL}/{filename}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.head(url, timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False


async def main():
    # ---------- Redis Consumer Group ----------
    try:
        await r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise e

    # ---------- ChromaDB ----------
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
            task = data.get("task")

            # ---------- INDEX ----------
            if task == "index":
                desc = data.get("description")
                if desc:
                    vec = await asyncio.to_thread(get_vec, desc, False)
                    await collection.add(
                        ids=[data.get("id")],
                        embeddings=[vec],
                        documents=[desc],
                        metadatas=[{
                            "timestamp": data.get("timestamp"),
                            "filepath": data.get("filepath")
                        }]
                    )
                    print(f"Indexed: {data.get('id')}")

            # ---------- RETRIEVE & GENERATE ----------
            elif task == "retrieve_and_generate":
                prompt = data.get("prompt")
                timestamp = data.get("timestamp")
                req_id = data.get("req_id")

                # 1. Vector search (top 5 semantically similar)
                vec = await asyncio.to_thread(get_vec, prompt, True)
                results = await collection.query(
                    query_embeddings=[vec],
                    n_results=5
                )

                # 2. Extract metadata and documents
                metadatas = results.get("metadatas", [[]])[0]
                documents = results.get("documents", [[]])[0]

                # 3. Build full text context (used for fallback)
                context_lines = []
                valid_entries = []  # (timestamp, filepath, doc, meta)

                for meta, doc in zip(metadatas, documents):
                    ts = meta.get("timestamp")
                    fp = meta.get("filepath")
                    if ts and fp:  # only entries with complete metadata
                        context_lines.append(f"- [{ts}] {doc}")
                        valid_entries.append((ts, fp, doc, meta))

                full_context = "MEMORY CONTEXT:\n" + "\n".join(context_lines)

                # 4. Concurrently check which images still exist
                if valid_entries:
                    check_tasks = [image_exists(fp) for (_, fp, _, _) in valid_entries]
                    existence_results = await asyncio.gather(*check_tasks)
                else:
                    existence_results = []

                # 5. Filter entries where image exists
                existing_entries = [
                    entry for entry, exists in zip(valid_entries, existence_results) if exists
                ]

                # 6. Dispatch: image + prompt OR text only
                if existing_entries:
                    # Sort by timestamp (most recent first)
                    existing_entries.sort(key=lambda x: x[0], reverse=True)
                    most_recent_ts, most_recent_fp, _, _ = existing_entries[0]

                    msg = {
                        "req_id": req_id,
                        "filepath": most_recent_fp,
                        "prompt": prompt,
                        "status": "image_ready",
                        "timestamp": timestamp   # original request timestamp
                    }
                    await r.xadd(OUTPUT_STREAM, {"data": json.dumps(msg)})
                    print(f"→ Vision Large (image + prompt): {req_id}")

                else:
                    # No surviving images → send text‑only with full context
                    msg = {
                        "req_id": req_id,
                        "prompt": prompt,
                        "context_text": full_context,
                        "type": "text_response",
                        "timestamp": timestamp
                    }
                    await r.xadd(OUTPUT_STREAM, {"data": json.dumps(msg)})
                    print(f"→ Vision Large (text only): {req_id}")

            # ---------- Acknowledge ----------
            await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)

        except Exception as e:
            print(f"Embed Error: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
