import json
import asyncio
import os
import redis.asyncio as redis
from redis.exceptions import ResponseError
import chromadb
from openai import OpenAI

REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma-service")
LLM_URL = os.getenv("LLM_URL", "http://embed-model:8084/v1")

INPUT_STREAM = "stream:embed:input"
OUTPUT_STREAM = "stream:vision:large:input"
CONSUMER_GROUP = "embed_worker"
CONSUMER_NAME = "worker_1"
CHROMA_COLLECTION = "visual_memory"

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
client = OpenAI(base_url=LLM_URL, api_key="sk-no-key")


def get_vec(text, query=False):
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

            # ---------- RETRIEVE ----------
            elif task == "retrieve_and_generate":
                prompt = data.get("prompt")
                timestamp = data.get("timestamp")

                vec = await asyncio.to_thread(get_vec, prompt, True)
                results = await collection.query(
                    query_embeddings=[vec],
                    n_results=5
                )

                context = "MEMORY CONTEXT:\n"
                if results.get("documents"):
                    for doc, meta in zip(
                        results["documents"][0],
                        results["metadatas"][0]
                    ):
                        context += f"- [{meta.get('timestamp')}] {doc}\n"

                msg = {
                    "req_id": data.get("req_id"),
                    "prompt": prompt,
                    "context_text": context,
                    "type": "text_response",
                    "timestamp": timestamp
                }

                await r.xadd(OUTPUT_STREAM, {"data": json.dumps(msg)})
                print(f"→ Vision Large (memory): {data.get('req_id')}")

            await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)

        except Exception as e:
            print(f"Embed Error: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
