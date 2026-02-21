import json
import asyncio
import os
import base64
import httpx
import redis.asyncio as redis
from redis.exceptions import ResponseError

REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8086")
IMAGE_SERVER_URL = os.getenv("IMAGE_SERVER_URL", "http://image-server:8000")

INPUT_STREAM = "stream:rerank:input"
OUTPUT_STREAM = "stream:vision:large:input"
STATUS_STREAM = "stream:interface:status"
CONSUMER_GROUP = "reranker_worker"
CONSUMER_NAME = "worker_1"

http_client = httpx.AsyncClient(timeout=30.0)
r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)


async def download_image_base64(image_url: str) -> str | None:
    """Download image and return base64 string, or None on failure."""
    try:
        resp = await http_client.get(image_url)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return None


async def call_reranker(query: str, documents: list[dict]) -> list[float]:
    """
    Call the llama.cpp reranker endpoint.
    Expected payload: {"query": "...", "documents": ["doc1", "doc2", ...]}
    Returns a list of scores in the same order as documents.
    """
    # Extract text from each document (empty string if none)
    doc_texts = [doc.get("text", "") for doc in documents]

    payload = {
        "query": query,
        "documents": doc_texts
    }

    try:
        # Try /rerank (common endpoint)
        resp = await http_client.post(f"{RERANKER_URL}/rerank", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "scores" in data:
            return data["scores"]
        else:
            print(f"Unexpected response: {data}")
            return [0.0] * len(documents)
    except Exception as e:
        print(f"Reranker error: {e}")
        return [0.0] * len(documents)


async def main():
    try:
        await r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    print("Reranker worker listening...")

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

            request_id = data["id"]
            prompt = data["prompt"]
            current_image_url = data["current_image_url"]
            past_image_urls = data["past_image_urls"]
            past_metadatas = data["past_metadatas"]
            past_conversations = data["past_conversations"]
            timestamp = data["timestamp"]

            # If there are no past images, forward directly
            if not past_image_urls:
                out = {
                    "id": request_id,
                    "prompt": prompt,
                    "current_image_url": current_image_url,
                    "past_image_urls": past_image_urls,
                    "past_metadatas": past_metadatas,
                    "past_conversations": past_conversations,
                    "timestamp": timestamp
                }
                await r.xadd(OUTPUT_STREAM, {"data": json.dumps(out)})
                await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)
                continue

            # Build documents for reranker (only text from metadata, if any)
            documents = []
            for meta in past_metadatas:
                # The metadata may contain a stored prompt under 'document' field
                text = meta.get("document", "")
                documents.append({"text": text})

            # Get scores from reranker
            scores = await call_reranker(prompt, documents)

            # If all scores are zero (e.g., error or no text), keep original order
            if all(s == 0.0 for s in scores):
                sorted_urls = past_image_urls
                sorted_metadatas = past_metadatas
            else:
                # Sort by score descending
                combined = sorted(zip(scores, past_image_urls, past_metadatas), key=lambda x: x[0], reverse=True)
                sorted_urls = [item[1] for item in combined]
                sorted_metadatas = [item[2] for item in combined]

            # Send status update to interface (optional)
            status_msg = {
                "type": "image_update",
                "request_id": request_id,
                "container": "reranker",
                "current_image_url": current_image_url,
                "past_image_urls": sorted_urls[:3],
                "timestamp": timestamp
            }
            await r.xadd(STATUS_STREAM, {"data": json.dumps(status_msg)})

            # Forward to vision large
            out = {
                "id": request_id,
                "prompt": prompt,
                "current_image_url": current_image_url,
                "past_image_urls": sorted_urls,
                "past_metadatas": sorted_metadatas,
                "past_conversations": past_conversations,
                "timestamp": timestamp
            }
            await r.xadd(OUTPUT_STREAM, {"data": json.dumps(out)})
            print(f"Reranked {len(sorted_urls)} images for {request_id} (scores: {scores})")

            await r.xack(INPUT_STREAM, CONSUMER_GROUP, mid)

        except Exception as e:
            print(f"Reranker worker error: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(http_client.aclose())
