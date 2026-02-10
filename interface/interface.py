import json
import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import redis.asyncio as redis
from redis.exceptions import ResponseError

app = FastAPI()
r = redis.Redis(host="redis-service", port=6379, decode_responses=True)

INTERFACE_INPUT_STREAM = "stream:interface:input"
INTERFACE_OUTPUT_STREAM = "stream:interface:output"
CONSUMER_GROUP = "interface_broadcast"
CONSUMER_NAME = "interface_worker"

# ---------------- HTML ----------------
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Vision System</title>
        <style>
            body{font-family:sans-serif;background:#222;color:#fff;padding:20px}
            #chat{height:400px;border:1px solid #444;overflow-y:scroll;margin-bottom:10px}
            .msg{padding:5px;border-bottom:1px solid #333}
        </style>
    </head>
    <body>
        <div id="chat"></div>
        <input id="in" style="width:80%"/>
        <button onclick="send()">Send</button>

        <script>
            var ws = new WebSocket("ws://" + location.host + "/ws");

            ws.onopen = () => console.log("WS OPEN");
            ws.onclose = () => console.log("WS CLOSED");
            ws.onerror = e => console.error("WS ERROR", e);

            ws.onmessage = function(e) {
                console.log("WS IN:", e.data);
                var d = JSON.parse(e.data);
                if (d.type === 'response' && d.content) {
                    document.getElementById("chat").innerHTML +=
                        "<div class='msg'>🤖 " + d.content + "</div>";
                }
            };

            function send() {
                var i = document.getElementById("in");
                ws.send(i.value);
                document.getElementById("chat").innerHTML +=
                    "<div class='msg'>👤 " + i.value + "</div>";
                i.value = "";
            }
        </script>
    </body>
</html>
"""

# ---------------- WebSocket Manager ----------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        print(f"WS CONNECTED | total={len(self.active_connections)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)
            print(f"WS DISCONNECTED | total={len(self.active_connections)}")

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_json(msg)
            except Exception as e:
                print("WS SEND FAILED:", e)
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()

# ---------------- Redis ----------------
async def ensure_consumer_group():
    try:
        await r.xgroup_create(
            INTERFACE_OUTPUT_STREAM,
            CONSUMER_GROUP,
            id="0",
            mkstream=True
        )
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise e

async def redis_listener():
    await ensure_consumer_group()
    print("Interface listening for responses...")

    while True:
        try:
            messages = await r.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={INTERFACE_OUTPUT_STREAM: ">"},
                count=10,
                block=2000
            )

            if not messages:
                continue

            for mid, mdata in messages[0][1]:
                raw = json.loads(mdata["data"])

                content = raw.get("content") or raw.get("response")
                if not content:
                    print("⚠️ Dropping empty response:", raw)
                    await r.xack(INTERFACE_OUTPUT_STREAM, CONSUMER_GROUP, mid)
                    continue

                await manager.broadcast({
                    "type": "response",
                    "id": raw.get("id"),
                    "content": content,
                    "timestamp": raw.get("timestamp")
                })

                await r.xack(INTERFACE_OUTPUT_STREAM, CONSUMER_GROUP, mid)

        except Exception as e:
            print(f"Interface Error: {e}")
            await asyncio.sleep(1)

# ---------------- FastAPI ----------------
@app.on_event("startup")
async def startup():
    asyncio.create_task(redis_listener())

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            req_id = str(uuid.uuid4())[:8]
            payload = {
                "id": req_id,
                "prompt": data,
                "timestamp": asyncio.get_event_loop().time()
            }
            await r.xadd(INTERFACE_INPUT_STREAM, {"data": json.dumps(payload)})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
