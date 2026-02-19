import json
import asyncio
import uuid
import os
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import redis.asyncio as redis
from redis.exceptions import ResponseError

app = FastAPI()
r = redis.Redis(host="redis-service", port=6379, decode_responses=True)

CAMERA_REQUEST_STREAM = "stream:camera:requests"
INTERFACE_OUTPUT_STREAM = "stream:interface:output"
INTERFACE_STATUS_STREAM = "stream:interface:status"
CONSUMER_GROUP = "interface_broadcast"
CONSUMER_NAME = "interface_worker"

CHAT_HISTORY_KEY = "chat:history"
MAX_HISTORY = 100

# Image server URL for browser (external)
EXTERNAL_IMAGE_URL = os.getenv("EXTERNAL_IMAGE_URL", "http://localhost:8090")

# ---------------- HTML----------------
html = """
<!DOCTYPE html>
<html>
<head>
    <title>💫 CYBER DREAM</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Avenir Next', 'Segoe UI', system-ui, sans-serif;
        }
        body {
            background: linear-gradient(145deg, #0b0b1a 0%, #1a0f2a 100%);
            color: #e0b0ff;
            display: flex;
            height: 100vh;
            overflow: hidden;
            backdrop-filter: blur(2px);
        }
        #main {
            display: flex;
            flex: 1;
            padding: 20px;
            gap: 20px;
        }
        /* Left: Chat area - glossy curved panel */
        #chat-container {
            flex: 2;
            display: flex;
            flex-direction: column;
            background: rgba(20, 10, 30, 0.6);
            border: 1px solid rgba(255, 120, 200, 0.5);
            border-radius: 40px 40px 20px 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.6), 0 0 0 2px rgba(255, 120, 200, 0.3) inset, 0 0 30px #ff80c0;
            backdrop-filter: blur(12px);
            padding: 25px;
        }
        #chat-header {
            border-bottom: 2px solid #ff99cc;
            padding-bottom: 15px;
            margin-bottom: 20px;
            font-size: 1.8em;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 4px;
            color: #ffe0f0;
            text-shadow: 0 0 8px #ff66b2, 0 0 20px #c080ff;
        }
        #chat {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 30px;
            scrollbar-width: thin;
            scrollbar-color: #ff99cc #330033;
        }
        #chat::-webkit-scrollbar {
            width: 6px;
        }
        #chat::-webkit-scrollbar-track {
            background: #330033;
            border-radius: 10px;
        }
        #chat::-webkit-scrollbar-thumb {
            background: #ff99cc;
            border-radius: 10px;
        }
        .msg {
            margin: 15px 0;
            padding: 12px 18px;
            border-radius: 30px 30px 30px 10px;
            background: rgba(255, 200, 240, 0.1);
            border: 1px solid rgba(255, 150, 200, 0.5);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            max-width: 85%;
            width: fit-content;
            backdrop-filter: blur(4px);
            font-size: 1.1em;
        }
        .msg.user {
            margin-left: auto;
            background: rgba(180, 130, 255, 0.2);
            border-color: #ac8cff;
            border-radius: 30px 30px 10px 30px;
            text-align: right;
        }
        .input-area {
            display: flex;
            margin-top: 20px;
            gap: 15px;
        }
        #in {
            flex: 1;
            padding: 18px 25px;
            background: rgba(10, 5, 20, 0.7);
            border: 2px solid #ff80bf;
            border-radius: 60px;
            color: #ffe6ff;
            font-size: 1.1em;
            backdrop-filter: blur(8px);
            box-shadow: 0 0 15px #ff80bf, inset 0 0 8px #b380ff;
        }
        #in:focus {
            outline: none;
            border-color: #ffb3ff;
            box-shadow: 0 0 25px #ffb3ff, inset 0 0 10px #d9b3ff;
        }
        button {
            padding: 18px 35px;
            background: linear-gradient(145deg, #ff80bf, #b380ff);
            border: none;
            border-radius: 60px;
            color: #0d0d1a;
            font-weight: bold;
            font-size: 1.2em;
            text-transform: uppercase;
            cursor: pointer;
            box-shadow: 0 8px 0 #4d3366, 0 10px 20px rgba(0,0,0,0.4);
            transition: 0.1s ease;
            letter-spacing: 1px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 0 #4d3366, 0 15px 25px #ffb3ff;
        }
        button:active {
            transform: translateY(8px);
            box-shadow: 0 2px 0 #4d3366;
        }

        /* Right panel - Wet Cyber Status */
        #status-panel {
            flex: 1;
            background: rgba(20, 10, 30, 0.6);
            border: 1px solid #80b3ff;
            border-radius: 40px 40px 20px 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.6), 0 0 0 2px rgba(128, 180, 255, 0.3) inset, 0 0 30px #80b3ff;
            backdrop-filter: blur(12px);
            padding: 25px;
            display: flex;
            flex-direction: column;
            gap: 25px;
        }
        #panel-header {
            border-bottom: 2px solid #99ccff;
            padding-bottom: 15px;
            font-size: 1.6em;
            font-weight: 400;
            text-transform: uppercase;
            color: #cce4ff;
            text-shadow: 0 0 8px #66aaff, 0 0 20px #bf80ff;
            letter-spacing: 3px;
        }
        #last-container {
            background: rgba(0,0,0,0.3);
            border: 1px solid #c3a0ff;
            border-radius: 40px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(8px);
        }
        #last-container div:first-child {
            font-size: 1em;
            color: #d9b3ff;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        #container-value {
            font-size: 2.2em;
            font-weight: bold;
            color: #ffb3ec;
            text-transform: uppercase;
            text-shadow: 0 0 15px #ff80bf, 0 0 30px #b380ff;
            animation: softPulse 2s infinite;
        }
        @keyframes softPulse {
            0% { opacity: 0.9; text-shadow: 0 0 10px #ff80bf; }
            50% { opacity: 1; text-shadow: 0 0 25px #ffb3ff, 0 0 40px #c080ff; }
            100% { opacity: 0.9; text-shadow: 0 0 10px #ff80bf; }
        }
        #image-preview {
            background: rgba(0,0,0,0.3);
            border: 1px solid #ff99cc;
            border-radius: 40px;
            padding: 20px;
        }
        #image-preview h3 {
            color: #ffcce6;
            margin-bottom: 15px;
            font-size: 1.3em;
            font-weight: 300;
            text-transform: uppercase;
            letter-spacing: 2px;
            text-align: center;
        }
        .thumbnails {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            justify-content: center;
        }
        .thumbnail {
            width: 90px;
            height: 90px;
            object-fit: cover;
            border: 3px solid #ff99cc;
            border-radius: 25px;
            box-shadow: 0 10px 15px rgba(0,0,0,0.5), 0 0 20px #ff99cc;
            transition: 0.3s;
            background: #1a0f2a; /* fallback while loading */
        }
        .thumbnail:hover {
            transform: scale(1.05) rotate(1deg);
            border-color: #ac8cff;
            box-shadow: 0 15px 25px #c080ff;
        }
        /* No glitch element – removed */
    </style>
</head>
<body>
    <div id="main">
        <!-- Left: Chat -->
        <div id="chat-container">
            <div id="chat-header">💬 CHAT</div>
            <div id="chat"></div>
            <div class="input-area">
                <input id="in" placeholder="ask me anything..." />
                <button onclick="send()">SEND</button>
            </div>
        </div>

        <!-- Right: Status Panel -->
        <div id="status-panel">
            <div id="panel-header">📡 STATUS</div>
            <div id="last-container">
                <div>LAST CONTAINER</div>
                <div id="container-value">...</div>
            </div>
            <div id="image-preview">
                <h3>📸 IMAGES</h3>
                <div class="thumbnails" id="thumbnails"></div>
            </div>
        </div>
    </div>

    <script>
        var ws = new WebSocket("ws://" + location.host + "/ws");

        ws.onopen = () => console.log("WS OPEN");
        ws.onclose = () => console.log("WS CLOSED");
        ws.onerror = e => console.error("WS ERROR", e);

        ws.onmessage = function(e) {
            console.log("WS IN:", e.data);
            var d = JSON.parse(e.data);

            if (d.type === "chat") {
                var chatDiv = document.getElementById("chat");
                var msgClass = d.role === "user" ? "user" : "";
                chatDiv.innerHTML += "<div class='msg " + msgClass + "'>" +
                    (d.role === "user" ? "👤 " : "🤖 ") + d.content + "</div>";
                chatDiv.scrollTop = chatDiv.scrollHeight;
            }
            else if (d.type === "status") {
                // Update last container
                document.getElementById("container-value").innerText = d.container.toUpperCase();
                // Update thumbnails if images provided
                if (d.images) {
                    var thumbsDiv = document.getElementById("thumbnails");
                    thumbsDiv.innerHTML = "";
                    if (d.images.current) {
                        var img = document.createElement("img");
                        img.src = d.images.current;
                        img.className = "thumbnail";
                        img.title = "current";
                        thumbsDiv.appendChild(img);
                    }
                    if (d.images.past && d.images.past.length) {
                        d.images.past.forEach((url, i) => {
                            var img = document.createElement("img");
                            img.src = url;
                            img.className = "thumbnail";
                            img.title = "past " + (i+1);
                            thumbsDiv.appendChild(img);
                        });
                    }
                }
            }
        };

        function send() {
            var i = document.getElementById("in");
            if (!i.value.trim()) return;
            ws.send(i.value);
            i.value = "";
        }

        document.getElementById("in").addEventListener("keypress", function(e) {
            if (e.key === "Enter") send();
        });
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

# ---------------- Redis Listeners ----------------
async def ensure_consumer_groups():
    for stream in [INTERFACE_OUTPUT_STREAM, INTERFACE_STATUS_STREAM]:
        try:
            await r.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise e

async def redis_response_listener():
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
                    "type": "chat",
                    "role": "assistant",
                    "content": content,
                    "timestamp": raw.get("timestamp")
                })
                await r.xack(INTERFACE_OUTPUT_STREAM, CONSUMER_GROUP, mid)
        except Exception as e:
            print(f"Response listener error: {e}")
            await asyncio.sleep(1)

async def redis_status_listener():
    while True:
        try:
            messages = await r.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={INTERFACE_STATUS_STREAM: ">"},
                count=10,
                block=2000
            )
            if not messages:
                continue

            for mid, mdata in messages[0][1]:
                raw = json.loads(mdata["data"])
                status_msg = {
                    "type": "status",
                    "container": raw.get("container", "unknown"),
                }
                if raw.get("type") == "image_update":
                    # Rewrite internal image URLs to external ones
                    current = raw.get("current_image_url", "")
                    past = raw.get("past_image_urls", [])
                    if current:
                        # replace internal host:port with external
                        # assumes internal URL starts with http://image-server:8000/
                        current = current.replace("http://image-server:8000", EXTERNAL_IMAGE_URL)
                    past = [url.replace("http://image-server:8000", EXTERNAL_IMAGE_URL) for url in past]
                    status_msg["images"] = {
                        "current": current,
                        "past": past
                    }
                await manager.broadcast(status_msg)
                await r.xack(INTERFACE_STATUS_STREAM, CONSUMER_GROUP, mid)
        except Exception as e:
            print(f"Status listener error: {e}")
            await asyncio.sleep(1)

# ---------------- FastAPI ----------------
@app.on_event("startup")
async def startup():
    await ensure_consumer_groups()
    asyncio.create_task(redis_response_listener())
    asyncio.create_task(redis_status_listener())

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            req_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()

            user_msg = {
                "id": req_id,
                "role": "user",
                "content": data,
                "timestamp": timestamp
            }
            await r.lpush(CHAT_HISTORY_KEY, json.dumps(user_msg))
            await r.ltrim(CHAT_HISTORY_KEY, 0, MAX_HISTORY - 1)

            await manager.broadcast({
                "type": "chat",
                "role": "user",
                "content": data,
                "timestamp": timestamp
            })

            payload = {
                "id": req_id,
                "prompt": data,
                "timestamp": timestamp
            }
            await r.xadd(CAMERA_REQUEST_STREAM, {"data": json.dumps(payload)})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
