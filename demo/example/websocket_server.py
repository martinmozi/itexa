import threading
import asyncio
import websockets

class WsServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self._host = host
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: dict[str, websockets.WebSocketServerProtocol] = {}
        self._server: websockets.WebSocketServer | None = None
        self._started = threading.Event()
        self._thread: threading.Thread | None = None

    def on_connect(self, client_id: str): pass
    def on_disconnect(self, client_id: str): pass
    def on_data(self, client_id: str, data: str): pass

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started.wait()

    def stop(self, timeout: float = 5.0):
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(timeout)
        self._thread.join(timeout)
        self._loop = None
        self._thread = None

    def send_data(self, client_id: str, data: str):
        if (ws := self._clients.get(client_id)) and self._loop:
            fut = asyncio.run_coroutine_threadsafe(self._safe_send(ws, data), self._loop)
            try:
                fut.result(timeout=5.0)
            except Exception:
                pass

    def broadcast(self, data: str):
        for cid in list(self._clients):
            self.send_data(cid, data)

    # --- interné ---
    def _run(self):
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        self._server = await websockets.serve(
            self._handler, self._host, self._port,
            ping_interval=20, ping_timeout=10,
        )
        self._started.set()
        try:
            await self._server.wait_closed()
        finally:
            self._started.clear()

    async def _shutdown(self):
        # zatvor všetky spojenia
        for ws in list(self._clients.values()):
            await ws.close(1001, "server shutting down")
        self._clients.clear()
        # zastav server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handler(self, ws, path):
        cid = str(id(ws))
        self._clients[cid] = ws
        try:
            self.on_connect(cid)
            async for msg in ws:
                self.on_data(cid, msg)
        except websockets.ConnectionClosedError:
            pass
        except websockets.ConnectionClosedOK:
            pass
        finally:
            self._clients.pop(cid, None)
            self.on_disconnect(cid)

    async def _safe_send(self, ws: websockets.WebSocketServerProtocol, data: str):
        try:
            await ws.send(data)
        except (websockets.ConnectionClosed, RuntimeError):
            pass