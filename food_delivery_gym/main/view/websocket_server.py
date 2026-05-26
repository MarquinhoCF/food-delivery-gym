from __future__ import annotations

import asyncio
import json
import threading
from typing import Set

import websockets
from websockets.server import WebSocketServerProtocol


class WebSocketBroadcaster:
    """Thread-safe broadcaster para o estado da simulação."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: Set[WebSocketServerProtocol] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self):
        """Inicia o servidor WS em thread separada."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_forever, daemon=True
        )
        self._thread.start()

    def _run_forever(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"[WebSocket] Servidor em ws://{self.host}:{self.port}")
            await asyncio.Future()   # roda para sempre

    async def _handler(self, ws: WebSocketServerProtocol):
        self._clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    def broadcast(self, payload: dict):
        """Chamado da thread da simulação (SimPy)."""
        if not self._clients or self._loop is None:
            return
        msg = json.dumps(payload, default=_json_serial)
        asyncio.run_coroutine_threadsafe(
            self._broadcast_async(msg), self._loop
        )

    async def _broadcast_async(self, msg: str):
        if self._clients:
            await asyncio.gather(
                *[ws.send(msg) for ws in list(self._clients)],
                return_exceptions=True,
            )


def _json_serial(obj):
    if hasattr(obj, "name"):
        return obj.name
    return str(obj)