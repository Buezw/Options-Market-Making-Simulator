"""REST / WebSocket client for the Deribit public API.

No account or authentication needed -- everything here hits public
endpoints/channels only.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import websockets

REST_URL = "https://www.deribit.com/api/v2"
WS_URL = "wss://www.deribit.com/ws/api/v2"

_SUBSCRIBE_CHUNK_SIZE = 200


def get_instruments(currency: str, kind: str) -> list[dict[str, Any]]:
    """Active instruments for a currency/kind ('option' or 'future')."""
    response = httpx.get(
        f"{REST_URL}/public/get_instruments",
        params={"currency": currency, "kind": kind, "expired": "false"},
        timeout=10.0,
    )
    response.raise_for_status()
    instruments = response.json()["result"]
    return [i for i in instruments if i.get("is_active", True)]


class DeribitWebSocketClient:
    """Thin async wrapper over Deribit's public WebSocket JSON-RPC API.

    Reconnection is intentionally not handled here -- that's the caller's
    (recorder's) concern, so this class stays a simple, testable transport.
    """

    def __init__(self) -> None:
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._id_counter = itertools.count(1)

    async def __aenter__(self) -> "DeribitWebSocketClient":
        self._ws = await websockets.connect(WS_URL)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._ws is not None:
            await self._ws.close()

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        assert self._ws is not None
        request_id = next(self._id_counter)
        await self._ws.send(
            json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        )

    async def subscribe(self, channels: list[str]) -> None:
        for i in range(0, len(channels), _SUBSCRIBE_CHUNK_SIZE):
            chunk = channels[i : i + _SUBSCRIBE_CHUNK_SIZE]
            await self._request("public/subscribe", {"channels": chunk})

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yields subscription notification payloads, handling heartbeats internally."""
        assert self._ws is not None
        async for raw in self._ws:
            msg = json.loads(raw)
            if msg.get("method") == "heartbeat":
                if msg.get("params", {}).get("type") == "test_request":
                    await self._request("public/test", {})
                continue
            if msg.get("method") == "subscription":
                yield msg["params"]


async def run_forever_with_reconnect(
    channels: list[str], on_message: Any, reconnect_delay: float = 5.0
) -> None:
    """Connect, subscribe, and dispatch messages to on_message, reconnecting on drop."""
    while True:
        try:
            async with DeribitWebSocketClient() as client:
                await client.subscribe(channels)
                async for params in client.messages():
                    on_message(params)
        except (websockets.ConnectionClosed, OSError):
            await asyncio.sleep(reconnect_delay)
