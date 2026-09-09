import asyncio
import json
import time
import threading

import websockets

from .base import BaseConnector

_JETSTREAM_URI = (
    "wss://jetstream2.us-east.bsky.network/subscribe"
    "?wantedCollections=app.bsky.feed.post"
)


class BlueskyConnector(BaseConnector):
    """
    Connects to the Bluesky Jetstream — a public WebSocket firehose of all
    Bluesky posts. No authentication required.

    Authority note: follower counts are not available in the firehose without
    an extra API call per user. We use a baseline of 1 so the model still
    receives a valid (if uniform) max_authority_log value. The four other
    features remain fully informative.

    Community note: we use the author's DID (unique per account) as the
    community field. This makes organic_ratio an accurate unique-author ratio
    and geo_score a unique-contributor count — both meaningful signals.
    """

    PLATFORM = "bluesky"

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Event parsing ──────────────────────────────────────────────────────────

    def parse_event(self, raw: dict) -> list[dict]:
        commit = raw.get("commit", {})
        if commit.get("operation") != "create":
            return []

        record = commit.get("record", {})
        text   = record.get("text", "")
        if not text:
            return []

        words = [w for w in text.lower().split() if len(w) > 4 and w.isalpha()]
        if not words:
            return []

        now        = time.time()
        did        = raw.get("did", "unknown")
        is_reply   = bool(record.get("reply"))

        return [
            {
                "keyword"      : word,
                "timestamp"    : now,
                "authority"    : 1,       # firehose doesn't expose follower counts
                "community"    : did,     # unique DID per author → accurate diversity
                "is_reshare"   : is_reply,
                "source_author": None,    # reply source DID not in Jetstream payload
            }
            for word in set(words)
        ]

    # ── Streaming ──────────────────────────────────────────────────────────────

    def start_stream(self, callback) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(callback,),
            daemon=True,
        )
        self._thread.start()

    def stop_stream(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def _run_loop(self, callback) -> None:
        asyncio.run(self._stream(callback))

    async def _stream(self, callback) -> None:
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(
                    _JETSTREAM_URI,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    async for message in ws:
                        if self._stop_event.is_set():
                            break
                        try:
                            data = json.loads(message)
                            for event in self.parse_event(data):
                                callback(event)
                        except Exception:
                            pass
            except Exception:
                if not self._stop_event.is_set():
                    await asyncio.sleep(5)  # back off before reconnect
