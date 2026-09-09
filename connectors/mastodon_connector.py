import time
from bs4 import BeautifulSoup
from mastodon import Mastodon, StreamListener
from .base import BaseConnector


class _Listener(StreamListener):
    def __init__(self, connector: "MastodonConnector", callback):
        self.connector = connector
        self.callback  = callback

    def on_update(self, status):
        for event in self.connector.parse_event(status):
            self.callback(event)


class MastodonConnector(BaseConnector):
    PLATFORM = "mastodon"

    def __init__(self, access_token: str, instance_url: str):
        self.client  = Mastodon(access_token=access_token, api_base_url=instance_url)
        self._handle = None

    def parse_event(self, raw: dict) -> list[dict]:
        text    = BeautifulSoup(raw["content"], "html.parser").get_text().lower()
        words   = [w for w in text.split() if len(w) > 4 and w.isalpha()]
        now     = time.time()
        authority     = raw["account"]["followers_count"]
        community     = raw["account"]["acct"].split("@")[-1] or "mastodon.social"
        is_reshare    = raw.get("reblog") is not None
        source_author = raw["reblog"]["account"]["acct"] if is_reshare else None

        return [
            {
                "keyword"      : word,
                "timestamp"    : now,
                "authority"    : authority,
                "community"    : community,
                "is_reshare"   : is_reshare,
                "source_author": source_author,
            }
            for word in set(words)
        ]

    def start_stream(self, callback) -> None:
        self._handle = self.client.stream_public(
            _Listener(self, callback),
            run_async=True,
            reconnect_async=True,
        )

    def stop_stream(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None
