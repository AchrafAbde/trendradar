from abc import ABC, abstractmethod
from typing import Callable


class BaseConnector(ABC):
    """
    Each platform connector must produce normalized event dicts:
    {
        'keyword'      : str,        # extracted keyword/topic
        'timestamp'    : float,      # unix timestamp
        'authority'    : int,        # followers / karma / subscriber count
        'community'    : str,        # instance / subreddit / channel / DID
        'is_reshare'   : bool,       # repost / reblog / crosspost
        'source_author': str | None, # original author if is_reshare, else None
    }
    """
    PLATFORM: str = None

    @abstractmethod
    def start_stream(self, callback: Callable[[dict], None]) -> None:
        """Start streaming. Must be non-blocking. Calls callback(event) per keyword."""
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop the background stream."""
        pass
