from collections import deque
import numpy as np


class PlatformNormalizer:
    """
    Converts raw feature values to percentile ranks within a rolling window
    of the platform's own observed distribution.

    This makes the feature space platform-agnostic: velocity=0.85 means
    "top 15% of all velocities seen so far on this platform", regardless
    of whether it's Mastodon, Reddit, or any future source.
    """

    def __init__(self, window_size: int = 2000, min_samples: int = 30):
        self.window_size = window_size
        self.min_samples = min_samples
        self._windows: dict[str, deque] = {}
        self._volume_history: deque = deque(maxlen=window_size)

    # ── Feature normalization ──────────────────────────────────────────────────

    def update(self, features: dict) -> None:
        """Record raw feature values into the rolling windows."""
        for feat, val in features.items():
            if feat not in self._windows:
                self._windows[feat] = deque(maxlen=self.window_size)
            self._windows[feat].append(float(val))

    def normalize(self, features: dict) -> dict:
        """
        Return percentile-ranked features (0-1).
        Falls back to raw value if not enough samples yet.
        """
        normalized = {}
        for feat, val in features.items():
            win = self._windows.get(feat)
            if win and len(win) >= self.min_samples:
                arr = np.array(win)
                normalized[feat] = float(np.mean(arr <= val))
            else:
                normalized[feat] = float(val)
        return normalized

    def is_ready(self) -> bool:
        """True once enough data has been seen to trust normalization."""
        return all(
            len(w) >= self.min_samples
            for w in self._windows.values()
        ) and bool(self._windows)

    # ── Dynamic virality threshold ─────────────────────────────────────────────

    def record_volume(self, count: float) -> None:
        """Record a 12h volume observation for dynamic threshold computation."""
        self._volume_history.append(count)

    def dynamic_threshold(self, percentile: float = 75.0) -> float:
        """
        P{percentile} of observed 12h volumes.
        Falls back to 9 (COVID-19 baseline) if not enough history.
        """
        if len(self._volume_history) < 20:
            return 9.0
        return float(np.percentile(list(self._volume_history), percentile))
