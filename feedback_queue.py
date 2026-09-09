import time
from collections import defaultdict


class FeedbackQueue:
    """
    Closes the delayed-label loop for online learning.

    Timeline per keyword:
        T+0 min  → prediction made, stored here with feature snapshot
        T+15 min → T15 window closes, we start counting long-term volume
        T+12 h   → compare cumulative count to dynamic threshold → label → learn

    The caller is responsible for:
      - calling `tick_volume(keyword)` every time a keyword is seen in the stream
      - calling `flush(model, normalizer)` periodically (e.g., every rerun cycle)
    """

    def __init__(self, delay: float = 43200.0):
        self.delay   = delay                          # seconds until label is assigned
        self._pending: list[dict] = []                # awaiting label
        self._counts: dict        = defaultdict(int)  # cumulative post counts

    # ── Incoming stream ────────────────────────────────────────────────────────

    def add_prediction(self, keyword: str, raw_features: dict,
                       norm_features: dict, proba: float) -> None:
        self._pending.append({
            "keyword"      : keyword,
            "raw_features" : raw_features,
            "norm_features": norm_features,
            "proba"        : proba,
            "t_predict"    : time.time(),
            "count_at_T15" : self._counts[keyword],
        })

    def tick_volume(self, keyword: str) -> None:
        """Call once per post that mentions this keyword, beyond T15."""
        self._counts[keyword] += 1

    # ── Delayed feedback ───────────────────────────────────────────────────────

    def flush(self, model, normalizer) -> list[dict]:
        """
        Label any prediction whose delay has elapsed, teach the model,
        return a list of labeled records for logging/display.
        """
        now    = time.time()
        ready  = []
        still_pending = []

        for item in self._pending:
            if now - item["t_predict"] < self.delay:
                still_pending.append(item)
                continue

            kw             = item["keyword"]
            total_12h      = self._counts[kw] - item["count_at_T15"]
            threshold      = normalizer.dynamic_threshold()
            label          = int(total_12h >= threshold)

            normalizer.record_volume(float(total_12h))
            model.learn(item["norm_features"], label)

            ready.append({
                "keyword"  : kw,
                "label"    : label,
                "volume"   : total_12h,
                "threshold": threshold,
                "proba"    : item["proba"],
            })

        self._pending = still_pending
        return ready

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def pending_count(self) -> int:
        return len(self._pending)
