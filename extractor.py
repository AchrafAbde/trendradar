import numpy as np
import networkx as nx


class Extractor:
    """
    Platform-agnostic sliding-window feature extractor.

    Accepts normalized events from any connector:
        add_event(keyword, timestamp, authority, community, is_reshare,
                  source_author=None)

    Returns a feature dict compatible with the River model and normalizer:
        velocity, burst_score, max_authority_log, organic_ratio, geo_score,
        max_pagerank
    """

    FEATURES = [
        "velocity", "burst_score", "max_authority_log",
        "organic_ratio", "geo_score", "max_pagerank",
    ]

    def __init__(self, window: int = 900):
        self.window = window  # T15 = 900 s
        self.data: dict = {}

    def add_event(self, keyword: str, timestamp: float, authority: int,
                  community: str, is_reshare: bool,
                  source_author: str | None = None) -> None:
        if keyword not in self.data:
            self.data[keyword] = {
                "times"        : [],
                "authority"    : [],
                "communities"  : set(),
                "authors"      : [],
                "reshare_graph": {},   # resharer -> set(sources)
            }

        d = self.data[keyword]
        d["times"].append(timestamp)
        d["authority"].append(authority)
        d["communities"].add(community)
        d["authors"].append(community)

        # Build directed reshare graph: resharer → original author
        if is_reshare and source_author:
            d["reshare_graph"].setdefault(community, set()).add(source_author)

        cutoff        = timestamp - self.window
        mask          = [t > cutoff for t in d["times"]]
        d["times"]    = [t for t, k in zip(d["times"],     mask) if k]
        d["authority"]= [a for a, k in zip(d["authority"], mask) if k]
        d["authors"]  = [a for a, k in zip(d["authors"],   mask) if k]

    def _compute_pagerank(self, keyword: str) -> float:
        """PageRank on the reshare graph — max score = most influential source."""
        graph_data = self.data[keyword].get("reshare_graph", {})
        if not graph_data:
            return 0.0
        G = nx.DiGraph()
        for resharer, sources in graph_data.items():
            for source in sources:
                G.add_edge(resharer, source)  # resharer points to original
        if G.number_of_edges() == 0:
            return 0.0
        try:
            pr = nx.pagerank(G, alpha=0.85)
            return float(max(pr.values()))
        except nx.PowerIterationFailedConvergence:
            return 0.0

    def get_features(self, keyword: str, now: float) -> dict | None:
        """Returns feature dict or None if fewer than 3 events in window."""
        if keyword not in self.data:
            return None

        d     = self.data[keyword]
        times = d["times"]
        if len(times) < 3:
            return None

        mid   = now - 450
        start = now - self.window

        v1 = sum(1 for t in times if start <= t < mid)
        v2 = sum(1 for t in times if mid  <= t <= now)

        velocity          = v1 + v2
        burst_score       = (v2 - v1) / np.sqrt(v1 + 1e-5)
        max_authority_log = np.log1p(float(max(d["authority"])) if d["authority"] else 0)
        n_unique          = len(set(d["authors"]))
        organic_ratio     = n_unique / len(d["authors"]) if d["authors"] else 0.0
        geo_score         = float(len(d["communities"]))
        max_pagerank      = self._compute_pagerank(keyword)

        return {
            "velocity"         : float(velocity),
            "burst_score"      : float(burst_score),
            "max_authority_log": float(max_authority_log),
            "organic_ratio"    : float(organic_ratio),
            "geo_score"        : float(geo_score),
            "max_pagerank"     : max_pagerank,
        }

    def get_reshare_graph(self, keyword: str) -> dict:
        """Returns the reshare graph dict for BFS / visualization."""
        if keyword not in self.data:
            return {}
        return dict(self.data[keyword].get("reshare_graph", {}))

    def purge_old(self, now: float, max_age: int = 7200) -> None:
        """Remove keywords with no events in the last max_age seconds."""
        cutoff = now - max_age
        stale  = [kw for kw, d in self.data.items()
                  if not d["times"] or max(d["times"]) < cutoff]
        for kw in stale:
            del self.data[kw]
