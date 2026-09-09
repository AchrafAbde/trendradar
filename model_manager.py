import joblib
from river import forest, metrics, preprocessing


class ModelManager:
    """
    Online XGBoost replacement using River's Adaptive Random Forest (ARF).

    Lifecycle:
        1. warm_up_model.py pre-trains this on historical feature tables
        2. app.py loads the saved model at startup
        3. On each scored keyword: predict() → display result
        4. feedback_queue.flush() calls learn() ~12h later with the true label
    """

    def __init__(self):
        self.model  = forest.ARFClassifier(n_models=10, seed=42)
        self.scaler = preprocessing.StandardScaler()
        self.metric = metrics.ROCAUC()
        self.n_learned  = 0
        self.n_correct  = 0

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, features: dict) -> float:
        """Return P(viral) in [0, 1]."""
        x = self.scaler.transform_one(features)
        return self.model.predict_proba_one(x).get(1, 0.0)

    # ── Online update ──────────────────────────────────────────────────────────

    def learn(self, features: dict, label: int) -> None:
        """Update model with one labeled example (called by FeedbackQueue)."""
        self.scaler.learn_one(features)
        x = self.scaler.transform_one(features)
        proba = self.model.predict_proba_one(x).get(1, 0.0)
        self.metric.update(label, proba)
        self.model.learn_one(x, label)
        self.n_learned += 1
        if (proba >= 0.5) == bool(label):
            self.n_correct += 1

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def roc_auc(self) -> float:
        return self.metric.get() if self.n_learned > 1 else 0.0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_learned if self.n_learned else 0.0

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "trendradar_river_model.pkl") -> None:
        joblib.dump({
            "model"    : self.model,
            "scaler"   : self.scaler,
            "n_learned": self.n_learned,
            "n_correct": self.n_correct,
        }, path)

    @classmethod
    def load(cls, path: str = "trendradar_river_model.pkl") -> "ModelManager":
        mgr  = cls()
        data = joblib.load(path)
        mgr.model     = data["model"]
        mgr.scaler    = data["scaler"]
        mgr.n_learned = data.get("n_learned", 0)
        mgr.n_correct = data.get("n_correct", 0)
        return mgr
