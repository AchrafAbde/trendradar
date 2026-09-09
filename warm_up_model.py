"""
Pre-train the River online model on the historical feature tables
(COVID-19, Trump, Biden datasets) before deployment.

Run once from TrendRadar_App/:
    python warm_up_model.py

Produces: trendradar_river_model.pkl  +  trendradar_normalizer.pkl
"""
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from model_manager import ModelManager
from normalizer    import PlatformNormalizer

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = pathlib.Path(__file__).parent.parent
TABLES = ROOT / "Feature Tables"

FILES = {
    "covid"  : TABLES / "feature_table_model.csv",
    "trump"  : TABLES / "feature_table_trump.csv",
    "biden"  : TABLES / "feature_table_biden.csv",
}

# Historical column → new platform-agnostic name
COL_MAP = {"max_followers_log": "max_authority_log"}
FEATURES = ["velocity", "burst_score", "max_authority_log", "organic_ratio", "geo_score"]
LABEL_COL = "Y"

# max_pagerank is always 0 for historical data (no reshare graph available)
PAGERANK_DEFAULT = {"max_pagerank": 0.0}

# ── Load & merge ──────────────────────────────────────────────────────────────
frames = []
for name, path in FILES.items():
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        continue
    df = pd.read_csv(path)
    df = df.rename(columns=COL_MAP)
    missing = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
    if missing:
        print(f"  [skip] {path.name} missing columns: {missing}")
        continue
    frames.append(df[FEATURES + [LABEL_COL]])
    print(f"  [ok]   {path.name}  ->  {len(df)} rows")

if not frames:
    raise SystemExit("No valid feature tables found. Check Feature Tables/ directory.")

data = shuffle(pd.concat(frames, ignore_index=True), random_state=42)
print(f"\nTotal: {len(data)} examples  |  viral rate: {data[LABEL_COL].mean():.1%}\n")

# ── Warm up normalizer first pass (build distribution) ────────────────────────
normalizer = PlatformNormalizer(window_size=len(data) + 500)
for _, row in data.iterrows():
    raw = {**row[FEATURES].to_dict(), **PAGERANK_DEFAULT}
    normalizer.update(raw)

# Seed volume history so dynamic threshold is calibrated from the start
# (use velocity as a proxy for 12h volume — imperfect but sufficient for warm-up)
for _, row in data.iterrows():
    normalizer.record_volume(float(row["velocity"]))

# ── Train River model ─────────────────────────────────────────────────────────
model = ModelManager()

for i, (_, row) in enumerate(data.iterrows()):
    raw   = {**row[FEATURES].to_dict(), **PAGERANK_DEFAULT}
    norm  = normalizer.normalize(raw)
    label = int(row[LABEL_COL])
    model.learn(norm, label)

    if (i + 1) % 200 == 0:
        print(f"  [{i+1}/{len(data)}]  ROC-AUC: {model.roc_auc:.3f}  Acc: {model.accuracy:.1%}")

print(f"\nFinal  ROC-AUC: {model.roc_auc:.3f}  Acc: {model.accuracy:.1%}  "
      f"n_learned: {model.n_learned}")

# ── Save ──────────────────────────────────────────────────────────────────────
model.save("trendradar_river_model.pkl")
joblib.dump(normalizer, "trendradar_normalizer.pkl")
print("\nSaved: trendradar_river_model.pkl  +  trendradar_normalizer.pkl")
