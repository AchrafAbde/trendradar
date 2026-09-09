import queue
import time
import pathlib
import joblib
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from datetime import datetime

from extractor              import Extractor
from normalizer             import PlatformNormalizer
from feedback_queue         import FeedbackQueue
from model_manager          import ModelManager
from connectors.mastodon_connector import MastodonConnector
from connectors.bluesky_connector  import BlueskyConnector

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrendRadar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-deep:    #090d12;
    --bg-card:    #0f1520;
    --bg-card2:   #141c2a;
    --border:     #1e2d42;
    --accent:     #00d4ff;
    --accent2:    #7c3aed;
    --green:      #00e676;
    --yellow:     #ffb300;
    --red:        #ff3d3d;
    --text:       #e2eaf4;
    --text-muted: #5a7a9a;
    --font-mono:  'Space Mono', monospace;
    --font-body:  'DM Sans', sans-serif;
}
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg-deep);
    color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }

.radar-header {
    display: flex; align-items: center; gap: 16px;
    padding: 20px 0 12px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.radar-logo {
    font-family: var(--font-mono); font-size: 28px; font-weight: 700;
    color: var(--accent); letter-spacing: -1px;
    text-shadow: 0 0 20px rgba(0,212,255,0.4);
}
.radar-tagline {
    font-size: 13px; color: var(--text-muted);
    letter-spacing: 2px; text-transform: uppercase;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 8px var(--green);
    animation: pulse 2s infinite; display: inline-block; margin-right: 6px;
}
.status-dot.off { background: var(--text-muted); box-shadow: none; animation: none; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

.metric-grid {
    display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 24px;
}
.metric-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 20px; position: relative; overflow: hidden;
}
.metric-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label { font-size:11px; color:var(--text-muted); text-transform:uppercase;
    letter-spacing:1.5px; margin-bottom:6px; font-family:var(--font-mono); }
.metric-value { font-family:var(--font-mono); font-size:26px; font-weight:700;
    color:var(--accent); line-height:1; }
.metric-sub { font-size:11px; color:var(--text-muted); margin-top:4px; }

.trend-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 20px; margin-bottom: 10px;
}
.trend-card.high   { border-left: 3px solid var(--red); }
.trend-card.medium { border-left: 3px solid var(--yellow); }
.trend-card.low    { border-left: 3px solid var(--green); }
.trend-keyword { font-family:var(--font-mono); font-size:18px; font-weight:700; color:var(--text); }
.prob-badge { display:inline-block; padding:3px 10px; border-radius:20px;
    font-family:var(--font-mono); font-size:13px; font-weight:700; }
.prob-high   { background:rgba(255,61,61,0.15);  color:var(--red);    border:1px solid rgba(255,61,61,0.3); }
.prob-medium { background:rgba(255,179,0,0.15);  color:var(--yellow); border:1px solid rgba(255,179,0,0.3); }
.prob-low    { background:rgba(0,230,118,0.15);  color:var(--green);  border:1px solid rgba(0,230,118,0.3); }
.feature-row { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
.feature-pill {
    background:var(--bg-card2); border:1px solid var(--border);
    border-radius:6px; padding:4px 10px; font-size:11px;
    font-family:var(--font-mono); color:var(--text-muted);
}
.feature-pill span { color:var(--accent); font-weight:700; }
.trend-time { font-size:11px; color:var(--text-muted); font-family:var(--font-mono); }

.feat-bar-track { background:var(--border); border-radius:4px; height:6px;
    margin-bottom:10px; overflow:hidden; }
.feat-bar-fill { height:100%; border-radius:4px;
    background:linear-gradient(90deg,var(--accent),var(--accent2)); }

.empty-state { text-align:center; padding:60px 20px; color:var(--text-muted); }
.radar-ring {
    width:80px; height:80px; border:2px solid var(--border); border-radius:50%;
    margin:0 auto 16px; position:relative; animation:radar-spin 3s linear infinite;
}
.radar-ring::after {
    content:''; position:absolute; top:50%; left:50%;
    width:50%; height:2px; background:linear-gradient(90deg,transparent,var(--accent));
    transform-origin:left center; transform:translateY(-50%);
}
@keyframes radar-spin { to { transform:rotate(360deg); } }

section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
.sidebar-section {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:8px; padding:14px; margin-bottom:12px;
}
.sidebar-title {
    font-family:var(--font-mono); font-size:11px; color:var(--text-muted);
    text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;
}
.feat-bar-fill-sidebar { height:4px; border-radius:2px; margin-bottom:8px;
    background:linear-gradient(90deg,var(--accent),var(--accent2)); }
.log-line { font-family:var(--font-mono); font-size:11px; color:var(--text-muted);
    padding:3px 0; border-bottom:1px solid rgba(30,45,66,0.5); }
.log-line .ts { color:var(--accent); }
.hr-glow { border:none; height:1px;
    background:linear-gradient(90deg,transparent,var(--border),transparent); margin:20px 0; }
.stButton > button {
    background: linear-gradient(135deg,var(--accent),var(--accent2)) !important;
    color:#000 !important; font-family:var(--font-mono) !important;
    font-weight:700 !important; letter-spacing:1px !important;
    border:none !important; border-radius:8px !important;
    padding:10px 20px !important; font-size:13px !important;
}
.stButton > button:hover { opacity:0.85 !important; }
</style>
""", unsafe_allow_html=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
MY_ACCESS_TOKEN = st.secrets.get("MASTODON_TOKEN", "")
MY_INSTANCE_URL = "https://mastodon.social"

FEATURE_LABELS = {
    "velocity"         : ("⚡ Velocity",    "Volume T15"),
    "burst_score"      : ("🌊 Burst",        "Accélération"),
    "max_authority_log": ("👑 Autorité",     "Max Authority"),
    "organic_ratio"    : ("🌱 Organique",    "Diversité"),
    "geo_score"        : ("🌍 Geo",          "Dispersion"),
    "max_pagerank"     : ("🕸 PageRank",     "Influenceur max"),
}
FEATURE_IMPORTANCE = {
    "velocity"         : 0.320,
    "burst_score"      : 0.192,
    "organic_ratio"    : 0.188,
    "geo_score"        : 0.119,
    "max_authority_log": 0.118,
    "max_pagerank"     : 0.063,
}

# ── ASSETS ────────────────────────────────────────────────────────────────────
APP_DIR = pathlib.Path(__file__).parent

@st.cache_resource
def load_model_assets():
    model_path      = APP_DIR / "trendradar_river_model.pkl"
    normalizer_path = APP_DIR / "trendradar_normalizer.pkl"

    model      = ModelManager.load(str(model_path)) if model_path.exists() else ModelManager()
    normalizer = joblib.load(str(normalizer_path))  if normalizer_path.exists() else PlatformNormalizer()
    return model, normalizer

def _build_connector(platform: str):
    if platform == "Bluesky":
        return BlueskyConnector()
    return MastodonConnector(MY_ACCESS_TOKEN, MY_INSTANCE_URL)

model, normalizer = load_model_assets()

# ── THREAD-SAFE EVENT QUEUE ───────────────────────────────────────────────────
# Module-level queue persists across Streamlit reruns within a session.
if "event_queue" not in st.session_state:
    st.session_state["event_queue"] = queue.Queue(maxsize=2000)
_event_queue: queue.Queue = st.session_state["event_queue"]

# ── SESSION STATE ─────────────────────────────────────────────────────────────
_defaults = {
    "extractor"    : Extractor(),
    "feedback"     : FeedbackQueue(delay=43200.0),
    "trends_found" : [],
    "stream_log"   : [],
    "is_streaming" : False,
    "total_scanned": 0,
    "labeled_log"  : [],
    "threshold"    : 0.40,
    "platform"     : "Bluesky",
    "connector"    : None,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Build connector on first load or if it hasn't been created yet
if st.session_state.connector is None:
    st.session_state.connector = _build_connector(st.session_state.platform)

# ── STREAM CALLBACK (runs in Mastodon background thread) ─────────────────────
def _on_event(event: dict) -> None:
    try:
        _event_queue.put_nowait(event)
    except queue.Full:
        pass  # drop if UI is too slow to drain

# ── PROCESS QUEUED EVENTS (main thread only) ──────────────────────────────────
def process_events() -> None:
    extractor  = st.session_state.extractor
    feedback   = st.session_state.feedback
    threshold  = st.session_state.threshold
    now        = time.time()
    batch_limit = 300

    for _ in range(batch_limit):
        try:
            event = _event_queue.get_nowait()
        except queue.Empty:
            break

        kw = event["keyword"]
        st.session_state.total_scanned += 1
        feedback.tick_volume(kw)

        extractor.add_event(
            kw,
            event["timestamp"],
            event["authority"],
            event["community"],
            event["is_reshare"],
            event.get("source_author"),
        )

        raw_feats = extractor.get_features(kw, now)
        if raw_feats is None or raw_feats["velocity"] <= 5:
            continue

        normalizer.update(raw_feats)
        norm_feats = normalizer.normalize(raw_feats)
        proba      = model.predict(norm_feats)

        log_entry = {
            "time" : datetime.now().strftime("%H:%M:%S"),
            "word" : kw,
            "proba": proba,
            "feats": raw_feats,
        }
        st.session_state.stream_log.insert(0, log_entry)
        st.session_state.stream_log = st.session_state.stream_log[:50]

        if proba >= threshold:
            existing = {t["keyword"] for t in st.session_state.trends_found}
            if kw not in existing:
                st.session_state.trends_found.insert(0, {
                    "keyword": kw,
                    "proba"  : proba,
                    "time"   : datetime.now().strftime("%H:%M:%S"),
                    "feats"  : raw_feats,
                })
                feedback.add_prediction(kw, raw_feats, norm_feats, proba)
                st.toast(f"🚨 Trend détecté : #{kw}  {proba:.0%}", icon="🔥")

    # ── Flush delayed labels ──────────────────────────────────────────────────
    labeled = feedback.flush(model, normalizer)
    if labeled:
        st.session_state.labeled_log = (labeled + st.session_state.labeled_log)[:20]

    # ── Periodic memory cleanup ───────────────────────────────────────────────
    extractor.purge_old(now)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def prob_class(p: float):
    if p >= 0.75: return "high",   "prob-high"
    if p >= 0.55: return "medium", "prob-medium"
    return "low", "prob-low"

def _bfs_layers(reshare_graph: dict, max_depth: int = 5,
                max_per_layer: int = 8) -> list[list[str]]:
    """
    BFS over the reshare graph starting from the original authors (roots).
    Returns a list of layers: layer[0] = original authors,
    layer[1] = first wave of resharers, etc.
    """
    if not reshare_graph:
        return []
    # Reverse graph: source -> [resharers]
    reverse: dict[str, list[str]] = {}
    for resharer, sources in reshare_graph.items():
        for source in sources:
            reverse.setdefault(source, []).append(resharer)
    all_sources   = set(reverse.keys())
    all_resharers = set(reshare_graph.keys())
    roots = sorted(all_sources - all_resharers,
                   key=lambda n: -len(reverse.get(n, [])))[:max_per_layer]
    if not roots:
        roots = sorted(all_sources,
                       key=lambda n: -len(reverse.get(n, [])))[:max_per_layer]
    visited = set(roots)
    layers  = [list(roots)]
    for _ in range(max_depth - 1):
        next_layer: list[str] = []
        for node in layers[-1]:
            for neighbor in reverse.get(node, []):
                if neighbor not in visited:
                    next_layer.append(neighbor)
                    visited.add(neighbor)
        if not next_layer:
            break
        layers.append(next_layer[:max_per_layer])
    return layers

def _plot_reshare_graph(reshare_graph: dict, keyword: str):
    """Returns a Plotly figure of the reshare network, or None if empty."""
    if not reshare_graph:
        return None
    G = nx.DiGraph()
    for resharer, sources in reshare_graph.items():
        for source in sources:
            G.add_edge(resharer, source)
    if G.number_of_nodes() < 2:
        return None
    try:
        pos = nx.spring_layout(G, seed=42, k=1.2)
        pr  = nx.pagerank(G, alpha=0.85)
    except Exception:
        return None
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=1, color="#1e2d42"), hoverinfo="none")
    node_x    = [pos[n][0] for n in G.nodes()]
    node_y    = [pos[n][1] for n in G.nodes()]
    node_size = [14 + int(pr[n] * 160) for n in G.nodes()]
    node_clr  = [pr[n] for n in G.nodes()]
    node_lbl  = [n[:18] + ("…" if len(n) > 18 else "") for n in G.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_lbl, textposition="top center",
        hovertext=[f"{n}<br>PageRank: {pr[n]:.3f}" for n in G.nodes()],
        hoverinfo="text",
        marker=dict(size=node_size, color=node_clr,
                    colorscale=[[0, "#0f1520"], [0.5, "#7c3aed"], [1, "#00d4ff"]],
                    showscale=False,
                    line=dict(width=1, color="#1e2d42")),
        textfont=dict(size=9, color="#5a7a9a"),
    )
    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#090d12", plot_bgcolor="#090d12",
            showlegend=False, height=340,
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text=f"Graphe de propagation — #{keyword}",
                       font=dict(color="#e2eaf4", size=12, family="Space Mono")),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    )

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:20px;font-weight:700;
                color:#00d4ff;margin-bottom:4px;">📡 TrendRadar</div>
    <div style="font-size:11px;color:#5a7a9a;letter-spacing:2px;
                text-transform:uppercase;margin-bottom:20px;">
        Early Detection System
    </div>""", unsafe_allow_html=True)

    # ── Platform selector ──
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Source</div>', unsafe_allow_html=True)
    selected_platform = st.radio(
        "platform", ["Bluesky", "Mastodon"],
        index=["Bluesky", "Mastodon"].index(st.session_state.platform),
        horizontal=True,
        label_visibility="collapsed",
    )
    if selected_platform != st.session_state.platform:
        if st.session_state.is_streaming:
            st.session_state.connector.stop_stream()
            st.session_state.is_streaming = False
        st.session_state.platform  = selected_platform
        st.session_state.connector = _build_connector(selected_platform)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── River model live stats ──
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Modèle River (Online)</div>', unsafe_allow_html=True)
    for label, val, color in [
        ("ROC-AUC (live)", f"{model.roc_auc:.3f}", "#00d4ff"),
        ("Accuracy",       f"{model.accuracy:.1%}", "#00e676"),
        ("Exemples appris",f"{model.n_learned:,}",  "#ffb300"),
        ("Threshold",      f"{st.session_state.threshold:.0%}", "#7c3aed"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:12px;color:#5a7a9a;">{label}</span>
            <span style="font-family:'Space Mono',monospace;font-size:12px;
                         font-weight:700;color:{color};">{val}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Feature importance ──
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Feature Importance</div>', unsafe_allow_html=True)
    for feat, imp in sorted(FEATURE_IMPORTANCE.items(), key=lambda x: -x[1]):
        label_str = FEATURE_LABELS[feat][0]
        pct_w     = int(imp * 100)
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
            <span style="font-size:11px;color:#e2eaf4;">{label_str}</span>
            <span style="font-family:'Space Mono',monospace;font-size:11px;
                         color:#00d4ff;">{imp:.1%}</span>
        </div>
        <div class="feat-bar-track">
            <div class="feat-bar-fill" style="width:{pct_w}%;"></div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Dataset info ──
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Données d\'entraînement</div>', unsafe_allow_html=True)
    for label, val in [
        ("COVID-19 Twitter",  "425"),
        ("US Elections 2020", "1,018"),
        ("Total hashtags",    "1,443"),
        ("Labels en attente", str(st.session_state.feedback.pending_count)),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span style="font-size:11px;color:#5a7a9a;">{label}</span>
            <span style="font-family:'Space Mono',monospace;font-size:11px;
                         color:#e2eaf4;">{val}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:10px;color:#5a7a9a;text-align:center;
                margin-top:16px;font-family:'Space Mono',monospace;">
        Source : {st.session_state.platform}<br>
        Fenêtre T15 · N≥3 · Seuil adaptatif
    </div>""", unsafe_allow_html=True)

# ── MAIN HEADER ───────────────────────────────────────────────────────────────
is_on      = st.session_state.is_streaming
dot_class  = "status-dot" if is_on else "status-dot off"
status_txt = "LIVE" if is_on else "OFFLINE"

_p75 = normalizer.dynamic_threshold()

st.markdown(f"""
<div class="radar-header">
    <div>
        <div class="radar-logo">TrendRadar</div>
        <div class="radar-tagline">Viral Signal Detection · {st.session_state.platform}</div>
    </div>
    <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
        <span class="{dot_class}"></span>
        <span style="font-family:'Space Mono',monospace;font-size:13px;
                     color:{'#00e676' if is_on else '#5a7a9a'};">{status_txt}</span>
    </div>
</div>
<div style="font-size:12px;color:#5a7a9a;margin-bottom:18px;padding:8px 12px;
            background:#0f1520;border-left:3px solid #1e2d42;border-radius:4px;">
    Le modèle prédit, pour chaque mot-clé actif, la probabilité qu'il devienne viral
    dans les 12&nbsp;prochaines heures
    &nbsp;·&nbsp; Seuil dynamique&nbsp;: <span style="color:#00d4ff;font-family:'Space Mono',monospace;">P75 = {_p75:.0f} posts</span>
    &nbsp;·&nbsp; Threshold&nbsp;: <span style="color:#7c3aed;font-family:'Space Mono',monospace;">P(viral) ≥ {st.session_state.threshold:.0%}</span>
</div>""", unsafe_allow_html=True)

# ── METRIC CARDS ──────────────────────────────────────────────────────────────
n_trends  = len(st.session_state.trends_found)
n_scanned = st.session_state.total_scanned
top_prob  = max((t["proba"] for t in st.session_state.trends_found), default=0.0)

_labeled   = st.session_state.labeled_log
_n_labeled = len(_labeled)
if _n_labeled > 0:
    _n_correct  = sum(1 for r in _labeled if r["label"] == 1)
    _precision  = f"{_n_correct / _n_labeled:.0%}"
    _prec_sub   = f"sur {_n_labeled} label{'s' if _n_labeled > 1 else ''}"
else:
    _precision = "—"
    _prec_sub  = "en attente de T+12h"

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">Prédictions Virales</div>
        <div class="metric-value">{n_trends}</div>
        <div class="metric-sub">P(viral) ≥ seuil</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Posts Analysés</div>
        <div class="metric-value">{n_scanned:,}</div>
        <div class="metric-sub">Flux {st.session_state.platform}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Précision Récente</div>
        <div class="metric-value">{_precision}</div>
        <div class="metric-sub">{_prec_sub}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Probabilité Max</div>
        <div class="metric-value">{top_prob:.0%}</div>
        <div class="metric-sub">Top trend actuel</div>
    </div>
</div>""", unsafe_allow_html=True)

# ── CONTROL + CONTENT ─────────────────────────────────────────────────────────
ctrl_col, feed_col, log_col = st.columns([1.2, 2.5, 1.5])

# ── Control Panel ──────────────────────────────────────────────────────────────
with ctrl_col:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:11px;
                color:#5a7a9a;text-transform:uppercase;letter-spacing:1.5px;
                margin-bottom:12px;">Contrôle</div>""", unsafe_allow_html=True)

    if not st.session_state.is_streaming:
        if st.button("🚀 Lancer le Scan", use_container_width=True):
            try:
                st.session_state.connector.start_stream(_on_event)
                st.session_state.is_streaming = True
                st.rerun()
            except Exception as e:
                st.error(f"Erreur connexion : {e}")
    else:
        if st.button("⏹ Arrêter", use_container_width=True):
            st.session_state.connector.stop_stream()
            st.session_state.is_streaming = False
            st.rerun()

    st.markdown('<div class="hr-glow"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:11px;
                color:#5a7a9a;text-transform:uppercase;letter-spacing:1.5px;
                margin-bottom:8px;">Threshold</div>""", unsafe_allow_html=True)

    threshold = st.slider(
        "P(viral) minimum", 0.20, 0.80,
        st.session_state.threshold, 0.05,
        label_visibility="collapsed",
    )
    st.session_state.threshold = threshold
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace;font-size:20px;font-weight:700;
                color:#00d4ff;text-align:center;margin:4px 0 16px 0;">{threshold:.0%}</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hr-glow"></div>', unsafe_allow_html=True)

    if st.button("🗑 Effacer les résultats", use_container_width=True):
        st.session_state.trends_found   = []
        st.session_state.stream_log     = []
        st.session_state.total_scanned  = 0
        st.session_state.labeled_log    = []
        st.session_state.extractor      = Extractor()
        while not _event_queue.empty():
            try: _event_queue.get_nowait()
            except queue.Empty: break
        st.rerun()

    st.markdown("""
    <div style="background:#0f1520;border:1px solid #1e2d42;border-radius:8px;
                padding:12px;margin-top:16px;font-size:11px;color:#5a7a9a;
                font-family:'Space Mono',monospace;line-height:1.8;">
        <div style="color:#00d4ff;margin-bottom:6px;">HOW IT WORKS</div>
        ① Écoute le flux Mastodon<br>
        ② Normalise les features (P75)<br>
        ③ River prédit P(viral)<br>
        ④ À T+12h → label → learn()
    </div>""", unsafe_allow_html=True)

# ── Trend Feed ────────────────────────────────────────────────────────────────
with feed_col:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:11px;
                color:#5a7a9a;text-transform:uppercase;letter-spacing:1.5px;
                margin-bottom:12px;">Prédictions de Viralité (T+12h)</div>""", unsafe_allow_html=True)

    if not st.session_state.trends_found:
        st.markdown("""
        <div class="empty-state">
            <div class="radar-ring"></div>
            <div style="font-family:'Space Mono',monospace;font-size:13px;
                        color:#5a7a9a;margin-bottom:8px;">SCANNING...</div>
            <div style="font-size:12px;color:#2a4060;">
                Lance le scan pour détecter les tendances virales
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        for trend in st.session_state.trends_found[:10]:
            p               = trend["proba"]
            card_cls, badge_cls = prob_class(p)
            feats           = trend.get("feats", {})
            bar_color       = "var(--red)" if p >= 0.75 else "var(--yellow)" if p >= 0.55 else "var(--green)"
            bar_pct         = int(p * 100)

            feat_pills = "".join(
                f'<div class="feature-pill">{FEATURE_LABELS.get(k, (k,""))[0]} '
                f'<span>{v:.2f}</span></div>'
                for k, v in feats.items()
            )
            st.markdown(f"""
            <div class="trend-card {card_cls}">
                <div style="display:flex;align-items:center;justify-content:space-between;
                            margin-bottom:10px;">
                    <div class="trend-keyword">#{trend['keyword']}</div>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span class="prob-badge {badge_cls}">{p:.1%}</span>
                        <span class="trend-time">{trend['time']}</span>
                    </div>
                </div>
                <div class="feat-bar-track">
                    <div style="height:100%;width:{bar_pct}%;border-radius:4px;
                                background:{bar_color};"></div>
                </div>
                <div class="feature-row">{feat_pills}</div>
            </div>""", unsafe_allow_html=True)

# ── Stream Log ────────────────────────────────────────────────────────────────
with log_col:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:11px;
                color:#5a7a9a;text-transform:uppercase;letter-spacing:1.5px;
                margin-bottom:12px;">Log du Stream</div>""", unsafe_allow_html=True)

    if not st.session_state.stream_log:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:11px;
                    color:#2a4060;padding:20px 0;text-align:center;">
            Aucun signal reçu
        </div>""", unsafe_allow_html=True)
    else:
        log_html = ""
        threshold = st.session_state.threshold
        for entry in st.session_state.stream_log[:20]:
            p        = entry["proba"]
            kw_color = "#ff3d3d" if p >= 0.75 else "#ffb300" if p >= 0.55 else "#e2eaf4"
            flag     = " 🔥" if p >= threshold else ""
            log_html += f"""
            <div class="log-line">
                <span class="ts">{entry['time']}</span>
                <span style="color:#2a4060;"> ▸ </span>
                <span style="color:{kw_color};">{entry['word']}{flag}</span>
                <span style="color:#2a4060;float:right;font-size:10px;">{p:.0%}</span>
            </div>"""
        st.markdown(f"""
        <div style="background:#0f1520;border:1px solid #1e2d42;border-radius:8px;
                    padding:10px 12px;max-height:500px;overflow-y:auto;">
            {log_html}
        </div>""", unsafe_allow_html=True)

    # ── Labeled log (online learning feedback) ────────────────────────────────
    if st.session_state.labeled_log:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:11px;
                    color:#5a7a9a;text-transform:uppercase;letter-spacing:1.5px;
                    margin-top:16px;margin-bottom:8px;">Labels reçus (T+12h)</div>
        """, unsafe_allow_html=True)
        for rec in st.session_state.labeled_log[:5]:
            color = "#00e676" if rec["label"] == 1 else "#5a7a9a"
            badge = "VIRAL" if rec["label"] == 1 else "NON-VIRAL"
            st.markdown(f"""
            <div class="log-line">
                <span style="color:{color};font-weight:700;">{badge}</span>
                <span style="color:#2a4060;"> ▸ </span>
                <span style="color:#e2eaf4;">#{rec['keyword']}</span>
                <span style="color:#2a4060;float:right;font-size:10px;">
                    vol:{rec['volume']:.0f} thr:{rec['threshold']:.0f}
                </span>
            </div>""", unsafe_allow_html=True)

# ── BFS Propagation Explorer ──────────────────────────────────────────────────
if st.session_state.trends_found:
    st.markdown('<div class="hr-glow"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:11px;
                color:#5a7a9a;text-transform:uppercase;letter-spacing:1.5px;
                margin-bottom:4px;">🕸 Propagation Explorer (BFS)</div>
    <div style="font-size:11px;color:#2a4060;margin-bottom:12px;">
        Graphe de repartage du mot-clé sélectionné — nœuds colorés par PageRank
    </div>""", unsafe_allow_html=True)

    kw_options = [t["keyword"] for t in st.session_state.trends_found[:10]]
    selected_bfs = st.selectbox(
        "Choisir un trend", kw_options, label_visibility="collapsed", key="bfs_kw"
    )

    if selected_bfs:
        graph = st.session_state.extractor.get_reshare_graph(selected_bfs)
        if not graph:
            st.markdown("""
            <div style="font-family:'Space Mono',monospace;font-size:11px;
                        color:#2a4060;padding:12px 0;">
                Aucune donnée de repartage capturée pour ce mot-clé.<br>
                (Mastodon expose les reblogs — Bluesky nécessite un cache d'auteurs)
            </div>""", unsafe_allow_html=True)
        else:
            bfs_col, graph_col = st.columns([1, 2])
            with bfs_col:
                layers = _bfs_layers(graph)
                layer_labels = ["Auteurs originaux", "1er relai", "2e relai",
                                "3e relai", "4e relai"]
                for i, layer in enumerate(layers):
                    label = layer_labels[i] if i < len(layer_labels) else f"Relai {i}"
                    color = ["#00d4ff", "#7c3aed", "#ffb300", "#00e676", "#ff3d3d"][i % 5]
                    nodes_html = " ".join(
                        f'<span style="background:#0f1520;border:1px solid {color}33;'
                        f'border-radius:4px;padding:2px 6px;font-size:10px;color:{color};'
                        f'font-family:Space Mono,monospace;">{n[:20]}</span>'
                        for n in layer
                    )
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                        <div style="font-size:10px;color:#5a7a9a;
                                    font-family:Space Mono,monospace;
                                    margin-bottom:4px;">
                            {label} ({len(layer)} nœuds)
                        </div>
                        <div style="display:flex;flex-wrap:wrap;gap:4px;">
                            {nodes_html}
                        </div>
                    </div>""", unsafe_allow_html=True)
                total_nodes = sum(len(l) for l in layers)
                st.markdown(f"""
                <div style="font-size:10px;color:#5a7a9a;font-family:'Space Mono',monospace;
                            margin-top:8px;border-top:1px solid #1e2d42;padding-top:8px;">
                    {total_nodes} nœuds · {len(layers)} couches BFS
                </div>""", unsafe_allow_html=True)
            with graph_col:
                fig = _plot_reshare_graph(graph, selected_bfs)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("""
                    <div style="font-family:'Space Mono',monospace;font-size:11px;
                                color:#2a4060;padding:20px;text-align:center;">
                        Graphe trop petit pour visualiser (min. 2 nœuds)
                    </div>""", unsafe_allow_html=True)

# ── Auto-refresh while streaming ──────────────────────────────────────────────
if st.session_state.is_streaming:
    process_events()
    time.sleep(2)
    st.rerun()
