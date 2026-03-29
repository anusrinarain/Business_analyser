import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import google.generativeai as genai

# ───────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SharkLens · Business Intelligence",
    page_icon="🦈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-base: #04080F;
    --bg-card: #080E1A;
    --bg-elevated: #0C1525;
    --border: rgba(56, 189, 248, 0.12);
    --border-bright: rgba(56, 189, 248, 0.35);
    --text-primary: #E8F4FD;
    --text-secondary: #7BA3C4;
    --text-muted: #3D6080;
    --accent-cyan: #38BDF8;
    --accent-violet: #A78BFA;
    --accent-green: #34D399;
    --accent-amber: #FBBF24;
    --accent-rose: #FB7185;
    --accent-indigo: #818CF8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060C18 0%, #04080F 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }

/* ── Main area ── */
.main .block-container {
    padding: 1.5rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── Nav menu ── */
.nav-link {
    border-radius: 10px !important;
    margin: 2px 6px !important;
    font-size: 13px !important;
    font-family: 'DM Sans' !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    padding: 9px 14px !important;
}
.nav-link-selected {
    background: linear-gradient(135deg, #0A1F3A 0%, #0E2848 100%) !important;
    border-left: 3px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
}
.nav-link:hover { background: rgba(56,189,248,0.06) !important; }
.nav-link .icon { color: inherit !important; }

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 22px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    opacity: 0.6;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(56,189,248,0.25);
    transform: translateY(-3px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.5);
}
div[data-testid="stMetricLabel"] > div {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600 !important;
}
div[data-testid="stMetricValue"] > div {
    color: var(--accent-cyan) !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    font-family: 'DM Mono' !important;
}
div[data-testid="stMetricDelta"] > div { font-size: 12px !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0A2040, #0E2E5A) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans' !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0E2848, #153560) !important;
    border-color: rgba(56,189,248,0.5) !important;
    box-shadow: 0 0 24px rgba(56,189,248,0.2) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0E6BAD, #1A87D0) !important;
    color: white !important;
    border-color: transparent !important;
    box-shadow: 0 4px 20px rgba(56,189,248,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1480CC, #2096E0) !important;
    box-shadow: 0 8px 32px rgba(56,189,248,0.4) !important;
}

/* ── Inputs ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans' !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: rgba(56,189,248,0.5) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.1) !important;
}
div[data-testid="stSlider"] > div > div > div {
    background: var(--accent-cyan) !important;
}

/* ── Tabs ── */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 2px !important;
}
div[data-testid="stTabs"] button {
    color: var(--text-muted) !important;
    font-family: 'DM Sans' !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #0A2040, #0E2848) !important;
    color: var(--accent-cyan) !important;
}

/* ── DataFrames ── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.stDataFrame [data-testid="stDataFrameGlideDataEditor"] {
    background: var(--bg-card) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(56,189,248,0.3) !important;
    border-radius: 14px !important;
    background: rgba(56,189,248,0.03) !important;
    padding: 20px !important;
}

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    margin-bottom: 10px !important;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet)) !important;
    border-radius: 6px !important;
}

/* ── Alerts ── */
.stSuccess { background: rgba(52,211,153,0.08) !important; border-left: 3px solid var(--accent-green) !important; border-radius: 8px !important; }
.stWarning { background: rgba(251,191,36,0.08) !important; border-left: 3px solid var(--accent-amber) !important; border-radius: 8px !important; }
.stError   { background: rgba(251,113,133,0.08) !important; border-left: 3px solid var(--accent-rose) !important; border-radius: 8px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: #1A3554; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #254D7A; }

/* ── Custom classes ── */
.page-hero {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.page-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan) 40%, var(--accent-violet) 70%, transparent);
}
.page-hero::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(56,189,248,0.06), transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 14px;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    color: var(--accent-cyan);
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.card {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-elevated));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-cyan  { border-left: 3px solid var(--accent-cyan) !important; }
.card-green { border-left: 3px solid var(--accent-green) !important; }
.card-amber { border-left: 3px solid var(--accent-amber) !important; }
.card-rose  { border-left: 3px solid var(--accent-rose) !important; }
.card-violet{ border-left: 3px solid var(--accent-violet) !important; }

.section-label {
    font-family: 'DM Sans';
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-muted);
    margin-bottom: 12px;
}

.mono { font-family: 'DM Mono' !important; }

.value-positive { color: var(--accent-green) !important; }
.value-negative { color: var(--accent-rose) !important; }
.value-neutral  { color: var(--accent-amber) !important; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ───────────────────────────────────────────────────────────────────
pio.templates.default = "plotly_dark"

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(8,14,26,0.6)',
    font=dict(family='DM Sans', color='#7BA3C4', size=12),
    margin=dict(l=20, r=20, t=44, b=20),
    xaxis=dict(gridcolor='rgba(56,189,248,0.07)', linecolor='rgba(56,189,248,0.15)',
               tickfont=dict(color='#3D6080', size=11)),
    yaxis=dict(gridcolor='rgba(56,189,248,0.07)', linecolor='rgba(56,189,248,0.15)',
               tickfont=dict(color='#3D6080', size=11)),
    title=dict(font=dict(family='Syne', size=15, color='#E8F4FD'), x=0.02),
)

PALETTE = ['#38BDF8','#A78BFA','#34D399','#FBBF24','#FB7185','#818CF8','#F472B6','#6EE7B7']

def apply_layout(fig, height=380):
    fig.update_layout(**CHART_LAYOUT, height=height)
    return fig

# ───────────────────────────────────────────────────────────────────
# GEMINI
# ───────────────────────────────────────────────────────────────────
def gemini_call(messages: list, system_context: str = "") -> str:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
        if not api_key:
            return (
                "**No Gemini API key found.** Add this to `.streamlit/secrets.toml`:\n\n"
                "```toml\nGEMINI_API_KEY = 'your_key_here'\n```\n\n"
                "Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)"
            )
        genai.configure(api_key=api_key)
        model_id = "gemini-1.5-flash"
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods and 'flash' in m.name.lower():
                    model_id = m.name
                    break
        except Exception:
            pass
        model = genai.GenerativeModel(model_id)
        system = (
            "You are SharkLens AI, a sharp, friendly business analyst embedded in a financial intelligence dashboard. "
            "You help entrepreneurs, students, and business owners understand their financials clearly.\n\n"
            f"SESSION CONTEXT:\n{system_context}\n\n"
            "Rules: Be specific and use actual numbers from the context. Keep answers practical and concise. "
            "Use markdown formatting. Avoid jargon — explain like you're on Shark Tank."
        )
        history = []
        for msg in messages[:-1]:
            history.append({"role": "user" if msg["role"] == "user" else "model",
                             "parts": [msg["content"]]})
        chat = model.start_chat(history=history)
        resp = chat.send_message(f"[System]\n{system}\n\n[User Question]\n{messages[-1]['content']}")
        return resp.text
    except Exception as e:
        err = str(e)
        if "quota" in err.lower() or "429" in err:
            return "⚠️ Rate limited. Wait a moment and try again."
        if "key" in err.lower() or "API_KEY" in err.upper():
            return "⚠️ Invalid API key. Check your Gemini key."
        return f"⚠️ Gemini error: {err}"

# ───────────────────────────────────────────────────────────────────
# ML ENGINE
# ───────────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame, target_col: str):
    """Train GradientBoosting + Ridge ensemble, return model, scaler, metrics."""
    df_clean = df.copy()
    # Encode categoricals
    for col in df_clean.select_dtypes(include='object').columns:
        if col != target_col:
            df_clean[col] = df_clean[col].astype('category').cat.codes
    df_clean = df_clean.dropna(subset=[target_col])
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

    feature_cols = [c for c in df_clean.columns if c != target_col]
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    if len(X) < 20:
        model = Ridge(alpha=1.0)
        model.fit(X_sc, y)
        preds = model.predict(X_sc)
        metrics = {
            "r2": r2_score(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
            "mae": mean_absolute_error(y, preds),
            "cv_score": None
        }
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42, subsample=0.8
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        cv = cross_val_score(model, X_sc, y, cv=min(5, len(X)//4), scoring='r2')
        metrics = {
            "r2": r2_score(y_te, preds),
            "rmse": np.sqrt(mean_squared_error(y_te, preds)),
            "mae": mean_absolute_error(y_te, preds),
            "cv_score": cv.mean()
        }

    return model, scaler, feature_cols, metrics

def predict_single(model, scaler, feature_cols, input_dict: dict) -> float:
    row = pd.DataFrame([input_dict])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols].fillna(0)
    X = scaler.transform(row.values)
    return float(model.predict(X)[0])

# ───────────────────────────────────────────────────────────────────
# SESSION STATE
# ───────────────────────────────────────────────────────────────────
_defaults = {
    "df": None,                  # main dataframe
    "manual_entries": [],        # list of dicts from manual form
    "model": None,
    "scaler": None,
    "feature_cols": [],
    "target_col": None,
    "train_metrics": None,
    "chat_history": [],
    "sim_chat_history": [],
    "last_prediction": None,
    "last_sim_results": None,
    "model_trained_on": None,    # 'dataset' or 'manual'
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ───────────────────────────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 12px 16px;text-align:center;border-bottom:1px solid rgba(56,189,248,0.1);margin-bottom:16px;">
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                    background:linear-gradient(135deg,#38BDF8,#A78BFA);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    letter-spacing:-0.5px;">🦈 SharkLens</div>
        <div style="color:#3D6080;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-top:3px;">
            Business Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    selected_page = option_menu(
        menu_title=None,
        options=["Dashboard", "Data Entry", "Predictions", "Simulation Lab",
                 "Visualisations", "AI Insights"],
        icons=["speedometer2", "pencil-square", "graph-up-arrow",
               "sliders2", "bar-chart-line", "stars"],
        default_index=0,
        styles={
            "container":        {"background-color": "transparent", "padding": "0"},
            "icon":             {"font-size": "14px"},
            "nav-link":         {"font-size": "13px", "padding": "9px 14px"},
            "nav-link-selected":{"background": "linear-gradient(135deg,#0A1F3A,#0E2848)",
                                 "color": "#38BDF8"},
        }
    )

    # ── Upload ──
    st.markdown('<div class="section-label" style="margin-top:20px;padding:0 8px;">Dataset</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.session_state.df = df_up
        st.session_state.manual_entries = []
        st.success(f"Loaded {len(df_up):,} rows")

    # ── Status card ──
    if st.session_state.df is not None:
        df_s = st.session_state.df
        st.markdown(f"""
        <div style="background:rgba(56,189,248,0.05);border:1px solid rgba(56,189,248,0.12);
                    border-radius:12px;padding:12px 14px;margin-top:10px;">
            <div style="color:#3D6080;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Dataset Info</div>
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="color:#7BA3C4;font-size:12px;">Rows</span>
                <span style="color:#38BDF8;font-family:'DM Mono';font-size:12px;">{df_s.shape[0]:,}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="color:#7BA3C4;font-size:12px;">Columns</span>
                <span style="color:#38BDF8;font-family:'DM Mono';font-size:12px;">{df_s.shape[1]}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#7BA3C4;font-size:12px;">Missing</span>
                <span style="color:{'#FB7185' if df_s.isnull().sum().sum()>0 else '#34D399'};font-family:'DM Mono';font-size:12px;">
                    {df_s.isnull().sum().sum()}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.manual_entries:
        st.markdown(f"""
        <div style="background:rgba(52,211,153,0.05);border:1px solid rgba(52,211,153,0.15);
                    border-radius:12px;padding:12px 14px;margin-top:10px;">
            <div style="color:#3D6080;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Manual Entries</div>
            <div style="color:#34D399;font-size:13px;font-weight:600;font-family:'DM Mono';">
                {len(st.session_state.manual_entries)} records logged
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.model is not None:
        m_score = st.session_state.train_metrics.get("r2", 0) if st.session_state.train_metrics else 0
        st.markdown(f"""
        <div style="background:rgba(167,139,250,0.05);border:1px solid rgba(167,139,250,0.15);
                    border-radius:12px;padding:12px 14px;margin-top:10px;">
            <div style="color:#3D6080;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Active Model</div>
            <div style="color:#A78BFA;font-size:12px;font-weight:600;">✦ Gradient Boosting</div>
            <div style="color:#7BA3C4;font-size:11px;margin-top:2px;">
                R² Score: <span style="color:#34D399;font-family:'DM Mono';">{m_score:.4f}</span>
            </div>
            <div style="color:#7BA3C4;font-size:11px;">Target: {st.session_state.target_col or '—'}</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════
if selected_page == "Dashboard":
    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge">📊 Live Dashboard</div>
        <h1 class="hero-title">Business Intelligence Overview</h1>
        <p class="hero-subtitle">Track revenue, profit margins, costs, and financial health — all in one place. Think of it as your personal Shark Tank boardroom.</p>
    </div>
    """, unsafe_allow_html=True)

    # Get working dataframe
    df_work = None
    if st.session_state.df is not None:
        df_work = st.session_state.df
    elif st.session_state.manual_entries:
        df_work = pd.DataFrame(st.session_state.manual_entries)

    if df_work is None:
        # Welcome state
        c1, c2, c3 = st.columns(3)
        for col, icon, title, desc, clr in [
            (c1, "📁", "Upload Dataset", "Load a CSV with your financial data", "#38BDF8"),
            (c2, "✏️", "Manual Entry", "Log monthly figures one by one", "#34D399"),
            (c3, "🦈", "Get AI Insights", "Ask questions about your business", "#A78BFA"),
        ]:
            with col:
                col.markdown(f"""
                <div class="card" style="text-align:center;padding:32px 20px;border-top:2px solid {clr}33;">
                    <div style="font-size:40px;margin-bottom:12px;">{icon}</div>
                    <div style="font-family:'Syne';font-size:16px;font-weight:700;
                                color:#E8F4FD;margin-bottom:8px;">{title}</div>
                    <div style="color:#3D6080;font-size:13px;line-height:1.6;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-cyan" style="margin-top:8px;">
            <div style="color:#E8F4FD;font-weight:600;font-size:14px;margin-bottom:10px;">
                What kind of data can SharkLens analyse?
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                <div>
                    <div style="color:#38BDF8;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Financial Fields Supported</div>
                    <div style="color:#7BA3C4;font-size:13px;line-height:2.0;">
                        Revenue / Sales &bull; COGS &bull; Gross Profit<br>
                        Operating Expenses &bull; EBITDA &bull; Net Profit<br>
                        Profit Margin % &bull; Cash Flow &bull; Burn Rate<br>
                        Units Sold &bull; Customer Count &bull; AOV<br>
                        Marketing Spend &bull; CAC &bull; LTV
                    </div>
                </div>
                <div>
                    <div style="color:#34D399;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">What SharkLens Predicts</div>
                    <div style="color:#7BA3C4;font-size:13px;line-height:2.0;">
                        Next Month's Profit / Revenue<br>
                        Profit Margin Trend<br>
                        Break-even Scenarios<br>
                        Risk Score for each scenario<br>
                        AI-generated recommendations
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── KPI Row ──
    num_cols = df_work.select_dtypes(include=np.number).columns.tolist()

    # Smart column detection
    def find_col(df, *keywords):
        for kw in keywords:
            matches = [c for c in df.columns if kw.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    rev_col    = find_col(df_work, 'revenue', 'sales', 'turnover', 'income')
    profit_col = find_col(df_work, 'profit', 'net_profit', 'net profit', 'earnings')
    cost_col   = find_col(df_work, 'cost', 'expense', 'cogs', 'expenditure')
    margin_col = find_col(df_work, 'margin', 'margin%', 'profit_margin')

    kpi_cols = st.columns(4)
    kpi_data = []

    for col_name, label, emoji, clr in [
        (rev_col,    "Total Revenue",   "💰", "#38BDF8"),
        (profit_col, "Net Profit",      "📈", "#34D399"),
        (cost_col,   "Total Costs",     "📉", "#FB7185"),
        (margin_col, "Avg Margin %",    "🎯", "#FBBF24"),
    ]:
        if col_name and col_name in df_work.columns:
            val = df_work[col_name].sum() if label != "Avg Margin %" else df_work[col_name].mean()
            kpi_data.append((label, val, clr))

    if not kpi_data and num_cols:
        for i, col in enumerate(num_cols[:4]):
            val = df_work[col].sum()
            kpi_data.append((col.replace('_',' ').title(), val, PALETTE[i]))

    for i, (label, val, clr) in enumerate(kpi_data[:4]):
        with kpi_cols[i]:
            fmt_val = f"₹{val:,.0f}" if val > 1000 else f"{val:,.2f}"
            st.metric(label, fmt_val)

    st.markdown("---")

    # ── Charts grid ──
    if num_cols:
        c1, c2 = st.columns([3, 2])

        with c1:
            st.markdown('<div class="section-label">Revenue & Profit Over Time</div>', unsafe_allow_html=True)
            time_col = find_col(df_work, 'month', 'date', 'period', 'year', 'quarter')
            plot_cols = [c for c in [rev_col, profit_col, cost_col] if c and c in df_work.columns][:3]

            if plot_cols:
                fig_trend = go.Figure()
                colors_trend = [PALETTE[0], PALETTE[2], PALETTE[4]]
                for idx, col in enumerate(plot_cols):
                    x_vals = df_work[time_col].astype(str) if time_col else list(range(len(df_work)))
                    fig_trend.add_trace(go.Scatter(
                        x=x_vals, y=df_work[col],
                        name=col.replace('_',' ').title(),
                        line=dict(color=colors_trend[idx], width=2.5),
                        fill='tozeroy' if idx == 0 else None,
                        fillcolor=f'rgba({int(colors_trend[0][1:3],16)},{int(colors_trend[0][3:5],16)},{int(colors_trend[0][5:7],16)},0.06)' if idx == 0 else None,
                        mode='lines+markers',
                        marker=dict(size=6, color=colors_trend[idx])
                    ))
                apply_layout(fig_trend, 360)
                fig_trend.update_layout(title="Financial Trend", showlegend=True,
                    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(56,189,248,0.15)',
                                borderwidth=1, font=dict(color='#7BA3C4', size=11)))
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                # fallback: plot all numeric
                fig_fb = px.line(df_work[num_cols[:4]], title="Numeric Trends",
                                 color_discrete_sequence=PALETTE)
                apply_layout(fig_fb, 360)
                st.plotly_chart(fig_fb, use_container_width=True)

        with c2:
            st.markdown('<div class="section-label">Cost Structure</div>', unsafe_allow_html=True)
            cost_candidates = [c for c in df_work.columns
                               if any(k in c.lower() for k in
                                      ['expense','cost','spend','marketing','rent','salary','tax','cogs'])]
            if cost_candidates:
                totals = {c: df_work[c].sum() for c in cost_candidates if pd.api.types.is_numeric_dtype(df_work[c])}
                if totals:
                    fig_pie = go.Figure(go.Pie(
                        labels=list(totals.keys()),
                        values=list(totals.values()),
                        hole=0.55,
                        marker_colors=PALETTE,
                        textfont=dict(size=11, color='white'),
                        textinfo='percent+label',
                    ))
                    fig_pie.update_layout(
                        title="Cost Breakdown",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='DM Sans', color='#7BA3C4'),
                        margin=dict(l=10, r=10, t=44, b=10),
                        height=360,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                # Top 5 numeric cols
                top5 = num_cols[:5]
                sums = {c: df_work[c].mean() for c in top5}
                fig_rad = go.Figure(go.Bar(
                    x=list(sums.values()), y=list(sums.keys()),
                    orientation='h',
                    marker=dict(color=PALETTE[:5], opacity=0.85)
                ))
                apply_layout(fig_rad, 360)
                fig_rad.update_layout(title="Average Values")
                st.plotly_chart(fig_rad, use_container_width=True)

        # ── Second row ──
        c3, c4 = st.columns(2)

        with c3:
            st.markdown('<div class="section-label">Profit Margin Distribution</div>', unsafe_allow_html=True)
            if margin_col and margin_col in df_work.columns:
                series = df_work[margin_col].dropna()
            elif profit_col and rev_col and profit_col in df_work.columns and rev_col in df_work.columns:
                series = (df_work[profit_col] / df_work[rev_col].replace(0, np.nan) * 100).dropna()
                series.name = "Computed Margin %"
            else:
                series = df_work[num_cols[0]] if num_cols else pd.Series()

            if len(series) > 0:
                fig_hist = px.histogram(series, nbins=20,
                                        color_discrete_sequence=['#A78BFA'],
                                        title=f"Distribution — {series.name if hasattr(series,'name') else 'Values'}")
                fig_hist.update_traces(opacity=0.85, marker_line_width=0)
                fig_hist.add_vline(x=float(series.mean()),
                                   line_color='#FBBF24', line_dash='dash', line_width=2,
                                   annotation_text=f"Mean: {series.mean():.1f}",
                                   annotation_font_color='#FBBF24')
                apply_layout(fig_hist, 320)
                st.plotly_chart(fig_hist, use_container_width=True)

        with c4:
            st.markdown('<div class="section-label">Month-over-Month Growth</div>', unsafe_allow_html=True)
            growth_series = None
            if rev_col and rev_col in df_work.columns and len(df_work) > 1:
                growth_series = df_work[rev_col].pct_change() * 100
                growth_col_name = rev_col
            elif num_cols and len(df_work) > 1:
                growth_series = df_work[num_cols[0]].pct_change() * 100
                growth_col_name = num_cols[0]

            if growth_series is not None:
                colors_mom = ['#34D399' if v >= 0 else '#FB7185' for v in growth_series.fillna(0)]
                x_mom = df_work[time_col].astype(str) if time_col else list(range(len(df_work)))
                fig_mom = go.Figure(go.Bar(
                    x=x_mom, y=growth_series.fillna(0),
                    marker_color=colors_mom, opacity=0.9, name="MoM Growth %"
                ))
                fig_mom.add_hline(y=0, line_color='rgba(56,189,248,0.4)', line_width=1)
                apply_layout(fig_mom, 320)
                fig_mom.update_layout(title=f"{growth_col_name.title()} MoM Growth %",
                                      yaxis_title="Growth %")
                st.plotly_chart(fig_mom, use_container_width=True)

    # ── Data Table ──
    st.markdown("---")
    st.markdown('<div class="section-label">Data Table</div>', unsafe_allow_html=True)
    st.dataframe(df_work, use_container_width=True, height=280)


# ═══════════════════════════════════════════════════════════════════
# PAGE: DATA ENTRY
# ═══════════════════════════════════════════════════════════════════
elif selected_page == "Data Entry":
    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge">✏️ Data Entry</div>
        <h1 class="hero-title">Log Your Business Figures</h1>
        <p class="hero-subtitle">No dataset? No problem. Fill in your monthly numbers manually and SharkLens will build a predictive model from your entries.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📝 Manual Entry Form", "📁 Dataset Viewer"])

    with tab1:
        st.markdown("""
        <div class="card card-cyan" style="margin-bottom:20px;">
            <div style="color:#E8F4FD;font-weight:600;font-size:14px;margin-bottom:8px;">Fill in your monthly business metrics</div>
            <div style="color:#7BA3C4;font-size:13px;line-height:1.7;">
                All fields marked with * are required for profit prediction. 
                Fill in as many as possible for better accuracy. Values should be in ₹ (INR) unless otherwise noted.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("manual_entry_form", clear_on_submit=True):
            st.markdown('<div class="section-label">Period & Revenue</div>', unsafe_allow_html=True)
            f1, f2, f3 = st.columns(3)
            with f1:
                month_label = st.text_input("Month / Period *", placeholder="e.g. Jan 2025")
            with f2:
                revenue = st.number_input("Monthly Revenue (₹) *", min_value=0.0, value=0.0, step=1000.0)
            with f3:
                units_sold = st.number_input("Units Sold", min_value=0, value=0, step=1)

            st.markdown('<div class="section-label" style="margin-top:16px;">Costs & Expenses</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                cogs = st.number_input("Cost of Goods Sold (₹) *", min_value=0.0, value=0.0, step=500.0)
            with c2:
                operating_exp = st.number_input("Operating Expenses (₹)", min_value=0.0, value=0.0, step=500.0)
            with c3:
                marketing_spend = st.number_input("Marketing Spend (₹)", min_value=0.0, value=0.0, step=100.0)
            with c4:
                other_costs = st.number_input("Other Costs (₹)", min_value=0.0, value=0.0, step=100.0)

            st.markdown('<div class="section-label" style="margin-top:16px;">Customer & Growth Metrics</div>', unsafe_allow_html=True)
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                customer_count = st.number_input("Customer Count", min_value=0, value=0, step=1)
            with d2:
                new_customers = st.number_input("New Customers", min_value=0, value=0, step=1)
            with d3:
                avg_order_value = st.number_input("Avg Order Value (₹)", min_value=0.0, value=0.0, step=10.0)
            with d4:
                repeat_rate = st.number_input("Repeat Purchase Rate (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)

            st.markdown('<div class="section-label" style="margin-top:16px;">Additional Context</div>', unsafe_allow_html=True)
            e1, e2, e3 = st.columns(3)
            with e1:
                industry = st.selectbox("Industry", ["Retail", "SaaS/Tech", "Food & Beverage",
                                                       "Manufacturing", "Services", "E-commerce",
                                                       "Healthcare", "Education", "Other"])
            with e2:
                growth_stage = st.selectbox("Growth Stage", ["Idea / Pre-revenue", "Early Stage (< 1yr)",
                                                              "Growth (1-3 yrs)", "Scaling (3-5 yrs)",
                                                              "Mature (5+ yrs)"])
            with e3:
                business_model = st.selectbox("Business Model", ["B2C", "B2B", "D2C", "Subscription",
                                                                   "Marketplace", "Franchise", "Other"])

            submitted = st.form_submit_button("Add Entry", type="primary", use_container_width=True)

        if submitted:
            if not month_label or revenue == 0:
                st.error("Month/Period and Revenue are required fields.")
            else:
                total_costs   = cogs + operating_exp + marketing_spend + other_costs
                gross_profit  = revenue - cogs
                net_profit    = revenue - total_costs
                gross_margin  = (gross_profit / revenue * 100) if revenue > 0 else 0
                net_margin    = (net_profit / revenue * 100) if revenue > 0 else 0

                entry = {
                    "period":            month_label,
                    "revenue":           revenue,
                    "units_sold":        units_sold,
                    "cogs":              cogs,
                    "operating_expenses":operating_exp,
                    "marketing_spend":   marketing_spend,
                    "other_costs":       other_costs,
                    "total_costs":       total_costs,
                    "gross_profit":      gross_profit,
                    "net_profit":        net_profit,
                    "gross_margin_pct":  round(gross_margin, 2),
                    "net_margin_pct":    round(net_margin, 2),
                    "customer_count":    customer_count,
                    "new_customers":     new_customers,
                    "avg_order_value":   avg_order_value,
                    "repeat_rate_pct":   repeat_rate,
                    "industry":          industry,
                    "growth_stage":      growth_stage,
                    "business_model":    business_model,
                }
                st.session_state.manual_entries.append(entry)
                st.session_state.df = pd.DataFrame(st.session_state.manual_entries)
                st.success(f"✅ Entry for **{month_label}** added! Net Profit: ₹{net_profit:,.0f} ({net_margin:.1f}% margin)")
                st.rerun()

        # Show current entries
        if st.session_state.manual_entries:
            df_entries = pd.DataFrame(st.session_state.manual_entries)
            st.markdown("---")
            st.markdown('<div class="section-label">Logged Entries</div>', unsafe_allow_html=True)
            st.dataframe(df_entries, use_container_width=True, height=300)

            c_dl, c_clr = st.columns([3, 1])
            with c_dl:
                csv_data = df_entries.to_csv(index=False)
                st.download_button("Download entries as CSV", csv_data,
                                   "sharklens_data.csv", "text/csv", use_container_width=True)
            with c_clr:
                if st.button("Clear all entries", use_container_width=True):
                    st.session_state.manual_entries = []
                    st.session_state.df = None
                    st.rerun()

    with tab2:
        if st.session_state.df is not None:
            df_view = st.session_state.df
            st.markdown(f"""
            <div class="card card-cyan" style="margin-bottom:16px;">
                <div style="color:#38BDF8;font-weight:600;font-size:14px;margin-bottom:4px;">
                    {len(df_view):,} records · {len(df_view.columns)} columns
                </div>
                <div style="color:#7BA3C4;font-size:13px;">
                    Numeric columns: {', '.join(df_view.select_dtypes(include=np.number).columns.tolist()[:8])}
                    {'...' if len(df_view.select_dtypes(include=np.number).columns) > 8 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df_view, use_container_width=True, height=400)

            # Column stats
            st.markdown('<div class="section-label" style="margin-top:16px;">Column Statistics</div>',
                        unsafe_allow_html=True)
            num_df = df_view.select_dtypes(include=np.number)
            if not num_df.empty:
                st.dataframe(num_df.describe().round(2), use_container_width=True)
        else:
            st.info("No data loaded yet. Upload a CSV from the sidebar or use the Manual Entry form above.")


# ═══════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS
# ═══════════════════════════════════════════════════════════════════
elif selected_page == "Predictions":
    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge">📈 ML Predictions</div>
        <h1 class="hero-title">Predictive Forecasting Engine</h1>
        <p class="hero-subtitle">Powered by Gradient Boosting — the same class of models used in enterprise finance forecasting. Train on your data, then predict any future scenario.</p>
    </div>
    """, unsafe_allow_html=True)

    df_pred = st.session_state.df if st.session_state.df is not None else (
        pd.DataFrame(st.session_state.manual_entries) if st.session_state.manual_entries else None
    )

    if df_pred is None:
        st.warning("Upload a dataset or log manual entries first.")
        st.stop()

    num_cols_pred = df_pred.select_dtypes(include=np.number).columns.tolist()
    if not num_cols_pred:
        st.error("Dataset has no numeric columns to predict.")
        st.stop()

    # ── Training section ──
    st.markdown("""
    <div class="card card-violet" style="margin-bottom:20px;">
        <div style="color:#A78BFA;font-weight:700;font-size:14px;margin-bottom:10px;">Step 1: Configure & Train Model</div>
        <div style="color:#7BA3C4;font-size:13px;line-height:1.7;">
            Select what you want to predict (target) and which features to use as input.
            The Gradient Boosting model will automatically learn patterns in your data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        # Smart default target
        default_target_idx = 0
        for i, c in enumerate(num_cols_pred):
            if any(k in c.lower() for k in ['profit', 'revenue', 'net', 'margin']):
                default_target_idx = i
                break
        target_col = st.selectbox("Target Variable (What to predict)",
                                   num_cols_pred, index=default_target_idx)

    with col_b:
        all_features = [c for c in num_cols_pred if c != target_col]
        selected_features = st.multiselect("Feature Columns (predictors)",
                                            all_features, default=all_features[:min(8, len(all_features))])

    if st.button("🚀 Train Gradient Boosting Model", type="primary", use_container_width=True):
        if len(selected_features) == 0:
            st.error("Select at least one feature column.")
        elif len(df_pred) < 4:
            st.error("Need at least 4 rows of data to train. Add more entries.")
        else:
            with st.spinner("Training model..."):
                train_df = df_pred[selected_features + [target_col]].copy()
                try:
                    model, scaler, feat_cols, metrics = train_model(train_df, target_col)
                    st.session_state.model        = model
                    st.session_state.scaler       = scaler
                    st.session_state.feature_cols = feat_cols
                    st.session_state.target_col   = target_col
                    st.session_state.train_metrics = metrics
                    st.success(f"✅ Model trained! R² = {metrics['r2']:.4f} | RMSE = ₹{metrics['rmse']:,.2f}")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    # ── Metrics ──
    if st.session_state.train_metrics:
        m = st.session_state.train_metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² Score",    f"{m['r2']:.4f}",  delta="Excellent" if m['r2'] > 0.85 else "Good" if m['r2'] > 0.65 else "Fair")
        m2.metric("RMSE",        f"₹{m['rmse']:,.2f}")
        m3.metric("MAE",         f"₹{m['mae']:,.2f}")
        m4.metric("CV R² Score", f"{m['cv_score']:.4f}" if m['cv_score'] is not None else "N/A")

    st.markdown("---")

    # ── Prediction form ──
    if st.session_state.model is not None:
        st.markdown("""
        <div class="card card-cyan">
            <div style="color:#38BDF8;font-weight:700;font-size:14px;margin-bottom:8px;">Step 2: Set Input Values & Predict</div>
            <div style="color:#7BA3C4;font-size:13px;">Adjust the sliders below to represent your next month's scenario.</div>
        </div>
        """, unsafe_allow_html=True)

        feat_inputs = {}
        feat_cols   = st.session_state.feature_cols
        n_feat      = len(feat_cols)

        for i in range(0, n_feat, 3):
            batch = feat_cols[i:i+3]
            cols_ui = st.columns(len(batch))
            for j, feat in enumerate(batch):
                with cols_ui[j]:
                    if feat in df_pred.columns and pd.api.types.is_numeric_dtype(df_pred[feat]):
                        mn  = float(df_pred[feat].min())
                        mx  = float(df_pred[feat].max())
                        avg = float(df_pred[feat].mean())
                        # Widen range 20% for future scenarios
                        range_ext = (mx - mn) * 0.2
                        feat_inputs[feat] = st.slider(
                            feat.replace('_', ' ').title(),
                            mn - range_ext, mx + range_ext, avg,
                            key=f"pred_{feat}"
                        )
                    else:
                        feat_inputs[feat] = 0.0

        if st.button("Generate Prediction", type="primary", use_container_width=True):
            try:
                pred_val = predict_single(
                    st.session_state.model,
                    st.session_state.scaler,
                    st.session_state.feature_cols,
                    feat_inputs
                )
                st.session_state.last_prediction = {"value": pred_val, "inputs": feat_inputs,
                                                    "target": target_col}

                target_min = float(df_pred[target_col].min())
                target_max = float(df_pred[target_col].max())
                percentile = int(np.clip((pred_val - target_min) / max(target_max - target_min, 1) * 100, 0, 100))

                color = "#34D399" if pred_val > 0 else "#FB7185"
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#080E1A,#0C1525);
                            border:1px solid rgba(56,189,248,0.2);border-radius:20px;
                            padding:32px;text-align:center;margin:20px 0;
                            box-shadow:0 16px 48px rgba(0,0,0,0.5);">
                    <div style="color:#3D6080;font-size:12px;text-transform:uppercase;
                                letter-spacing:2px;margin-bottom:8px;">Predicted {target_col.replace('_',' ').title()}</div>
                    <div style="font-family:'DM Mono';font-size:52px;font-weight:500;
                                color:{color};letter-spacing:-2px;">
                        {'₹' if pred_val > 100 else ''}{pred_val:,.2f}
                    </div>
                    <div style="color:#3D6080;font-size:13px;margin-top:12px;">
                        Sits at approximately the <span style="color:#FBBF24;font-weight:600;">{percentile}th percentile</span>
                        of your historical data range
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Feature importance
                if hasattr(st.session_state.model, 'feature_importances_'):
                    imps = st.session_state.model.feature_importances_
                    imp_df = pd.DataFrame({"Feature": feat_cols, "Importance": imps})
                    imp_df = imp_df.sort_values("Importance", ascending=True).tail(10)
                    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h',
                                     title="What Drove This Prediction",
                                     color="Importance",
                                     color_continuous_scale=["#A78BFA", "#38BDF8", "#34D399"])
                    apply_layout(fig_imp, 340)
                    fig_imp.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_imp, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("Train the model first using Step 1 above.")


# ═══════════════════════════════════════════════════════════════════
# PAGE: SIMULATION LAB
# ═══════════════════════════════════════════════════════════════════
elif selected_page == "Simulation Lab":
    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge">🧪 Simulation Lab</div>
        <h1 class="hero-title">What-If Scenario Engine</h1>
        <p class="hero-subtitle">Model multiple business scenarios in parallel. Stress-test your financials, find break-even points, and see how risks compound — before they happen.</p>
    </div>
    """, unsafe_allow_html=True)

    df_sim = st.session_state.df if st.session_state.df is not None else (
        pd.DataFrame(st.session_state.manual_entries) if st.session_state.manual_entries else None
    )

    if df_sim is None:
        st.warning("Load data first.")
        st.stop()

    num_cols_sim = df_sim.select_dtypes(include=np.number).columns.tolist()

    # ── Scenario builder ──
    st.markdown("""
    <div class="card card-amber" style="margin-bottom:20px;">
        <div style="color:#FBBF24;font-weight:700;font-size:14px;margin-bottom:8px;">Scenario Configuration</div>
        <div style="color:#7BA3C4;font-size:13px;line-height:1.7;">
            Define your base financials and apply percentage changes to model different futures.
            SharkLens calculates Break-even, Risk Score, and Profit under each scenario.
        </div>
    </div>
    """, unsafe_allow_html=True)

    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        base_revenue  = st.number_input("Base Monthly Revenue (₹)", min_value=0.0,
                                         value=float(df_sim.get('revenue', df_sim[num_cols_sim[0]] if num_cols_sim else pd.Series([100000])).mean() if 'revenue' in df_sim.columns else 100000),
                                         step=5000.0, key="sim_rev")
    with s_col2:
        base_cogs     = st.number_input("Base COGS (₹)", min_value=0.0,
                                         value=float(df_sim.get('cogs', df_sim[num_cols_sim[0]]*0.4 if num_cols_sim else pd.Series([40000])).mean() if 'cogs' in df_sim.columns else base_revenue * 0.4),
                                         step=1000.0, key="sim_cogs")
    with s_col3:
        base_opex     = st.number_input("Base Operating Expenses (₹)", min_value=0.0,
                                         value=float(df_sim['operating_expenses'].mean() if 'operating_expenses' in df_sim.columns else base_revenue * 0.25),
                                         step=1000.0, key="sim_opex")

    st.markdown('<div class="section-label" style="margin-top:16px;">Define 3 Scenarios</div>', unsafe_allow_html=True)

    scenarios = {}
    colors_sc = [("#38BDF8", "Optimistic 🚀"), ("#FBBF24", "Base Case 📊"), ("#FB7185", "Pessimistic ⚠️")]
    sc_cols   = st.columns(3)

    for i, ((clr, sc_name), sc_col) in enumerate(zip(colors_sc, sc_cols)):
        with sc_col:
            st.markdown(f"""
            <div style="background:rgba(0,0,0,0.2);border:1px solid {clr}33;border-radius:12px;
                        padding:14px;margin-bottom:12px;">
                <div style="color:{clr};font-weight:700;font-size:13px;margin-bottom:10px;">{sc_name}</div>
            </div>
            """, unsafe_allow_html=True)
            rev_chg  = st.slider(f"Revenue Change %",  -50, 100, [20, 0, -20][i], 5, key=f"sc{i}_rev")
            cogs_chg = st.slider(f"COGS Change %",      -30,  50, [-10, 0, 20][i], 5, key=f"sc{i}_cogs")
            opex_chg = st.slider(f"Opex Change %",      -30,  50, [-5, 0, 15][i],  5, key=f"sc{i}_opex")

            sc_rev    = base_revenue * (1 + rev_chg / 100)
            sc_cogs   = base_cogs   * (1 + cogs_chg / 100)
            sc_opex   = base_opex   * (1 + opex_chg / 100)
            sc_profit = sc_rev - sc_cogs - sc_opex
            sc_margin = (sc_profit / sc_rev * 100) if sc_rev > 0 else 0
            sc_be     = (sc_cogs + sc_opex)  # break-even revenue needed
            risk_score = max(0, min(100,
                50 + (sc_cogs / max(sc_rev, 1)) * 30 - (sc_margin) * 0.5
                + (max(0, -sc_profit) / max(base_revenue, 1)) * 50
            ))

            scenarios[sc_name] = {
                "revenue": sc_rev, "cogs": sc_cogs, "opex": sc_opex,
                "profit": sc_profit, "margin": sc_margin,
                "breakeven": sc_be, "risk_score": risk_score, "color": clr
            }

            profit_color = clr if sc_profit > 0 else "#FB7185"
            st.markdown(f"""
            <div style="background:{clr}0D;border:1px solid {clr}33;border-radius:10px;padding:12px;margin-top:8px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:#7BA3C4;font-size:12px;">Net Profit</span>
                    <span style="color:{profit_color};font-family:'DM Mono';font-size:12px;font-weight:600;">
                        ₹{sc_profit:,.0f}
                    </span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:#7BA3C4;font-size:12px;">Margin</span>
                    <span style="color:{clr};font-family:'DM Mono';font-size:12px;">{sc_margin:.1f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#7BA3C4;font-size:12px;">Risk Score</span>
                    <span style="color:{'#34D399' if risk_score<30 else '#FBBF24' if risk_score<60 else '#FB7185'};
                                font-family:'DM Mono';font-size:12px;">{risk_score:.0f}/100</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Scenario comparison charts ──
    sc_df = pd.DataFrame([
        {
            "Scenario": name,
            "Revenue": data["revenue"],
            "COGS": data["cogs"],
            "Opex": data["opex"],
            "Net Profit": data["profit"],
            "Margin %": data["margin"],
            "Risk Score": data["risk_score"],
        }
        for name, data in scenarios.items()
    ])

    ch1, ch2 = st.columns(2)
    with ch1:
        fig_sc = go.Figure()
        for metric, clr2 in [("Revenue","#38BDF8"), ("COGS","#FB7185"), ("Net Profit","#34D399")]:
            fig_sc.add_trace(go.Bar(
                name=metric, x=sc_df["Scenario"], y=sc_df[metric],
                marker_color=clr2, opacity=0.85
            ))
        apply_layout(fig_sc, 360)
        fig_sc.update_layout(title="Scenario Comparison — Financials", barmode='group')
        st.plotly_chart(fig_sc, use_container_width=True)

    with ch2:
        fig_risk = go.Figure()
        risk_colors = [v["color"] for v in scenarios.values()]
        fig_risk.add_trace(go.Bar(
            x=sc_df["Scenario"], y=sc_df["Risk Score"],
            marker_color=risk_colors, opacity=0.9,
            text=sc_df["Risk Score"].round(1).astype(str) + " / 100",
            textposition='outside', textfont=dict(color='white', size=12)
        ))
        fig_risk.add_hline(y=50, line_color='#FBBF24', line_dash='dash', line_width=2,
                           annotation_text="Moderate Risk Threshold",
                           annotation_font_color='#FBBF24')
        apply_layout(fig_risk, 360)
        fig_risk.update_layout(title="Risk Score by Scenario", yaxis_range=[0, 110])
        st.plotly_chart(fig_risk, use_container_width=True)

    # ── Break-even analysis ──
    st.markdown("---")
    st.markdown('<div class="section-label">Break-Even Analysis</div>', unsafe_allow_html=True)

    revenue_range = np.linspace(0, base_revenue * 2.5, 300)
    fixed_costs   = base_opex
    variable_cost_ratio = base_cogs / max(base_revenue, 1)
    total_costs_line    = fixed_costs + variable_cost_ratio * revenue_range
    be_revenue          = fixed_costs / max(1 - variable_cost_ratio, 0.01)

    fig_be = go.Figure()
    fig_be.add_trace(go.Scatter(x=revenue_range, y=revenue_range,
        name="Revenue", line=dict(color="#38BDF8", width=2.5)))
    fig_be.add_trace(go.Scatter(x=revenue_range, y=total_costs_line,
        name="Total Costs", line=dict(color="#FB7185", width=2.5),
        fill='tonexty', fillcolor='rgba(251,113,133,0.06)'))
    fig_be.add_vline(x=be_revenue, line_color='#34D399', line_dash='dash', line_width=2,
                     annotation_text=f"Break-even: ₹{be_revenue:,.0f}",
                     annotation_font_color='#34D399', annotation_font_size=13)
    fig_be.add_vline(x=base_revenue, line_color='#FBBF24', line_dash='dot', line_width=2,
                     annotation_text=f"Current: ₹{base_revenue:,.0f}",
                     annotation_font_color='#FBBF24', annotation_font_size=12)
    apply_layout(fig_be, 360)
    fig_be.update_layout(title="Break-Even Chart",
                          xaxis_title="Revenue (₹)", yaxis_title="Amount (₹)",
                          legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#7BA3C4')))
    st.plotly_chart(fig_be, use_container_width=True)

    # Store for AI
    sim_summary = (
        f"3 scenarios: "
        + " | ".join([f"{n}: revenue=₹{d['revenue']:,.0f}, profit=₹{d['profit']:,.0f}, margin={d['margin']:.1f}%, risk={d['risk_score']:.0f}/100"
                       for n, d in scenarios.items()])
        + f" | Break-even at ₹{be_revenue:,.0f}"
    )
    st.session_state.last_sim_results = sim_summary

    # ── Sim AI chat ──
    st.markdown("---")
    st.markdown('<div class="section-label">AI Scenario Analyst</div>', unsafe_allow_html=True)

    q1, q2, q3 = st.columns(3)
    quick_q = None
    if q1.button("Which scenario should I target?", use_container_width=True):
        quick_q = "Based on these 3 scenarios, which should I realistically target and why? What's the key lever I should pull?"
    if q2.button("Explain the risk scores", use_container_width=True):
        quick_q = "Explain what the risk scores mean for each scenario in plain English. What specific risks should I watch out for?"
    if q3.button("How do I reach break-even faster?", use_container_width=True):
        quick_q = "Given my break-even figure, what are the fastest 3 strategies to reach it? Be specific about numbers."

    if quick_q:
        st.session_state.sim_chat_history.append({"role":"user","content":quick_q})

    for msg in st.session_state.sim_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Auto-respond to quick questions
    if st.session_state.sim_chat_history and st.session_state.sim_chat_history[-1]["role"] == "user":
        last_user = st.session_state.sim_chat_history[-1]
        if not (len(st.session_state.sim_chat_history) > 1 and
                st.session_state.sim_chat_history[-2]["role"] == "assistant"):
            with st.chat_message("assistant"):
                with st.spinner("Analysing..."):
                    ctx = f"Simulation results: {sim_summary} | Break-even revenue: ₹{be_revenue:,.0f}"
                    resp = gemini_call(st.session_state.sim_chat_history, ctx)
                st.markdown(resp)
            st.session_state.sim_chat_history.append({"role":"assistant","content":resp})

    if chat_in := st.chat_input("Ask about these scenarios...", key="sim_chat"):
        st.session_state.sim_chat_history.append({"role":"user","content":chat_in})
        with st.chat_message("user"):
            st.markdown(chat_in)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ctx = f"Simulation: {sim_summary}"
                resp = gemini_call(st.session_state.sim_chat_history, ctx)
            st.markdown(resp)
        st.session_state.sim_chat_history.append({"role":"assistant","content":resp})


# ═══════════════════════════════════════════════════════════════════
# PAGE: VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════
elif selected_page == "Visualisations":
    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge">📊 Analytics</div>
        <h1 class="hero-title">Deep Financial Visualisations</h1>
        <p class="hero-subtitle">Explore correlations, distributions, trends, and outliers in your business data through interactive charts.</p>
    </div>
    """, unsafe_allow_html=True)

    df_vis = st.session_state.df if st.session_state.df is not None else (
        pd.DataFrame(st.session_state.manual_entries) if st.session_state.manual_entries else None
    )

    if df_vis is None:
        st.warning("Load data first.")
        st.stop()

    num_cols_v = df_vis.select_dtypes(include=np.number).columns.tolist()
    cat_cols_v = df_vis.select_dtypes(include='object').columns.tolist()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Trends", "Correlations", "Distributions", "Profitability", "Advanced"
    ])

    with tab1:
        if num_cols_v:
            v1, v2 = st.columns([1, 3])
            with v1:
                cols_to_plot = st.multiselect("Select Metrics", num_cols_v,
                                               default=num_cols_v[:min(3, len(num_cols_v))],
                                               max_selections=6)
                chart_type   = st.radio("Chart Type", ["Line", "Area", "Bar"], horizontal=False)
                time_col_v   = st.selectbox("X-Axis", ["Row Index"] + cat_cols_v + num_cols_v)
            with v2:
                if cols_to_plot:
                    x_val = df_vis.index if time_col_v == "Row Index" else df_vis[time_col_v].astype(str)
                    fig_t = go.Figure()
                    for idx2, col2 in enumerate(cols_to_plot):
                        clr2 = PALETTE[idx2 % len(PALETTE)]
                        if chart_type == "Line":
                            fig_t.add_trace(go.Scatter(x=x_val, y=df_vis[col2], name=col2,
                                line=dict(color=clr2, width=2.5), mode='lines+markers',
                                marker=dict(size=6)))
                        elif chart_type == "Area":
                            fig_t.add_trace(go.Scatter(x=x_val, y=df_vis[col2], name=col2,
                                fill='tozeroy', line=dict(color=clr2, width=2),
                                fillcolor=clr2.replace('#','rgba(').replace(')',',0.08)') + '08)'))
                        else:
                            fig_t.add_trace(go.Bar(x=x_val, y=df_vis[col2], name=col2,
                                marker_color=clr2, opacity=0.85))
                    apply_layout(fig_t, 440)
                    fig_t.update_layout(title="Financial Trend Analysis",
                        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#7BA3C4')))
                    st.plotly_chart(fig_t, use_container_width=True)

    with tab2:
        if len(num_cols_v) >= 2:
            corr = df_vis[num_cols_v].corr()
            fig_hm = px.imshow(corr, text_auto=".2f",
                               color_continuous_scale=["#FB7185","#080E1A","#38BDF8"],
                               zmin=-1, zmax=1, title="Correlation Matrix")
            apply_layout(fig_hm, 520)
            fig_hm.update_layout(coloraxis_colorbar=dict(tickfont=dict(color='#7BA3C4')))
            fig_hm.update_traces(textfont=dict(size=10, color='white'))
            st.plotly_chart(fig_hm, use_container_width=True)

            # Scatter
            st.markdown('<div class="section-label" style="margin-top:8px;">Scatter Explorer</div>',
                        unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                x_sc = st.selectbox("X Axis", num_cols_v, index=0, key="sc_x")
            with sc2:
                y_sc = st.selectbox("Y Axis", num_cols_v, index=min(1, len(num_cols_v)-1), key="sc_y")
            with sc3:
                sz_sc = st.selectbox("Bubble Size (optional)", ["None"] + num_cols_v, key="sc_sz")

            fig_sc2 = px.scatter(df_vis, x=x_sc, y=y_sc,
                                  size=sz_sc if sz_sc != "None" else None,
                                  color_discrete_sequence=['#A78BFA'],
                                  trendline='ols' if len(df_vis) > 5 else None,
                                  title=f"{x_sc.title()} vs {y_sc.title()}")
            apply_layout(fig_sc2, 380)
            st.plotly_chart(fig_sc2, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns.")

    with tab3:
        if num_cols_v:
            d1, d2 = st.columns([1, 3])
            with d1:
                dist_col = st.selectbox("Select Column", num_cols_v)
                show_box = st.checkbox("Show box plot", True)
            with d2:
                fig_dist = px.histogram(df_vis, x=dist_col, nbins=25,
                    marginal="box" if show_box else None,
                    color_discrete_sequence=['#A78BFA'],
                    title=f"Distribution — {dist_col.replace('_',' ').title()}")
                fig_dist.update_traces(marker_line_width=0, opacity=0.85)
                fig_dist.add_vline(x=df_vis[dist_col].mean(), line_color='#FBBF24',
                                    line_dash='dash', line_width=2,
                                    annotation_text="Mean", annotation_font_color='#FBBF24')
                fig_dist.add_vline(x=df_vis[dist_col].median(), line_color='#34D399',
                                    line_dash='dot', line_width=2,
                                    annotation_text="Median", annotation_font_color='#34D399')
                apply_layout(fig_dist, 380)
                st.plotly_chart(fig_dist, use_container_width=True)

            # Stats
            stats = df_vis[dist_col].describe()
            iqr   = stats['75%'] - stats['25%']
            outlier_count = len(df_vis[(df_vis[dist_col] < stats['25%']-1.5*iqr) |
                                       (df_vis[dist_col] > stats['75%']+1.5*iqr)])
            st.markdown(f"""
            <div class="card card-cyan" style="margin-top:8px;">
                <div class="section-label">Statistical Summary — {dist_col}</div>
                <div style="display:flex;flex-wrap:wrap;gap:16px;">
                    {''.join([f'<div style="background:rgba(56,189,248,0.06);border-radius:8px;padding:8px 14px;">'
                              f'<div style="color:#3D6080;font-size:10px;text-transform:uppercase;letter-spacing:1px;">{k}</div>'
                              f'<div style="color:#38BDF8;font-family:DM Mono;font-size:14px;font-weight:500;">{v:.3f}</div></div>'
                              for k,v in stats.items()])}
                    <div style="background:rgba(251,113,133,0.06);border-radius:8px;padding:8px 14px;">
                        <div style="color:#3D6080;font-size:10px;text-transform:uppercase;letter-spacing:1px;">Outliers</div>
                        <div style="color:#FB7185;font-family:DM Mono;font-size:14px;font-weight:500;">{outlier_count}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-label">Profitability Deep Dive</div>', unsafe_allow_html=True)
        # Waterfall
        rev_c  = find_col(df_vis, 'revenue','sales') if 'find_col' in dir() else None
        cogs_c = find_col(df_vis, 'cogs','cost_of') if 'find_col' in dir() else None
        opex_c = find_col(df_vis, 'operating','opex') if 'find_col' in dir() else None
        prof_c = find_col(df_vis, 'profit','net_profit') if 'find_col' in dir() else None

        def find_col_local(df, *kws):
            for kw in kws:
                m = [c for c in df.columns if kw.lower() in c.lower()]
                if m: return m[0]
            return None

        rev_c  = find_col_local(df_vis, 'revenue','sales','turnover')
        cogs_c = find_col_local(df_vis, 'cogs','cost_of_goods')
        opex_c = find_col_local(df_vis, 'operating_expenses','opex','operating_exp')
        prof_c = find_col_local(df_vis, 'net_profit','profit')

        if rev_c and cogs_c:
            avg_rev   = df_vis[rev_c].mean()
            avg_cogs  = df_vis[cogs_c].mean()
            avg_opex  = df_vis[opex_c].mean() if opex_c else avg_rev * 0.25
            avg_prof  = avg_rev - avg_cogs - avg_opex

            wf_labels  = ["Revenue", "- COGS", "Gross Profit", "- Opex", "Net Profit"]
            wf_vals    = [avg_rev, -avg_cogs, avg_rev - avg_cogs, -avg_opex, avg_prof]
            wf_measure = ["absolute","relative","total","relative","total"]

            fig_wf = go.Figure(go.Waterfall(
                x=wf_labels, y=wf_vals, measure=wf_measure,
                connector=dict(line=dict(color="rgba(56,189,248,0.3)", width=1)),
                decreasing=dict(marker=dict(color="#FB7185", opacity=0.85)),
                increasing=dict(marker=dict(color="#38BDF8", opacity=0.85)),
                totals=dict(marker=dict(color="#34D399", opacity=0.95)),
                text=[f"₹{abs(v):,.0f}" for v in wf_vals],
                textposition="outside",
                textfont=dict(color="white", size=12)
            ))
            apply_layout(fig_wf, 400)
            fig_wf.update_layout(title="P&L Waterfall (Average Period)", showlegend=False)
            st.plotly_chart(fig_wf, use_container_width=True)
        else:
            # Generic profitability
            if num_cols_v:
                fig_prof = px.box(df_vis[num_cols_v[:6]], title="Value Range by Metric",
                                   color_discrete_sequence=PALETTE)
                apply_layout(fig_prof, 400)
                st.plotly_chart(fig_prof, use_container_width=True)
            st.info("Add 'revenue' and 'cogs' columns for a detailed P&L waterfall.")

    with tab5:
        st.markdown('<div class="section-label">Radar — Multi-Metric Comparison</div>',
                    unsafe_allow_html=True)
        if len(num_cols_v) >= 3:
            radar_cols = st.multiselect("Select dimensions (3–8)", num_cols_v,
                                         default=num_cols_v[:min(6, len(num_cols_v))], max_selections=8)
            if len(radar_cols) >= 3:
                # Normalize to 0-1
                radar_df   = df_vis[radar_cols].copy()
                radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)
                # Show last 5 periods
                n_show = min(5, len(radar_norm))
                fig_rad = go.Figure()
                for i2 in range(n_show):
                    vals = radar_norm.iloc[i2].tolist() + [radar_norm.iloc[i2].tolist()[0]]
                    fig_rad.add_trace(go.Scatterpolar(
                        r=vals, theta=radar_cols + [radar_cols[0]],
                        fill='toself', name=f"Period {i2+1}",
                        line=dict(color=PALETTE[i2 % len(PALETTE)], width=2),
                        fillcolor='rgba(0,0,0,0)'
                    ))
                fig_rad.update_layout(
                    polar=dict(
                        bgcolor='rgba(8,14,26,0.8)',
                        radialaxis=dict(visible=True, range=[0,1],
                                        tickfont=dict(color='#3D6080', size=9),
                                        gridcolor='rgba(56,189,248,0.1)'),
                        angularaxis=dict(tickfont=dict(color='#7BA3C4', size=12),
                                         gridcolor='rgba(56,189,248,0.1)')
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    title=dict(text="Multi-Metric Radar", font=dict(family='Syne', size=15, color='#E8F4FD')),
                    font=dict(family='DM Sans', color='#7BA3C4'),
                    height=480, margin=dict(l=40,r=40,t=60,b=20),
                    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#7BA3C4'))
                )
                st.plotly_chart(fig_rad, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: AI INSIGHTS
# ═══════════════════════════════════════════════════════════════════
elif selected_page == "AI Insights":
    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge">🤖 Powered by Gemini</div>
        <h1 class="hero-title">AI Business Consultant</h1>
        <p class="hero-subtitle">Your personal Shark Tank analyst. Ask anything about your financials — from "am I profitable?" to "how do I scale to ₹1Cr?" — and get specific, data-backed answers.</p>
    </div>
    """, unsafe_allow_html=True)

    # Build full context
    df_ai = st.session_state.df if st.session_state.df is not None else (
        pd.DataFrame(st.session_state.manual_entries) if st.session_state.manual_entries else None
    )

    data_ctx = "No data loaded."
    if df_ai is not None:
        num_ai = df_ai.select_dtypes(include=np.number)
        stats_summary = {}
        for col in num_ai.columns[:8]:
            stats_summary[col] = {
                "mean": round(float(num_ai[col].mean()), 2),
                "max": round(float(num_ai[col].max()), 2),
                "min": round(float(num_ai[col].min()), 2),
            }
        data_ctx = (f"Dataset: {df_ai.shape[0]} rows, {df_ai.shape[1]} columns. "
                    f"Stats: {json.dumps(stats_summary)}")

    model_ctx = ""
    if st.session_state.train_metrics:
        m = st.session_state.train_metrics
        model_ctx = (f"ML Model: GradientBoosting, Target={st.session_state.target_col}, "
                     f"R²={m['r2']:.4f}, RMSE={m['rmse']:.2f}")

    pred_ctx = ""
    if st.session_state.last_prediction:
        p = st.session_state.last_prediction
        pred_ctx = (f"Last Prediction: {p['target']}={p['value']:.2f}, "
                    f"Inputs={json.dumps({k: round(v,2) for k,v in list(p['inputs'].items())[:6]})}")

    sim_ctx = st.session_state.last_sim_results or ""
    full_ctx = f"{data_ctx} | {model_ctx} | {pred_ctx} | {sim_ctx}"

    # ── Report generator ──
    st.markdown('<div class="section-label">Generate Full Business Report</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card card-violet" style="margin-bottom:16px;">
        <div style="color:#A78BFA;font-weight:600;font-size:14px;margin-bottom:6px;">Executive Report</div>
        <div style="color:#7BA3C4;font-size:13px;line-height:1.7;">
            Generates a comprehensive business analysis report — financials, performance, risks, and 
            recommendations — written in plain English for any non-technical audience. 
            Perfect for pitching to investors or presenting to stakeholders.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Generate Full Analysis Report", type="primary", use_container_width=True):
        if df_ai is None and not st.session_state.train_metrics:
            st.warning("Add data or train a model first.")
        else:
            report_prompt = (
                "You are SharkLens AI. Write a comprehensive, professional business intelligence report "
                "based on the session data below. Audience: investors, business owners, or students. "
                "Write in full sentences and paragraphs. Be specific — use the actual numbers. "
                "Use bold for key numbers. Keep total length ~500 words.\n\n"
                f"SESSION DATA:\n{full_ctx}\n\n"
                "Structure the report with these sections:\n"
                "## Executive Summary\n(2-3 sentences, key finding)\n\n"
                "## Financial Performance\n(revenue, profit, margins — are they healthy?)\n\n"
                "## Growth & Trends\n(what direction is the business heading?)\n\n"
                "## Risk Assessment\n(what are the main financial risks right now?)\n\n"
                "## Model Insights\n(if model trained — what does the prediction tell us?)\n\n"
                "## Recommendations\n(3 specific, actionable recommendations with numbers where possible)\n\n"
                "End with a 1-sentence Shark Tank pitch verdict: would a shark invest? Why or why not?"
            )
            with st.spinner("Writing your report..."):
                report = gemini_call([{"role":"user","content":report_prompt}], full_ctx)

            st.markdown(f"""
            <div class="card card-cyan" style="margin-top:12px;">
                <div style="color:#38BDF8;font-weight:700;font-size:13px;margin-bottom:16px;
                            font-family:'Syne';letter-spacing:0.5px;">
                    📋 SharkLens Business Intelligence Report
                </div>
                <div style="color:#CBD5E0;font-size:14px;line-height:1.9;">
                    {report.replace(chr(10),'<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.download_button("Download Report (.txt)", report,
                               "sharklens_report.txt", "text/plain", use_container_width=True)

    st.markdown("---")

    # ── Quick prompts ──
    st.markdown('<div class="section-label">Quick Analysis Prompts</div>', unsafe_allow_html=True)
    qp1, qp2, qp3, qp4 = st.columns(4)
    quick_p = None
    if qp1.button("Am I profitable?",           use_container_width=True):
        quick_p = "Based on my data, am I profitable? Give me specific numbers and tell me if my margins are healthy for my industry."
    if qp2.button("Predict next month",         use_container_width=True):
        quick_p = "Based on trends in my data and any predictions made, what is my most likely profit next month? What assumptions are you making?"
    if qp3.button("Top 3 cost-cutting tips",    use_container_width=True):
        quick_p = "What are my top 3 specific opportunities to cut costs or improve margins, based on the numbers I've entered?"
    if qp4.button("Shark Tank pitch verdict",   use_container_width=True):
        quick_p = "If I walked into Shark Tank with these numbers, what would the sharks say? Be brutally honest. What's my biggest weakness and my biggest strength?"

    if quick_p:
        st.session_state.chat_history.append({"role":"user","content":quick_p})

    # ── Chat ──
    st.markdown("---")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Auto respond to quick prompts
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        last = st.session_state.chat_history[-1]
        needs_reply = (len(st.session_state.chat_history) == 1 or
                       st.session_state.chat_history[-2]["role"] != "assistant")
        if needs_reply:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = gemini_call(st.session_state.chat_history, full_ctx)
                st.markdown(reply)
            st.session_state.chat_history.append({"role":"assistant","content":reply})

    if user_msg := st.chat_input("Ask anything about your business financials..."):
        st.session_state.chat_history.append({"role":"user","content":user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Consulting the sharks..."):
                ai_resp = gemini_call(st.session_state.chat_history, full_ctx)
            st.markdown(ai_resp)
        st.session_state.chat_history.append({"role":"assistant","content":ai_resp})

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()