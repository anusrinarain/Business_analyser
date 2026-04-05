import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import google.generativeai as genai

st.set_page_config(
    page_title="SharkLens · Business Intelligence",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

.main .block-container { padding: 1.5rem 2.5rem !important; max-width: 1400px !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060C18 0%, #04080F 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }

.nav-link {
    border-radius: 10px !important; margin: 2px 6px !important;
    font-size: 13px !important; font-family: 'DM Sans' !important;
    font-weight: 500 !important; color: var(--text-muted) !important;
    padding: 9px 14px !important;
}
.nav-link-selected {
    background: linear-gradient(135deg, #0A1F3A 0%, #0E2848 100%) !important;
    border-left: 3px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
}
.nav-link:hover { background: rgba(56,189,248,0.06) !important; }

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 22px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
    transition: all 0.25s ease;
    position: relative; overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    opacity: 0.6;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(56,189,248,0.25);
    transform: translateY(-3px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.5);
}
div[data-testid="stMetricLabel"] > div {
    color: var(--text-muted) !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600 !important;
}
div[data-testid="stMetricValue"] > div {
    color: var(--accent-cyan) !important; font-size: 26px !important;
    font-weight: 700 !important; font-family: 'DM Mono' !important;
}
div[data-testid="stMetricDelta"] > div { font-size: 12px !important; }

.stButton > button {
    background: linear-gradient(135deg, #0A2040, #0E2E5A) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 10px !important; font-family: 'DM Sans' !important;
    font-weight: 600 !important; font-size: 13px !important;
    padding: 8px 20px !important; transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0E2848, #153560) !important;
    border-color: rgba(56,189,248,0.5) !important;
    box-shadow: 0 0 24px rgba(56,189,248,0.2) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0E6BAD, #1A87D0) !important;
    color: white !important; border-color: transparent !important;
    box-shadow: 0 4px 20px rgba(56,189,248,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1480CC, #2096E0) !important;
    box-shadow: 0 8px 32px rgba(56,189,248,0.4) !important;
}

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

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg-card) !important; border-radius: 12px !important;
    padding: 4px !important; border: 1px solid var(--border) !important; gap: 2px !important;
}
div[data-testid="stTabs"] button {
    color: var(--text-muted) !important; font-family: 'DM Sans' !important;
    font-weight: 500 !important; font-size: 13px !important;
    border-radius: 8px !important; padding: 8px 16px !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #0A2040, #0E2848) !important;
    color: var(--accent-cyan) !important;
}

.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important; overflow: hidden !important;
}

[data-testid="stFileUploader"] {
    border: 1px dashed rgba(56,189,248,0.3) !important;
    border-radius: 14px !important;
    background: rgba(56,189,248,0.03) !important;
    padding: 20px !important;
}

[data-testid="stChatMessage"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important; margin-bottom: 10px !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet)) !important;
    border-radius: 6px !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: #1A3554; border-radius: 4px; }

.page-hero {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border); border-radius: 20px;
    padding: 28px 36px; margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.page-hero::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan) 40%, var(--accent-violet) 70%, transparent);
}
.page-hero::after {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(56,189,248,0.06), transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800;
    color: var(--text-primary); margin: 0 0 6px 0; letter-spacing: -0.5px;
}
.hero-subtitle { color: var(--text-muted); font-size: 14px; margin: 0; font-weight: 400; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.2);
    border-radius: 20px; padding: 4px 12px; font-size: 11px;
    color: var(--accent-cyan); font-weight: 600; letter-spacing: 0.5px;
    text-transform: uppercase; margin-bottom: 12px;
}

.card {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-elevated));
    border: 1px solid var(--border); border-radius: 16px;
    padding: 20px 24px; margin-bottom: 16px;
}
.card-cyan   { border-left: 3px solid var(--accent-cyan)   !important; }
.card-green  { border-left: 3px solid var(--accent-green)  !important; }
.card-amber  { border-left: 3px solid var(--accent-amber)  !important; }
.card-rose   { border-left: 3px solid var(--accent-rose)   !important; }
.card-violet { border-left: 3px solid var(--accent-violet) !important; }

.section-label {
    font-family: 'DM Sans'; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 2px;
    color: var(--text-muted); margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

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

REVENUE_KEYS    = ['total_amount','revenue','sales','turnover','gmv','income','net_sales']
DISCOUNT_KEYS   = ['discount','discount_amount','promo','coupon']
COST_KEYS       = ['cogs','cost_of_goods','cost','expenditure']
OPEX_KEYS       = ['operating','opex','operating_exp','operating_expenses']
PROFIT_KEYS     = ['net_profit','profit','earnings','net_income']
MARGIN_KEYS     = ['margin','profit_margin','margin_pct','net_margin']
QUANTITY_KEYS   = ['quantity','units_sold','items','qty','orders','volume']
RATING_KEYS     = ['rating','customer_rating','score','satisfaction','review']
DELIVERY_KEYS   = ['delivery','delivery_time','shipping_days','lead_time']
SESSION_KEYS    = ['session','session_duration','time_on_site','visit_duration']
ID_KEYS         = ['id','_id','order_id','customer_id','transaction_id']
DATE_KEYS       = ['date','datetime','timestamp','period','month','year']
DEMO_KEYS       = ['age','gender','city','country','region','state']

def classify_col(col_name: str) -> str:
    cn = col_name.lower().replace(' ','_')
    for k in ID_KEYS:
        if k in cn: return 'id'
    for k in DATE_KEYS:
        if cn == k or cn.startswith(k): return 'date'
    for k in REVENUE_KEYS:
        if k in cn: return 'revenue'
    for k in DISCOUNT_KEYS:
        if k in cn: return 'discount'
    for k in COST_KEYS:
        if k in cn: return 'cost'
    for k in OPEX_KEYS:
        if k in cn: return 'opex'
    for k in PROFIT_KEYS:
        if k in cn: return 'profit'
    for k in MARGIN_KEYS:
        if k in cn: return 'margin'
    for k in QUANTITY_KEYS:
        if k in cn: return 'quantity'
    for k in RATING_KEYS:
        if k in cn: return 'rating'
    for k in DELIVERY_KEYS:
        if k in cn: return 'delivery'
    for k in SESSION_KEYS:
        if k in cn: return 'session'
    for k in DEMO_KEYS:
        if k in cn: return 'demographic'
    return 'other_numeric'

def get_kpis(df: pd.DataFrame) -> list:
    kpis = []
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    col_types = {c: classify_col(c) for c in num_cols}

    SUM_TYPES = {'revenue','discount','cost','opex','profit','quantity'}
    AVG_TYPES = {'margin','rating','delivery','session','demographic','other_numeric'}

    added = set()
    priority_order = ['revenue','profit','quantity','discount','cost','opex',
                      'margin','rating','delivery','session']

    for ptype in priority_order:
        if len(kpis) >= 4:
            break
        for col, ctype in col_types.items():
            if ctype == ptype and col not in added:
                if ctype in SUM_TYPES:
                    val  = df[col].sum()
                    agg  = 'Total'
                else:
                    val  = df[col].mean()
                    agg  = 'Avg'
                clean = col.replace('_', ' ').title()
                label = clean if agg.lower() in clean.lower() else f"{agg} {clean}"
                kpis.append((label, val, ctype))
                added.add(col)
                break

    if len(kpis) < 4:
        fallback_avgs = [('Total Orders', len(df), 'count')]
        for item in fallback_avgs:
            if len(kpis) < 4:
                kpis.append(item)

    return kpis[:4]

def find_col(df, *keyword_lists) -> str | None:
    for keywords in keyword_lists:
        for kw in keywords:
            for c in df.columns:
                if kw.lower() in c.lower().replace(' ','_'):
                    return c
    return None

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
        model  = genai.GenerativeModel(model_id)
        system = (
            "You are SharkLens AI, a sharp, friendly business analyst embedded in a financial intelligence dashboard. "
            "You help entrepreneurs, students, and business owners understand their financials clearly.\n\n"
            f"SESSION CONTEXT:\n{system_context}\n\n"
            "Rules: Be specific and use actual numbers from the context. Keep answers practical and concise. "
            "Use markdown formatting. Avoid jargon — explain clearly."
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
            return "Rate limited. Wait a moment and try again."
        if "key" in err.lower() or "API_KEY" in err.upper():
            return "Invalid API key. Check your Gemini key."
        return f"Gemini error: {err}"

def train_model(df: pd.DataFrame, target_col: str):
    df_c = df.copy()
    for col in df_c.select_dtypes(include='object').columns:
        if col != target_col:
            df_c[col] = df_c[col].astype('category').cat.codes
    df_c = df_c.dropna(subset=[target_col])
    df_c = df_c.fillna(df_c.median(numeric_only=True))

    feature_cols = [c for c in df_c.columns if c != target_col]
    X = df_c[feature_cols].values
    y = df_c[target_col].values

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    if len(X) < 20:
        model  = Ridge(alpha=1.0)
        model.fit(X_sc, y)
        preds  = model.predict(X_sc)
        metrics = {
            "r2": float(r2_score(y, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "mae": float(mean_absolute_error(y, preds)),
            "cv_score": None
        }
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sc, y, test_size=0.2, random_state=42
        )
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42, subsample=0.8
        )
        model.fit(X_tr, y_tr)
        preds  = model.predict(X_te)
        n_folds = min(5, max(2, len(X) // 10))
        cv     = cross_val_score(model, X_sc, y, cv=n_folds, scoring='r2')
        metrics = {
            "r2":       float(r2_score(y_te, preds)),
            "rmse":     float(np.sqrt(mean_squared_error(y_te, preds))),
            "mae":      float(mean_absolute_error(y_te, preds)),
            "cv_score": float(cv.mean())
        }

    return model, scaler, feature_cols, metrics

def predict_single(model, scaler, feature_cols: list, input_dict: dict) -> float:
    row = pd.DataFrame([input_dict])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0.0
    row = row[feature_cols].fillna(0.0)
    X   = scaler.transform(row.values.astype(float))
    return float(model.predict(X)[0])

_defaults = {
    "df": None,
    "manual_entries": lambda: [],
    "model": None,
    "scaler": None,
    "feature_cols": lambda: [],
    "target_col": None,
    "train_metrics": None,
    "chat_history": lambda: [],
    "sim_chat_history": lambda: [],
    "last_prediction": None,
    "last_sim_results": None,
    "last_uploaded_file": None,
    "uploader_key": 0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v() if callable(v) else v

with st.sidebar:
    st.markdown("""
    <div style="padding:20px 12px 16px;text-align:center;
                border-bottom:1px solid rgba(56,189,248,0.1);margin-bottom:16px;">
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                    background:linear-gradient(135deg,#38BDF8,#A78BFA);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    letter-spacing:-0.5px;">SharkLens</div>
        <div style="color:#3D6080;font-size:10px;letter-spacing:2px;
                    text-transform:uppercase;margin-top:3px;">Business Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    selected_page = option_menu(
        menu_title=None,
        options=["Dashboard","Data Entry","Predictions","Simulation Lab",
                 "Visualisations","AI Insights"],
        icons=["speedometer2","pencil-square","graph-up-arrow",
               "sliders2","bar-chart-line","stars"],
        default_index=0,
        styles={
            "container":        {"background-color":"transparent","padding":"0"},
            "icon":             {"font-size":"14px"},
            "nav-link":         {"font-size":"13px","padding":"9px 14px"},
            "nav-link-selected":{"background":"linear-gradient(135deg,#0A1F3A,#0E2848)",
                                 "color":"#38BDF8"},
        }
    )

    st.markdown('<div class="section-label" style="margin-top:20px;padding:0 8px;">Dataset</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.session_state.df = df_up
        st.session_state.manual_entries = []
        st.success(f"Loaded {len(df_up):,} rows")

    if st.session_state.df is not None:
        df_s = st.session_state.df
        missing = int(df_s.isnull().sum().sum())
        mc = '#FB7185' if missing > 0 else '#34D399'
        st.markdown(f"""
        <div style="background:rgba(56,189,248,0.05);border:1px solid rgba(56,189,248,0.12);
                    border-radius:12px;padding:12px 14px;margin-top:10px;">
            <div style="color:#3D6080;font-size:10px;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:8px;">Dataset Info</div>
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
                <span style="color:{mc};font-family:'DM Mono';font-size:12px;">{missing}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.manual_entries:
        st.markdown(f"""
        <div style="background:rgba(52,211,153,0.05);border:1px solid rgba(52,211,153,0.15);
                    border-radius:12px;padding:12px 14px;margin-top:10px;">
            <div style="color:#3D6080;font-size:10px;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:6px;">Manual Entries</div>
            <div style="color:#34D399;font-size:13px;font-weight:600;font-family:'DM Mono';">
                {len(st.session_state.manual_entries)} records
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.model is not None and st.session_state.train_metrics:
        m_s = st.session_state.train_metrics
        st.markdown(f"""
        <div style="background:rgba(167,139,250,0.05);border:1px solid rgba(167,139,250,0.15);
                    border-radius:12px;padding:12px 14px;margin-top:10px;">
            <div style="color:#3D6080;font-size:10px;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:6px;">Active Model</div>
            <div style="color:#A78BFA;font-size:12px;font-weight:600;">Gradient Boosting</div>
            <div style="color:#7BA3C4;font-size:11px;margin-top:2px;">
                R² Score:
                <span style="color:#34D399;font-family:'DM Mono';">{m_s['r2']:.4f}</span>
            </div>
            <div style="color:#7BA3C4;font-size:11px;">Target: {st.session_state.target_col or '—'}</div>
        </div>
        """, unsafe_allow_html=True)


if selected_page == "Dashboard":
    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge">Live Dashboard</div>
        <h1 class="hero-title">Business Intelligence Overview</h1>
        <p class="hero-subtitle">Track revenue, order volume, customer behaviour, and financial health. Metrics automatically adapt to your dataset's column structure.</p>
    </div>
    """, unsafe_allow_html=True)

    df_work = st.session_state.df
    if df_work is None and st.session_state.manual_entries:
        df_work = pd.DataFrame(st.session_state.manual_entries)

    if df_work is None:
        c1, c2, c3 = st.columns(3)
        for col_ui, title, desc, clr in [
            (c1, "Upload Dataset",  "Load a CSV with your financial or transactional data", "#38BDF8"),
            (c2, "Manual Entry",    "Log monthly figures one by one",                       "#34D399"),
            (c3, "AI Insights",     "Ask questions about your business financials",         "#A78BFA"),
        ]:
            with col_ui:
                col_ui.markdown(f"""
                <div class="card" style="text-align:center;padding:32px 20px;border-top:2px solid {clr}33;">
                    <div style="font-family:'Syne';font-size:16px;font-weight:700;
                                color:#E8F4FD;margin-bottom:8px;">{title}</div>
                    <div style="color:#3D6080;font-size:13px;line-height:1.6;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.stop()

    num_cols    = df_work.select_dtypes(include=np.number).columns.tolist()
    cat_cols    = df_work.select_dtypes(include='object').columns.tolist()

    kpis = get_kpis(df_work)
    kpi_ui = st.columns(max(len(kpis), 1))
    for i, (label, val, ctype) in enumerate(kpis):
        with kpi_ui[i]:
            if ctype == 'rating':
                fmt = f"{val:.2f} / 5"
            elif ctype in ('delivery','session','demographic'):
                fmt = f"{val:.1f}"
            elif ctype == 'count':
                fmt = f"{int(val):,}"
            elif val > 1_000:
                fmt = f"{val:,.0f}"
            else:
                fmt = f"{val:.2f}"
            st.metric(label, fmt)

    st.markdown("---")

    rev_col  = find_col(df_work, REVENUE_KEYS)
    prof_col = find_col(df_work, PROFIT_KEYS)
    disc_col = find_col(df_work, DISCOUNT_KEYS)
    qty_col  = find_col(df_work, QUANTITY_KEYS)
    date_col = find_col(df_work, DATE_KEYS)
    cat_col  = find_col(df_work, ['product_category','category','product_type','item_category'])
    rat_col  = find_col(df_work, RATING_KEYS)
    pay_col  = find_col(df_work, ['payment','payment_method','pay_type'])

    ch1, ch2 = st.columns([3, 2])

    with ch1:
        st.markdown('<div class="section-label">Revenue Over Time</div>', unsafe_allow_html=True)

        if rev_col and date_col:
            df_time = df_work.copy()
            df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
            df_time = df_time.dropna(subset=[date_col])
            df_time['_period'] = df_time[date_col].dt.to_period('M').astype(str)
            monthly = df_time.groupby('_period')[rev_col].sum().reset_index()
            monthly.columns = ['Period', 'Revenue']

            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=monthly['Period'], y=monthly['Revenue'],
                name='Revenue', line=dict(color='#38BDF8', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(56,189,248,0.06)',
                mode='lines+markers', marker=dict(size=6, color='#38BDF8')
            ))
            apply_layout(fig_t, 360)
            fig_t.update_layout(title="Monthly Revenue", showlegend=False)
            st.plotly_chart(fig_t, use_container_width=True)
        elif num_cols:
            plot_c = [c for c in [rev_col, prof_col] if c is not None]
            if not plot_c:
                plot_c = num_cols[:3]
            fig_fb = go.Figure()
            for idx, c in enumerate(plot_c):
                fig_fb.add_trace(go.Scatter(
                    x=list(range(len(df_work))), y=df_work[c],
                    name=c, line=dict(color=PALETTE[idx], width=2),
                    mode='lines'
                ))
            apply_layout(fig_fb, 360)
            fig_fb.update_layout(title="Numeric Trends",
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#7BA3C4')))
            st.plotly_chart(fig_fb, use_container_width=True)

    with ch2:
        st.markdown('<div class="section-label">Category Breakdown</div>', unsafe_allow_html=True)

        if cat_col and rev_col:
            cat_rev = df_work.groupby(cat_col)[rev_col].sum().reset_index()
            cat_rev.columns = ['Category', 'Revenue']
            cat_rev = cat_rev.sort_values('Revenue', ascending=False)

            fig_pie = go.Figure(go.Pie(
                labels=cat_rev['Category'], values=cat_rev['Revenue'],
                hole=0.55, marker_colors=PALETTE,
                textfont=dict(size=11, color='white'),
                textinfo='percent+label',
            ))
            fig_pie.update_layout(
                title="Revenue by Category",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', color='#7BA3C4'),
                margin=dict(l=10, r=10, t=44, b=10),
                height=360, showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        elif pay_col and rev_col:
            pay_rev = df_work.groupby(pay_col)[rev_col].sum().reset_index()
            pay_rev.columns = ['Method', 'Revenue']
            fig_pay = go.Figure(go.Pie(
                labels=pay_rev['Method'], values=pay_rev['Revenue'],
                hole=0.55, marker_colors=PALETTE,
                textinfo='percent+label',
                textfont=dict(size=11, color='white'),
            ))
            fig_pay.update_layout(
                title="Revenue by Payment Method",
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', color='#7BA3C4'),
                margin=dict(l=10,r=10,t=44,b=10),
                height=360, showlegend=False,
            )
            st.plotly_chart(fig_pay, use_container_width=True)
        elif num_cols:
            avgs = {c: df_work[c].mean() for c in num_cols
                    if classify_col(c) not in ('id','date','demographic') and not np.isnan(df_work[c].mean())}
            if avgs:
                fig_avg = go.Figure(go.Bar(
                    x=list(avgs.values()),
                    y=[c.replace('_',' ').title() for c in avgs.keys()],
                    orientation='h',
                    marker=dict(color=PALETTE[:len(avgs)], opacity=0.85)
                ))
                apply_layout(fig_avg, 360)
                fig_avg.update_layout(title="Average Values")
                st.plotly_chart(fig_avg, use_container_width=True)

    ch3, ch4 = st.columns(2)

    with ch3:
        st.markdown('<div class="section-label">Month-over-Month Growth</div>', unsafe_allow_html=True)

        if rev_col and date_col and 'monthly' in dir():
            mom = monthly.copy()
            mom['Growth_pct'] = mom['Revenue'].pct_change() * 100
            mom = mom.dropna(subset=['Growth_pct'])
            colors_mom = ['#34D399' if v >= 0 else '#FB7185' for v in mom['Growth_pct']]
            fig_mom = go.Figure(go.Bar(
                x=mom['Period'], y=mom['Growth_pct'],
                marker_color=colors_mom, opacity=0.9
            ))
            fig_mom.add_hline(y=0, line_color='rgba(56,189,248,0.4)', line_width=1)
            apply_layout(fig_mom, 320)
            fig_mom.update_layout(title="Revenue MoM Growth (%)", yaxis_title="Growth %")
            st.plotly_chart(fig_mom, use_container_width=True)
        elif rev_col and num_cols:
            mom_s = df_work[rev_col].pct_change() * 100
            colors_mom2 = ['#34D399' if v >= 0 else '#FB7185' for v in mom_s.fillna(0)]
            fig_mom2 = go.Figure(go.Bar(
                x=list(range(len(mom_s))), y=mom_s.fillna(0),
                marker_color=colors_mom2, opacity=0.9
            ))
            fig_mom2.add_hline(y=0, line_color='rgba(56,189,248,0.4)', line_width=1)
            apply_layout(fig_mom2, 320)
            fig_mom2.update_layout(title="Revenue Row-over-Row Change (%)", yaxis_title="Change %")
            st.plotly_chart(fig_mom2, use_container_width=True)

    with ch4:
        st.markdown('<div class="section-label">Customer Rating Distribution</div>', unsafe_allow_html=True)

        if rat_col:
            rating_counts = df_work[rat_col].value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            fig_rat = go.Figure(go.Bar(
                x=rating_counts['Rating'].astype(str),
                y=rating_counts['Count'],
                marker_color=PALETTE[:len(rating_counts)],
                opacity=0.85
            ))
            apply_layout(fig_rat, 320)
            fig_rat.update_layout(title="Customer Ratings Distribution",
                                   xaxis_title="Rating", yaxis_title="Count")
            st.plotly_chart(fig_rat, use_container_width=True)
        elif num_cols:
            last_col = [c for c in num_cols
                        if classify_col(c) not in ('id','date')][-1] if num_cols else None
            if last_col:
                fig_dist = px.histogram(df_work, x=last_col, nbins=25,
                                         color_discrete_sequence=['#A78BFA'],
                                         title=f"Distribution — {last_col.replace('_',' ').title()}")
                fig_dist.update_traces(marker_line_width=0, opacity=0.85)
                apply_layout(fig_dist, 320)
                st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Data Table</div>', unsafe_allow_html=True)
    st.dataframe(df_work, use_container_width=True, height=280)


elif selected_page == "Data Entry":
    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge">Data Entry</div>
        <h1 class="hero-title">Log Your Business Figures</h1>
        <p class="hero-subtitle">No dataset? Fill in your monthly numbers manually. SharkLens calculates derived metrics and builds a predictive model from your entries.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Manual Entry Form", "Dataset Viewer"])

    with tab1:
        st.markdown("""
        <div class="card card-cyan" style="margin-bottom:20px;">
            <div style="color:#E8F4FD;font-weight:600;font-size:14px;margin-bottom:8px;">Monthly business metrics form</div>
            <div style="color:#7BA3C4;font-size:13px;line-height:1.7;">
                Fields marked * are required. All monetary values in INR.
                Gross Profit, Net Profit, and Margins are calculated automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("manual_entry_form", clear_on_submit=True):
            st.markdown('<div class="section-label">Period and Revenue</div>', unsafe_allow_html=True)
            f1, f2, f3 = st.columns(3)
            with f1:
                month_label = st.text_input("Month / Period *", placeholder="e.g. Jan 2025")
            with f2:
                revenue = st.number_input("Monthly Revenue (INR) *", min_value=0.0, value=0.0, step=1000.0)
            with f3:
                units_sold = st.number_input("Units Sold", min_value=0, value=0, step=1)

            st.markdown('<div class="section-label" style="margin-top:16px;">Costs</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                cogs = st.number_input("Cost of Goods Sold (INR) *", min_value=0.0, value=0.0, step=500.0)
            with c2:
                operating_exp = st.number_input("Operating Expenses (INR)", min_value=0.0, value=0.0, step=500.0)
            with c3:
                marketing_spend = st.number_input("Marketing Spend (INR)", min_value=0.0, value=0.0, step=100.0)
            with c4:
                other_costs = st.number_input("Other Costs (INR)", min_value=0.0, value=0.0, step=100.0)

            st.markdown('<div class="section-label" style="margin-top:16px;">Customer Metrics</div>', unsafe_allow_html=True)
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                customer_count = st.number_input("Customer Count", min_value=0, value=0, step=1)
            with d2:
                new_customers = st.number_input("New Customers", min_value=0, value=0, step=1)
            with d3:
                avg_order_value = st.number_input("Avg Order Value (INR)", min_value=0.0, value=0.0, step=10.0)
            with d4:
                repeat_rate = st.number_input("Repeat Rate (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)

            st.markdown('<div class="section-label" style="margin-top:16px;">Business Context</div>', unsafe_allow_html=True)
            e1, e2, e3 = st.columns(3)
            with e1:
                industry = st.selectbox("Industry",
                    ["Retail","SaaS/Tech","Food & Beverage","Manufacturing",
                     "Services","E-commerce","Healthcare","Education","Other"])
            with e2:
                growth_stage = st.selectbox("Growth Stage",
                    ["Idea / Pre-revenue","Early Stage (< 1yr)",
                     "Growth (1-3 yrs)","Scaling (3-5 yrs)","Mature (5+ yrs)"])
            with e3:
                business_model = st.selectbox("Business Model",
                    ["B2C","B2B","D2C","Subscription","Marketplace","Franchise","Other"])

            submitted = st.form_submit_button("Add Entry", type="primary", use_container_width=True)

        if submitted:
            if not month_label or revenue == 0:
                st.error("Month/Period and Revenue are required.")
            else:
                total_costs  = cogs + operating_exp + marketing_spend + other_costs
                gross_profit = revenue - cogs
                net_profit   = revenue - total_costs
                gross_margin = (gross_profit / revenue * 100) if revenue > 0 else 0.0
                net_margin   = (net_profit   / revenue * 100) if revenue > 0 else 0.0

                entry = {
                    "period":             month_label,
                    "revenue":            revenue,
                    "units_sold":         units_sold,
                    "cogs":               cogs,
                    "operating_expenses": operating_exp,
                    "marketing_spend":    marketing_spend,
                    "other_costs":        other_costs,
                    "total_costs":        total_costs,
                    "gross_profit":       gross_profit,
                    "net_profit":         net_profit,
                    "gross_margin_pct":   round(gross_margin, 2),
                    "net_margin_pct":     round(net_margin,   2),
                    "customer_count":     customer_count,
                    "new_customers":      new_customers,
                    "avg_order_value":    avg_order_value,
                    "repeat_rate_pct":    repeat_rate,
                    "industry":           industry,
                    "growth_stage":       growth_stage,
                    "business_model":     business_model,
                }
                st.session_state.manual_entries.append(entry)
                st.session_state.df = pd.DataFrame(st.session_state.manual_entries)
                st.success(f"Entry for {month_label} added. Net Profit: INR {net_profit:,.0f} ({net_margin:.1f}% margin)")
                st.rerun()

        if st.session_state.manual_entries:
            df_e = pd.DataFrame(st.session_state.manual_entries)
            st.markdown("---")
            st.markdown('<div class="section-label">Logged Entries</div>', unsafe_allow_html=True)
            st.dataframe(df_e, use_container_width=True, height=300)

            col_dl, col_clr = st.columns([3,1])
            with col_dl:
                st.download_button("Download entries as CSV",
                                   df_e.to_csv(index=False),
                                   "sharklens_data.csv","text/csv",
                                   use_container_width=True)
            with col_clr:
                if st.button("Clear all entries", use_container_width=True):
                    st.session_state.manual_entries = []
                    st.session_state.df = None
                    st.rerun()

    with tab2:
        if st.session_state.df is not None:
            df_v = st.session_state.df
            num_v = df_v.select_dtypes(include=np.number).columns.tolist()
            st.markdown(f"""
            <div class="card card-cyan" style="margin-bottom:16px;">
                <div style="color:#38BDF8;font-weight:600;font-size:14px;margin-bottom:4px;">
                    {len(df_v):,} records &middot; {len(df_v.columns)} columns
                </div>
                <div style="color:#7BA3C4;font-size:13px;">
                    Numeric: {', '.join(num_v[:10])}{'...' if len(num_v) > 10 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df_v, use_container_width=True, height=400)

            st.markdown('<div class="section-label" style="margin-top:16px;">Column Statistics</div>',
                        unsafe_allow_html=True)
            if num_v:
                st.dataframe(df_v[num_v].describe().round(3), use_container_width=True)
        else:
            st.info("No data loaded. Upload a CSV from the sidebar or use the Manual Entry form.")


elif selected_page == "Predictions":
    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge">ML Predictions</div>
        <h1 class="hero-title">Predictive Forecasting Engine</h1>
        <p class="hero-subtitle">Powered by Gradient Boosting — an ensemble of decision trees that iteratively corrects its own errors. Train on your data, then predict any future scenario.</p>
    </div>
    """, unsafe_allow_html=True)

    df_pred = st.session_state.df
    if df_pred is None and st.session_state.manual_entries:
        df_pred = pd.DataFrame(st.session_state.manual_entries)

    if df_pred is None:
        st.warning("Upload a dataset or log manual entries first.")
        st.stop()

    num_cols_p = df_pred.select_dtypes(include=np.number).columns.tolist()
    if not num_cols_p:
        st.error("Dataset has no numeric columns.")
        st.stop()

    non_id_num = [c for c in num_cols_p if classify_col(c) not in ('id',)]

    st.markdown("""
    <div class="card card-violet" style="margin-bottom:20px;">
        <div style="color:#A78BFA;font-weight:700;font-size:14px;margin-bottom:10px;">Step 1: Configure and Train Model</div>
        <div style="color:#7BA3C4;font-size:13px;line-height:1.7;">
            Select what you want to predict (target) and which features to use as input.
            The model learns statistical patterns between your features and the target variable.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        default_idx = 0
        for i, c in enumerate(non_id_num):
            if any(k in c.lower() for k in REVENUE_KEYS + PROFIT_KEYS):
                default_idx = i
                break
        target_col_p = st.selectbox("Target Variable (what to predict)", non_id_num,
                                     index=default_idx)

    with col_b:
        all_feats = [c for c in non_id_num if c != target_col_p]
        sel_feats = st.multiselect("Feature Columns (predictors)", all_feats,
                                    default=all_feats[:min(8, len(all_feats))])

    if st.button("Train Gradient Boosting Model", type="primary", use_container_width=True):
        if not sel_feats:
            st.error("Select at least one feature column.")
        elif len(df_pred) < 4:
            st.error("Need at least 4 rows of data.")
        else:
            with st.spinner("Training..."):
                train_df = df_pred[sel_feats + [target_col_p]].copy()
                try:
                    model, scaler, feat_cols, metrics = train_model(train_df, target_col_p)
                    st.session_state.model        = model
                    st.session_state.scaler       = scaler
                    st.session_state.feature_cols = feat_cols
                    st.session_state.target_col   = target_col_p
                    st.session_state.train_metrics = metrics
                    st.success(f"Model trained. R² = {metrics['r2']:.4f}  RMSE = {metrics['rmse']:,.2f}")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    if st.session_state.train_metrics:
        tm = st.session_state.train_metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² Score",    f"{tm['r2']:.4f}",
                  delta="Excellent" if tm['r2'] > 0.85 else "Good" if tm['r2'] > 0.65 else "Fair")
        m2.metric("RMSE",        f"{tm['rmse']:,.2f}")
        m3.metric("MAE",         f"{tm['mae']:,.2f}")
        m4.metric("CV R² Score", f"{tm['cv_score']:.4f}" if tm['cv_score'] is not None else "N/A")

    st.markdown("---")

    if st.session_state.model is not None:
        st.markdown("""
        <div class="card card-cyan">
            <div style="color:#38BDF8;font-weight:700;font-size:14px;margin-bottom:8px;">Step 2: Set Input Values and Predict</div>
            <div style="color:#7BA3C4;font-size:13px;">Adjust the sliders to represent your next scenario. The model will predict the target variable.</div>
        </div>
        """, unsafe_allow_html=True)

        feat_inputs = {}
        fc = st.session_state.feature_cols

        for i in range(0, len(fc), 3):
            batch   = fc[i:i+3]
            cols_ui = st.columns(len(batch))
            for j, feat in enumerate(batch):
                with cols_ui[j]:
                    if feat in df_pred.columns and pd.api.types.is_numeric_dtype(df_pred[feat]):
                        mn   = float(df_pred[feat].min())
                        mx   = float(df_pred[feat].max())
                        avg  = float(df_pred[feat].mean())
                        ext  = (mx - mn) * 0.2
                        feat_inputs[feat] = st.slider(
                            feat.replace('_',' ').title(),
                            mn - ext, mx + ext, avg,
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
                st.session_state.last_prediction = {
                    "value":  pred_val,
                    "inputs": feat_inputs,
                    "target": target_col_p
                }

                t_min  = float(df_pred[target_col_p].min())
                t_max  = float(df_pred[target_col_p].max())
                t_rng  = t_max - t_min
                pctile = int(np.clip((pred_val - t_min) / t_rng * 100, 0, 100)) if t_rng > 0 else 50

                color = "#34D399" if pred_val > 0 else "#FB7185"
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#080E1A,#0C1525);
                            border:1px solid rgba(56,189,248,0.2);border-radius:20px;
                            padding:32px;text-align:center;margin:20px 0;
                            box-shadow:0 16px 48px rgba(0,0,0,0.5);">
                    <div style="color:#3D6080;font-size:12px;text-transform:uppercase;
                                letter-spacing:2px;margin-bottom:8px;">
                        Predicted {target_col_p.replace('_',' ').title()}
                    </div>
                    <div style="font-family:'DM Mono';font-size:52px;font-weight:500;
                                color:{color};letter-spacing:-2px;">
                        {pred_val:,.2f}
                    </div>
                    <div style="color:#3D6080;font-size:13px;margin-top:12px;">
                        Approximately the
                        <span style="color:#FBBF24;font-weight:600;">{pctile}th percentile</span>
                        of historical range ({t_min:,.1f} to {t_max:,.1f})
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if hasattr(st.session_state.model, 'feature_importances_'):
                    imps   = st.session_state.model.feature_importances_
                    imp_df = pd.DataFrame({"Feature": fc, "Importance": imps})
                    imp_df = imp_df.sort_values("Importance", ascending=True).tail(10)
                    fig_imp = px.bar(
                        imp_df, x="Importance", y="Feature", orientation='h',
                        title="Feature Importance — What Drives This Prediction",
                        color="Importance",
                        color_continuous_scale=["#A78BFA","#38BDF8","#34D399"]
                    )
                    apply_layout(fig_imp, 340)
                    fig_imp.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_imp, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("Train the model using Step 1 above.")


elif selected_page == "Simulation Lab":
    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge">Simulation Lab</div>
        <h1 class="hero-title">What-If Scenario Engine</h1>
        <p class="hero-subtitle">Model Optimistic, Base, and Pessimistic scenarios. Stress-test financials, compute break-even revenue, and quantify risk before it materialises.</p>
    </div>
    """, unsafe_allow_html=True)

    df_sim = st.session_state.df
    if df_sim is None and st.session_state.manual_entries:
        df_sim = pd.DataFrame(st.session_state.manual_entries)

    if df_sim is None:
        st.warning("Load data first.")
        st.stop()

    rev_col_s  = find_col(df_sim, REVENUE_KEYS)
    cogs_col_s = find_col(df_sim, COST_KEYS)
    opex_col_s = find_col(df_sim, OPEX_KEYS)

    default_rev  = float(df_sim[rev_col_s].mean())  if rev_col_s  else 100000.0
    default_cogs = float(df_sim[cogs_col_s].mean()) if cogs_col_s else default_rev * 0.4
    default_opex = float(df_sim[opex_col_s].mean()) if opex_col_s else default_rev * 0.25

    st.markdown("""
    <div class="card card-amber" style="margin-bottom:20px;">
        <div style="color:#FBBF24;font-weight:700;font-size:14px;margin-bottom:8px;">Scenario Configuration</div>
        <div style="color:#7BA3C4;font-size:13px;line-height:1.7;">
            Set your base financials and apply percentage changes to model three futures.
            The Risk Score uses cost ratio and margin to reflect financial vulnerability.
        </div>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        base_rev  = st.number_input("Base Monthly Revenue (INR)", min_value=0.0,
                                     value=default_rev, step=5000.0, key="sim_rev")
    with sc2:
        base_cogs = st.number_input("Base COGS (INR)", min_value=0.0,
                                     value=default_cogs, step=1000.0, key="sim_cogs")
    with sc3:
        base_opex = st.number_input("Base Operating Expenses (INR)", min_value=0.0,
                                     value=default_opex, step=1000.0, key="sim_opex")

    st.markdown('<div class="section-label" style="margin-top:16px;">Define 3 Scenarios</div>',
                unsafe_allow_html=True)

    scenarios = {}
    sc_configs = [
        ("#38BDF8", "Optimistic",   20,  -10, -5),
        ("#FBBF24", "Base Case",     0,    0,  0),
        ("#FB7185", "Pessimistic", -20,   20, 15),
    ]
    sim_cols = st.columns(3)

    for (clr, sc_name, def_rev, def_cogs, def_opex), ui_col in zip(sc_configs, sim_cols):
        with ui_col:
            st.markdown(f"""
            <div style="background:{clr}0D;border:1px solid {clr}33;border-radius:12px;
                        padding:14px;margin-bottom:12px;">
                <div style="color:{clr};font-weight:700;font-size:13px;margin-bottom:10px;">{sc_name}</div>
            </div>
            """, unsafe_allow_html=True)

            rev_chg  = st.slider("Revenue Change %",  -50, 100, def_rev,  5, key=f"{sc_name}_rev")
            cogs_chg = st.slider("COGS Change %",      -30,  50, def_cogs, 5, key=f"{sc_name}_cogs")
            opex_chg = st.slider("Opex Change %",      -30,  50, def_opex, 5, key=f"{sc_name}_opex")

            sc_rev    = base_rev  * (1 + rev_chg  / 100)
            sc_cogs   = base_cogs * (1 + cogs_chg / 100)
            sc_opex   = base_opex * (1 + opex_chg / 100)

            gross_profit  = sc_rev - sc_cogs
            net_profit    = sc_rev - sc_cogs - sc_opex
            gross_margin  = (gross_profit / sc_rev * 100) if sc_rev > 0 else 0.0
            net_margin    = (net_profit   / sc_rev * 100) if sc_rev > 0 else 0.0
            cost_ratio    = (sc_cogs + sc_opex) / max(sc_rev, 1)
            risk_score    = float(np.clip(
                cost_ratio * 60 + max(0.0, -net_margin) * 0.4,
                0.0, 100.0
            ))

            scenarios[sc_name] = {
                "revenue": sc_rev, "cogs": sc_cogs, "opex": sc_opex,
                "gross_profit": gross_profit, "net_profit": net_profit,
                "gross_margin": gross_margin, "net_margin": net_margin,
                "risk_score": risk_score, "color": clr
            }

            profit_color = clr if net_profit > 0 else "#FB7185"
            st.markdown(f"""
            <div style="background:{clr}0D;border:1px solid {clr}33;border-radius:10px;
                        padding:12px;margin-top:8px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:#7BA3C4;font-size:12px;">Gross Profit</span>
                    <span style="color:{clr};font-family:'DM Mono';font-size:12px;font-weight:600;">
                        {gross_profit:,.0f}
                    </span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:#7BA3C4;font-size:12px;">Net Profit</span>
                    <span style="color:{profit_color};font-family:'DM Mono';font-size:12px;font-weight:600;">
                        {net_profit:,.0f}
                    </span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:#7BA3C4;font-size:12px;">Net Margin</span>
                    <span style="color:{clr};font-family:'DM Mono';font-size:12px;">{net_margin:.1f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#7BA3C4;font-size:12px;">Risk Score</span>
                    <span style="color:{'#34D399' if risk_score<30 else '#FBBF24' if risk_score<60 else '#FB7185'};
                                font-family:'DM Mono';font-size:12px;">{risk_score:.0f} / 100</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    sc_df = pd.DataFrame([
        {
            "Scenario":    name,
            "Revenue":     d["revenue"],
            "Gross Profit":d["gross_profit"],
            "Net Profit":  d["net_profit"],
            "Net Margin %":d["net_margin"],
            "Risk Score":  d["risk_score"],
        }
        for name, d in scenarios.items()
    ])

    ch_s1, ch_s2 = st.columns(2)
    with ch_s1:
        fig_sc = go.Figure()
        for metric, clr2 in [("Revenue","#38BDF8"),("Gross Profit","#A78BFA"),("Net Profit","#34D399")]:
            fig_sc.add_trace(go.Bar(
                name=metric, x=sc_df["Scenario"], y=sc_df[metric],
                marker_color=clr2, opacity=0.85
            ))
        apply_layout(fig_sc, 360)
        fig_sc.update_layout(title="Scenario Comparison — Financials", barmode='group')
        st.plotly_chart(fig_sc, use_container_width=True)

    with ch_s2:
        fig_risk = go.Figure()
        r_colors = [v["color"] for v in scenarios.values()]
        fig_risk.add_trace(go.Bar(
            x=sc_df["Scenario"], y=sc_df["Risk Score"],
            marker_color=r_colors, opacity=0.9,
            text=sc_df["Risk Score"].round(1).astype(str) + " / 100",
            textposition='outside', textfont=dict(color='white', size=12)
        ))
        fig_risk.add_hline(y=50, line_color='#FBBF24', line_dash='dash', line_width=2,
                           annotation_text="Moderate Risk Threshold",
                           annotation_font_color='#FBBF24')
        apply_layout(fig_risk, 360)
        fig_risk.update_layout(title="Risk Score by Scenario", yaxis_range=[0, 115])
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Break-Even Analysis</div>', unsafe_allow_html=True)

    vcr        = base_cogs / max(base_rev, 1)
    be_revenue = base_opex / max(1 - vcr, 0.001)
    rev_range  = np.linspace(0, max(base_rev * 2.5, be_revenue * 1.5), 300)
    tc_line    = base_opex + vcr * rev_range

    fig_be = go.Figure()
    fig_be.add_trace(go.Scatter(
        x=rev_range, y=rev_range,
        name="Revenue", line=dict(color="#38BDF8", width=2.5)
    ))
    fig_be.add_trace(go.Scatter(
        x=rev_range, y=tc_line,
        name="Total Costs", line=dict(color="#FB7185", width=2.5),
        fill='tonexty', fillcolor='rgba(251,113,133,0.05)'
    ))
    fig_be.add_vline(x=be_revenue, line_color='#34D399', line_dash='dash', line_width=2,
                     annotation_text=f"Break-even: {be_revenue:,.0f}",
                     annotation_font_color='#34D399', annotation_font_size=13)
    fig_be.add_vline(x=base_rev, line_color='#FBBF24', line_dash='dot', line_width=2,
                     annotation_text=f"Current: {base_rev:,.0f}",
                     annotation_font_color='#FBBF24', annotation_font_size=12)
    apply_layout(fig_be, 360)
    fig_be.update_layout(
        title="Break-Even Chart",
        xaxis_title="Revenue (INR)", yaxis_title="Amount (INR)",
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#7BA3C4'))
    )
    st.plotly_chart(fig_be, use_container_width=True)

    sim_summary = (
        " | ".join([
            f"{n}: rev={d['revenue']:,.0f}, net_profit={d['net_profit']:,.0f}, "
            f"net_margin={d['net_margin']:.1f}%, risk={d['risk_score']:.0f}/100"
            for n, d in scenarios.items()
        ])
        + f" | Break-even revenue: {be_revenue:,.0f}"
    )
    st.session_state.last_sim_results = sim_summary

    st.markdown("---")
    st.markdown('<div class="section-label">AI Scenario Analyst</div>', unsafe_allow_html=True)

    qp1, qp2, qp3 = st.columns(3)
    quick_q = None
    if qp1.button("Which scenario to target?",     use_container_width=True):
        quick_q = "Based on these 3 scenarios, which should I realistically target and why? What is the key lever to pull?"
    if qp2.button("Explain the risk scores",       use_container_width=True):
        quick_q = "Explain what the risk scores mean for each scenario. What specific risks should I watch for?"
    if qp3.button("How to reach break-even faster", use_container_width=True):
        quick_q = "Given my break-even figure, what are the 3 fastest strategies to reach it? Be specific about numbers."

    if quick_q:
        st.session_state.sim_chat_history.append({"role":"user","content":quick_q})

    for msg in st.session_state.sim_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if (st.session_state.sim_chat_history and
            st.session_state.sim_chat_history[-1]["role"] == "user"):
        needs_reply = (
            len(st.session_state.sim_chat_history) == 1 or
            st.session_state.sim_chat_history[-2]["role"] != "assistant"
        )
        if needs_reply:
            with st.chat_message("assistant"):
                with st.spinner("Analysing..."):
                    ctx  = f"Simulation: {sim_summary} | Break-even: {be_revenue:,.0f}"
                    resp = gemini_call(st.session_state.sim_chat_history, ctx)
                st.markdown(resp)
            st.session_state.sim_chat_history.append({"role":"assistant","content":resp})

    if chat_in := st.chat_input("Ask about these scenarios...", key="sim_chat"):
        st.session_state.sim_chat_history.append({"role":"user","content":chat_in})
        with st.chat_message("user"):
            st.markdown(chat_in)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ctx  = f"Simulation: {sim_summary}"
                resp = gemini_call(st.session_state.sim_chat_history, ctx)
            st.markdown(resp)
        st.session_state.sim_chat_history.append({"role":"assistant","content":resp})


elif selected_page == "Visualisations":
    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge">Analytics</div>
        <h1 class="hero-title">Deep Financial Visualisations</h1>
        <p class="hero-subtitle">Explore trends, correlations, and distributions across your business data.</p>
    </div>
    """, unsafe_allow_html=True)

    df_vis = st.session_state.df
    if df_vis is None and st.session_state.manual_entries:
        df_vis = pd.DataFrame(st.session_state.manual_entries)

    if df_vis is None:
        st.warning("Load data first.")
        st.stop()

    num_v = df_vis.select_dtypes(include=np.number).columns.tolist()
    cat_v = df_vis.select_dtypes(include='object').columns.tolist()

    id_keywords = ['_id', 'order_id', 'customer_id', 'transaction_id', 'invoice']
    clean_cat_v = [
        c for c in cat_v
        if not any(kw in c.lower() for kw in id_keywords)
    ]

    tab1, tab2, tab3 = st.tabs(["Trends", "Correlations", "Distributions"])

    with tab1:
        if num_v:
            v1, v2 = st.columns([1, 3])
            with v1:
                plot_cols = st.multiselect(
                    "Select Metrics", num_v,
                    default=num_v[:min(3, len(num_v))],
                    max_selections=6,
                    key="vis_plot_cols"
                )
                chart_type = st.radio(
                    "Chart Type", ["Line", "Area", "Bar"],
                    key="vis_chart_type"
                )
                x_options = ["Row Index"] + clean_cat_v + num_v
                time_col_v = st.selectbox(
                    "X-Axis", x_options,
                    key="vis_x_axis"
                )
            with v2:
                if plot_cols:
                    x_val = (
                        df_vis.index
                        if time_col_v == "Row Index"
                        else df_vis[time_col_v].astype(str)
                    )
                    fig_t = go.Figure()
                    for idx2, col2 in enumerate(plot_cols):
                        c2 = PALETTE[idx2 % len(PALETTE)]
                        if chart_type == "Line":
                            fig_t.add_trace(go.Scatter(
                                x=x_val, y=df_vis[col2], name=col2,
                                line=dict(color=c2, width=2.5),
                                mode='lines+markers', marker=dict(size=5)
                            ))
                        elif chart_type == "Area":
                            fig_t.add_trace(go.Scatter(
                                x=x_val, y=df_vis[col2], name=col2,
                                fill='tozeroy', line=dict(color=c2, width=2),
                                fillcolor=c2 + '14'
                            ))
                        else:
                            fig_t.add_trace(go.Bar(
                                x=x_val, y=df_vis[col2], name=col2,
                                marker_color=c2, opacity=0.85
                            ))
                    apply_layout(fig_t, 440)
                    fig_t.update_layout(
                        title="Trend Analysis",
                        legend=dict(
                            bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#7BA3C4')
                        )
                    )
                    st.plotly_chart(fig_t, use_container_width=True)
                else:
                    st.info("Select at least one metric from the left panel.")
        else:
            st.info("No numeric columns found in the dataset.")

    with tab2:
        if len(num_v) >= 2:
            corr = df_vis[num_v].corr()
            fig_hm = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale=["#FB7185", "#080E1A", "#38BDF8"],
                zmin=-1, zmax=1,
                title="Pearson Correlation Matrix"
            )
            apply_layout(fig_hm, 520)
            fig_hm.update_traces(textfont=dict(size=10, color='white'))
            st.plotly_chart(fig_hm, use_container_width=True)

            st.markdown(
                '<div class="section-label" style="margin-top:8px;">Scatter Explorer</div>',
                unsafe_allow_html=True
            )
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                x_sc = st.selectbox(
                    "X Axis", num_v, index=0, key="vis_sc_x"
                )
            with sc2:
                y_sc = st.selectbox(
                    "Y Axis", num_v,
                    index=min(1, len(num_v) - 1),
                    key="vis_sc_y"
                )
            with sc3:
                colour_options = ["None"] + clean_cat_v
                color_sc = st.selectbox(
                    "Colour By", colour_options, key="vis_sc_col"
                )

            fig_sc2 = px.scatter(
                df_vis, x=x_sc, y=y_sc,
                color=color_sc if color_sc != "None" else None,
                color_discrete_sequence=PALETTE,
                trendline='ols' if len(df_vis) > 5 else None,
                title=f"{x_sc} vs {y_sc}"
            )
            apply_layout(fig_sc2, 380)
            st.plotly_chart(fig_sc2, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns.")

    with tab3:
        if num_v:
            d1, d2 = st.columns([1, 3])
            with d1:
                dist_col = st.selectbox(
                    "Select Column", num_v, key="vis_dist_col"
                )
                show_box = st.checkbox(
                    "Show box plot overlay", True, key="vis_show_box"
                )
            with d2:
                fig_dist = px.histogram(
                    df_vis, x=dist_col,
                    nbins=25,
                    marginal="box" if show_box else None,
                    color_discrete_sequence=['#A78BFA'],
                    title=f"Distribution — {dist_col.replace('_', ' ').title()}"
                )
                fig_dist.update_traces(marker_line_width=0, opacity=0.85)
                fig_dist.add_vline(
                    x=float(df_vis[dist_col].mean()),
                    line_color='#FBBF24', line_dash='dash', line_width=2,
                    annotation_text="Mean",
                    annotation_font_color='#FBBF24'
                )
                fig_dist.add_vline(
                    x=float(df_vis[dist_col].median()),
                    line_color='#34D399', line_dash='dot', line_width=2,
                    annotation_text="Median",
                    annotation_font_color='#34D399'
                )
                apply_layout(fig_dist, 400)
                st.plotly_chart(fig_dist, use_container_width=True)

            stats = df_vis[dist_col].describe()
            iqr   = float(stats['75%'] - stats['25%'])
            lo    = float(stats['25%']) - 1.5 * iqr
            hi    = float(stats['75%']) + 1.5 * iqr
            n_out = int(((df_vis[dist_col] < lo) | (df_vis[dist_col] > hi)).sum())

            stat_pills = ''.join([
                f'<div style="background:rgba(56,189,248,0.06);border-radius:8px;padding:8px 14px;">'
                f'<div style="color:#3D6080;font-size:10px;text-transform:uppercase;'
                f'letter-spacing:1px;">{k}</div>'
                f'<div style="color:#38BDF8;font-family:DM Mono;font-size:14px;'
                f'font-weight:500;">{v:.3f}</div></div>'
                for k, v in stats.items()
            ])
            st.markdown(f"""
            <div class="card card-cyan" style="margin-top:8px;">
                <div class="section-label">Statistical Summary — {dist_col}</div>
                <div style="display:flex;flex-wrap:wrap;gap:12px;">
                    {stat_pills}
                    <div style="background:rgba(251,113,133,0.06);border-radius:8px;padding:8px 14px;">
                        <div style="color:#3D6080;font-size:10px;text-transform:uppercase;letter-spacing:1px;">IQR Outliers</div>
                        <div style="color:#FB7185;font-family:DM Mono;font-size:14px;font-weight:500;">{n_out}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No numeric columns found.")


elif selected_page == "AI Insights":
    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge">Powered by Gemini</div>
        <h1 class="hero-title">AI Business Consultant</h1>
        <p class="hero-subtitle">Ask anything about your data — from profitability to scale-up strategy. The AI has full context of your dataset, model performance, and simulation results.</p>
    </div>
    """, unsafe_allow_html=True)

    df_ai = st.session_state.df
    if df_ai is None and st.session_state.manual_entries:
        df_ai = pd.DataFrame(st.session_state.manual_entries)

    data_ctx   = "No data loaded."
    if df_ai is not None:
        num_ai = df_ai.select_dtypes(include=np.number)
        stats  = {}
        for c in num_ai.columns[:8]:
            stats[c] = {
                "mean": round(float(num_ai[c].mean()), 2),
                "max":  round(float(num_ai[c].max()),  2),
                "min":  round(float(num_ai[c].min()),  2),
            }
        data_ctx = (f"Dataset: {df_ai.shape[0]} rows, {df_ai.shape[1]} columns. "
                    f"Stats: {json.dumps(stats)}")

    model_ctx = ""
    if st.session_state.train_metrics:
        tm        = st.session_state.train_metrics
        model_ctx = (f"Model: GradientBoosting, Target={st.session_state.target_col}, "
                     f"R²={tm['r2']:.4f}, RMSE={tm['rmse']:.2f}, MAE={tm['mae']:.2f}")

    pred_ctx = ""
    if st.session_state.last_prediction:
        p        = st.session_state.last_prediction
        pred_ctx = (f"Last prediction: {p['target']}={p['value']:.2f}, "
                    f"Inputs={json.dumps({k:round(float(v),2) for k,v in list(p['inputs'].items())[:6]})}")

    sim_ctx  = st.session_state.last_sim_results or ""
    full_ctx = f"{data_ctx} | {model_ctx} | {pred_ctx} | {sim_ctx}"

    st.markdown('<div class="section-label">Generate Full Business Report</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card card-violet" style="margin-bottom:16px;">
        <div style="color:#A78BFA;font-weight:600;font-size:14px;margin-bottom:6px;">Executive Report</div>
        <div style="color:#7BA3C4;font-size:13px;line-height:1.7;">
            Generates a comprehensive business analysis — financials, performance, risks, and
            recommendations — written in plain English for any non-technical audience.
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
                "Use bold for key numbers. Total length ~500 words.\n\n"
                f"SESSION DATA:\n{full_ctx}\n\n"
                "Sections:\n"
                "## Executive Summary\n"
                "## Financial Performance\n"
                "## Growth and Trends\n"
                "## Risk Assessment\n"
                "## Model Insights\n"
                "## Recommendations\n\n"
                "End with a 1-sentence Shark Tank pitch verdict."
            )
            with st.spinner("Writing report..."):
                report = gemini_call([{"role":"user","content":report_prompt}], full_ctx)

            st.markdown(f"""
            <div class="card card-cyan" style="margin-top:12px;">
                <div style="color:#38BDF8;font-weight:700;font-size:13px;margin-bottom:16px;
                            font-family:'Syne';letter-spacing:0.5px;">
                    SharkLens Business Intelligence Report
                </div>
                <div style="color:#CBD5E0;font-size:14px;line-height:1.9;">
                    {report.replace(chr(10),'<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.download_button("Download Report (.txt)", report,
                               "sharklens_report.txt", "text/plain",
                               use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Quick Analysis Prompts</div>', unsafe_allow_html=True)

    qp1, qp2, qp3, qp4 = st.columns(4)
    quick_p = None
    if qp1.button("Am I profitable?",         use_container_width=True):
        quick_p = "Based on my data, am I profitable? Give specific numbers and tell me if my margins are healthy."
    if qp2.button("Predict next month",       use_container_width=True):
        quick_p = "Based on trends and any predictions made, what is my most likely outcome next month?"
    if qp3.button("Top 3 cost-cutting tips",  use_container_width=True):
        quick_p = "What are my top 3 specific opportunities to cut costs or improve margins, based on my numbers?"
    if qp4.button("Shark Tank verdict",       use_container_width=True):
        quick_p = "If I walked into Shark Tank with these numbers, what would the sharks say? Be brutally honest."

    if quick_p:
        st.session_state.chat_history.append({"role":"user","content":quick_p})

    st.markdown("---")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if (st.session_state.chat_history and
            st.session_state.chat_history[-1]["role"] == "user"):
        needs_reply = (
            len(st.session_state.chat_history) == 1 or
            st.session_state.chat_history[-2]["role"] != "assistant"
        )
        if needs_reply:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = gemini_call(st.session_state.chat_history, full_ctx)
                st.markdown(reply)
            st.session_state.chat_history.append({"role":"assistant","content":reply})

    if user_msg := st.chat_input("Ask anything about your business data..."):
        st.session_state.chat_history.append({"role":"user","content":user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Consulting..."):
                ai_resp = gemini_call(st.session_state.chat_history, full_ctx)
            st.markdown(ai_resp)
        st.session_state.chat_history.append({"role":"assistant","content":ai_resp})

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()