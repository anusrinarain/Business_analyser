# ══════════════════════════════════════════════════════════════════════════════
#  VentureScope  —  Business Intelligence & Prediction Platform  v2.0
#  Landing page → data input → full analytics + LLM insights
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VentureScope · Business Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── COLOUR TOKENS ─────────────────────────────────────────────────────────────
C = {
    "bg":      "#080B12",
    "surface": "#0D1117",
    "card":    "#111820",
    "card2":   "#141C28",
    "border":  "#1C2538",
    "border2": "#242E44",
    "text":    "#E2E8F5",
    "muted":   "#5C6E8A",
    "dim":     "#3A4A62",
    "accent":  "#C9963A",   # amber gold
    "blue":    "#3D7FE8",
    "green":   "#2EB87A",
    "red":     "#D94F63",
    "purple":  "#8B5CF6",
    "teal":    "#0EA5A0",
    "orange":  "#E8744A",
    "ch1": "#3D7FE8", "ch2": "#C9963A", "ch3": "#2EB87A",
    "ch4": "#D94F63", "ch5": "#8B5CF6", "ch6": "#E8744A", "ch7": "#0EA5A0",
}
PAL = [C["ch1"],C["ch2"],C["ch3"],C["ch4"],C["ch5"],C["ch6"],C["ch7"]]

MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
MONTHS_S = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

BENCHMARKS = {
    "Retail":        {"net":5,  "gross":40, "ebitda":8,  "rent":8,  "salary":15,"mkt":4},
    "Food&Beverage": {"net":7,  "gross":65, "ebitda":12, "rent":10, "salary":30,"mkt":3},
    "Services":      {"net":15, "gross":80, "ebitda":20, "rent":5,  "salary":40,"mkt":6},
    "Manufacturing": {"net":8,  "gross":45, "ebitda":14, "rent":6,  "salary":20,"mkt":2},
    "E-Commerce":    {"net":10, "gross":50, "ebitda":15, "rent":1,  "salary":15,"mkt":15},
    "Healthcare":    {"net":12, "gross":55, "ebitda":18, "rent":7,  "salary":45,"mkt":3},
    "Technology":    {"net":18, "gross":72, "ebitda":22, "rent":4,  "salary":50,"mkt":10},
    "Education":     {"net":13, "gross":70, "ebitda":17, "rent":6,  "salary":50,"mkt":5},
}

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family:'Plus Jakarta Sans',sans-serif !important;
    color:{C["text"]} !important;
    background:{C["bg"]} !important;
}}
.main {{ background:{C["bg"]} !important; }}
.block-container {{ padding:1.5rem 2rem 3rem !important; max-width:1480px; }}
[data-testid="stSidebar"] {{
    background:{C["surface"]} !important;
    border-right:1px solid {C["border"]} !important;
}}
[data-testid="stSidebar"] * {{ color:{C["text"]} !important; }}
#MainMenu,footer,header {{ visibility:hidden; }}
[data-testid="stDecoration"] {{ display:none; }}

/* ── KPI CARDS ── */
.kpi {{
    background:{C["card"]}; border:1px solid {C["border"]};
    border-radius:16px; padding:20px 18px 16px;
    position:relative; overflow:hidden;
    transition:border-color .2s,transform .15s;
    margin-bottom:10px; height:108px;
}}
.kpi:hover {{ border-color:{C["accent"]}44; transform:translateY(-2px); }}
.kpi-accent {{ position:absolute; top:0; left:0; right:0; height:2px; }}
.kpi-label {{ font-size:9.5px; font-weight:700; letter-spacing:1.8px;
    text-transform:uppercase; color:{C["muted"]}; margin-bottom:8px; }}
.kpi-val {{ font-size:25px; font-weight:700;
    font-family:'JetBrains Mono',monospace; color:{C["text"]}; line-height:1.1; }}
.kpi-delta {{ font-size:11px; font-weight:500; margin-top:6px; }}
.kpi-sub {{ font-size:10px; color:{C["muted"]}; margin-top:4px; }}

/* ── SECTION LABEL ── */
.sec {{ font-size:10px; font-weight:700; letter-spacing:2.2px;
    text-transform:uppercase; color:{C["muted"]};
    margin:32px 0 14px; padding-bottom:8px;
    border-bottom:1px solid {C["border"]}; }}

/* ── PANELS ── */
.panel {{
    background:{C["card"]}; border:1px solid {C["border"]};
    border-radius:14px; padding:20px; margin-bottom:14px;
}}
.panel-blue  {{ border-left:3px solid {C["blue"]}  !important; }}
.panel-green {{ border-left:3px solid {C["green"]} !important; }}
.panel-gold  {{ border-left:3px solid {C["accent"]}!important; }}
.panel-red   {{ border-left:3px solid {C["red"]}   !important; }}

/* ── ALERTS ── */
.alert {{
    border-radius:0 10px 10px 0; padding:13px 18px;
    margin:10px 0; font-size:13.5px; line-height:1.7; color:{C["text"]};
}}
.alert-info  {{ background:{C["blue"]}0F;  border-left:3px solid {C["blue"]}; }}
.alert-ok    {{ background:{C["green"]}0F; border-left:3px solid {C["green"]}; }}
.alert-warn  {{ background:{C["accent"]}0F;border-left:3px solid {C["accent"]}; }}
.alert-bad   {{ background:{C["red"]}0F;   border-left:3px solid {C["red"]}; }}

/* ── PILLS ── */
.pill {{
    display:inline-flex; align-items:center;
    border-radius:20px; padding:3px 12px;
    font-size:11px; font-weight:600; letter-spacing:.3px;
    margin:2px 4px 2px 0;
}}
.pill-g {{ background:{C["green"]}18; border:1px solid {C["green"]}44; color:{C["green"]}; }}
.pill-r {{ background:{C["red"]}18;   border:1px solid {C["red"]}44;   color:{C["red"]}; }}
.pill-a {{ background:{C["accent"]}18;border:1px solid {C["accent"]}44;color:{C["accent"]}; }}
.pill-b {{ background:{C["blue"]}18;  border:1px solid {C["blue"]}44;  color:{C["blue"]}; }}

/* ── BUTTONS ── */
.stButton > button {{
    background:{C["accent"]} !important; color:#07090F !important;
    border:none !important; border-radius:10px !important;
    font-weight:700 !important; font-family:'Plus Jakarta Sans',sans-serif !important;
    padding:10px 24px !important; transition:opacity .18s,transform .12s !important;
    letter-spacing:.2px;
}}
.stButton > button:hover {{ opacity:.85 !important; transform:translateY(-1px) !important; }}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{
    background:{C["card"]}; border-radius:10px; padding:3px; gap:2px;
    border:1px solid {C["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius:8px; padding:7px 16px;
    font-weight:500; font-size:12.5px; color:{C["muted"]};
}}
.stTabs [aria-selected="true"] {{
    background:{C["bg"]} !important; color:{C["text"]} !important;
    box-shadow:0 1px 8px rgba(0,0,0,.5);
}}

/* ── INPUTS ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input, textarea {{
    background:{C["bg"]} !important; border:1px solid {C["border2"]} !important;
    color:{C["text"]} !important; border-radius:8px !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
textarea:focus {{ border-color:{C["accent"]}88 !important; }}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {{
    background:{C["card"]}; border:1px solid {C["border"]}; border-radius:10px;
}}
/* ── PROGRESS ── */
.stProgress > div > div {{
    background:linear-gradient(90deg,{C["blue"]},{C["accent"]}) !important;
    border-radius:4px !important;
}}
/* ── EXPANDER ── */
[data-testid="stExpander"] {{
    background:{C["card"]} !important;
    border:1px solid {C["border"]} !important;
    border-radius:10px !important;
}}
/* ── LOGO ── */
.vs-logo {{
    font-family:'Plus Jakarta Sans',sans-serif;
    font-size:20px; font-weight:800; color:{C["text"]}; letter-spacing:-.5px;
}}
.vs-logo span {{ color:{C["accent"]}; }}
/* ── PAGE HEAD ── */
.ph {{ margin-bottom:24px; padding-bottom:14px; border-bottom:1px solid {C["border"]}; }}
.ph-title {{
    font-size:22px; font-weight:800; color:{C["text"]}; letter-spacing:-.3px; margin:0 0 4px; }}
.ph-sub {{ font-size:13px; color:{C["muted"]}; margin:0; }}

/* ── LANDING PAGE ── */
.hero {{
    background:linear-gradient(135deg,{C["card"]} 0%,{C["card2"]} 100%);
    border:1px solid {C["border"]}; border-radius:20px;
    padding:48px 40px; margin-bottom:24px;
    position:relative; overflow:hidden;
}}
.hero::before {{
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,{C["blue"]},{C["accent"]},{C["purple"]});
}}
.hero-title {{
    font-size:38px; font-weight:800; color:{C["text"]};
    letter-spacing:-1px; line-height:1.15; margin-bottom:12px;
}}
.hero-title span {{ color:{C["accent"]}; }}
.hero-sub {{
    font-size:15px; color:{C["muted"]}; line-height:1.7;
    max-width:560px; margin-bottom:28px;
}}
.feature-card {{
    background:{C["bg"]}; border:1px solid {C["border"]};
    border-radius:12px; padding:18px 16px;
    transition:border-color .18s,transform .15s;
}}
.feature-card:hover {{ border-color:{C["accent"]}44; transform:translateY(-2px); }}
.feature-icon {{
    width:36px; height:36px; border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-size:16px; margin-bottom:10px;
}}
.feature-title {{ font-size:13px; font-weight:700; color:{C["text"]}; margin-bottom:4px; }}
.feature-desc  {{ font-size:11.5px; color:{C["muted"]}; line-height:1.55; }}

/* ── AI OUTPUT BOX ── */
.ai-box {{
    background:{C["card2"]}; border:1px solid {C["border2"]};
    border-left:3px solid {C["blue"]};
    border-radius:0 12px 12px 0;
    padding:18px 20px; margin:12px 0;
    font-size:14px; line-height:1.8; color:{C["text"]};
}}
.ai-box h2,h3 {{ color:{C["text"]} !important; font-family:'Plus Jakarta Sans',sans-serif !important; }}
.ai-header {{
    display:flex; align-items:center; gap:10px;
    margin-bottom:12px;
}}
.ai-dot {{
    width:8px; height:8px; border-radius:50%;
    background:{C["blue"]}; flex-shrink:0;
    box-shadow:0 0 8px {C["blue"]}88;
}}

/* ── SCORECARD ROW ── */
.sc-row {{
    display:flex; justify-content:space-between; align-items:center;
    padding:11px 16px; border-bottom:1px solid {C["border"]};
    border-radius:6px; margin-bottom:3px;
}}
.sc-row:last-child {{ border-bottom:none; }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════
_defaults = {
    "page": "Home", "data": None, "ready": False,
    "biz_name": "", "biz_type": "Retail", "founded": 2020,
    "mc_results": None, "scenarios": {},
    "gemini_key": "", "llm_ok": None,
    "chat_hist": [],   # for AI chat
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Load key from secrets silently — never expose in UI
try:
    _k = st.secrets.get("GEMINI_API_KEY", "")
    if _k and not st.session_state.gemini_key:
        st.session_state.gemini_key = _k
except Exception:
    pass

# ═══════════════════════════════════════════════════════════
#  PURE HELPERS
# ═══════════════════════════════════════════════════════════
def fmt(v):
    s = "−" if v < 0 else ""
    v = abs(v)
    if v >= 1e7: return f"{s}₹{v/1e7:.2f}Cr"
    if v >= 1e5: return f"{s}₹{v/1e5:.2f}L"
    if v >= 1e3: return f"{s}₹{v/1e3:.1f}K"
    return f"{s}₹{v:,.0f}"

def pct(v): return f"{v:.1f}%"

def mom_delta(df, col):
    if len(df) < 2: return None
    p = df[col].iloc[-2]
    return (df[col].iloc[-1] - p) / abs(p) * 100 if p != 0 else None

def calc_metrics(df):
    df = df.copy()
    df["Gross_Profit"]  = df["Revenue"] - df["COGS"]
    df["Gross_Margin"]  = (df["Gross_Profit"] / df["Revenue"] * 100).round(2)
    df["EBITDA"]        = df["Revenue"] - df["COGS"] - df["Salaries"] - df["Marketing"] - df["Misc"]
    df["EBITDA_Margin"] = (df["EBITDA"]   / df["Revenue"] * 100).round(2)
    df["Net_Profit"]    = df["Revenue"] - df["Total_Expenses"]
    df["Net_Margin"]    = (df["Net_Profit"] / df["Revenue"] * 100).round(2)
    df["Burn_Rate"]     = (df["Total_Expenses"] - df["Revenue"]).clip(lower=0)
    df["OpEx_Pct"]      = (df["Total_Expenses"] / df["Revenue"] * 100).round(2)
    df["Rent_Pct"]      = (df["Rent"]      / df["Revenue"] * 100).round(2)
    df["Salary_Pct"]    = (df["Salaries"]  / df["Revenue"] * 100).round(2)
    df["Mkt_Pct"]       = (df["Marketing"] / df["Revenue"] * 100).round(2)
    return df

def sample_data():
    np.random.seed(42)
    rows = []
    base = 220_000
    for i, m in enumerate(MONTHS):
        rev  = max(base + i*5500 + np.random.normal(0, 15000), 0)
        rent = 20_000
        sal  = max(rev*.23 + np.random.normal(0,2000), 0)
        util = max(5000 + np.random.normal(0,500), 0)
        mkt  = max(rev*.05 + np.random.normal(0,1000), 0)
        cogs = max(rev*.36 + np.random.normal(0,5000), 0)
        misc = max(np.random.uniform(3000, 8000), 0)
        tot  = rent+sal+util+mkt+cogs+misc
        rows.append(dict(Month=m,Revenue=rev,Rent=rent,Salaries=sal,
                         Utilities=util,Marketing=mkt,COGS=cogs,
                         Misc=misc,Total_Expenses=tot))
    return pd.DataFrame(rows)

def cb(fig, title="", h=360):
    """Apply consistent dark chart base."""
    fig.update_layout(
        height=h, paper_bgcolor=C["card"], plot_bgcolor=C["bg"],
        font=dict(family="Plus Jakarta Sans",color=C["muted"],size=12),
        margin=dict(l=10,r=10,t=46,b=16),
        title=dict(text=title,font=dict(size=12.5,color=C["text"],
                   family="Plus Jakarta Sans"),x=0.01,y=0.98),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=11,color=C["text"]),
                    orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
        xaxis=dict(showgrid=False,linecolor=C["border"],
                   tickfont=dict(color=C["muted"],size=11)),
        yaxis=dict(showgrid=True,gridcolor=C["border"]+"44",
                   linecolor=C["border"],tickfont=dict(color=C["muted"],size=11)),
    )
    return fig

def kpi_card(col, label, value, *, delta=None, color=None, sub=None):
    bar = color or C["accent"]
    d_html = ""
    if delta is not None:
        dc = C["green"] if delta >= 0 else C["red"]
        d_html = f'<div class="kpi-delta" style="color:{dc}">{"+" if delta>=0 else ""}{delta:.1f}% MoM</div>'
    s_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    with col:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-accent" style="background:{bar}"></div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-val">{value}</div>
            {d_html}{s_html}
        </div>""", unsafe_allow_html=True)

def ph(title, sub=""):
    st.markdown(f"""
    <div class="ph">
        <div class="ph-title">{title}</div>
        <p class="ph-sub">{sub}</p>
    </div>""", unsafe_allow_html=True)

def sec(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)

def need_data():
    if not st.session_state.ready:
        st.markdown("""<div class="alert alert-warn">
            No data loaded yet. Use the sidebar to enter data, upload a file,
            or click <strong>Load Sample Data</strong> to explore with demo numbers.
        </div>""", unsafe_allow_html=True)
        if st.button("Load Sample Data Now", key="quick_sample"):
            st.session_state.data    = calc_metrics(sample_data())
            st.session_state.ready   = True
            st.session_state.biz_name= "Sample Retail Co."
            st.session_state.biz_type= "Retail"
            st.rerun()
        return True
    return False

# ═══════════════════════════════════════════════════════════
#  LLM  — Gemini 2.0 Flash
# ═══════════════════════════════════════════════════════════
def call_llm(prompt: str, system: str = "") -> str:
    key = st.session_state.gemini_key
    if not key:
        return "**No API key configured.** Add your Gemini key in the sidebar."
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system if system else (
                "You are a sharp financial advisor for Indian small and medium businesses. "
                "You speak plainly, use ₹ for currency, cite specific numbers, "
                "and give actionable advice. Be concise — no filler words."
            )
        )
        resp = model.generate_content(prompt)
        st.session_state.llm_ok = True
        return resp.text
    except Exception as e:
        st.session_state.llm_ok = False
        err = str(e)
        # Friendly error messages
        if "API_KEY" in err.upper() or "invalid" in err.lower() or "api key" in err.lower():
            return "**Invalid API key.** Double-check your key at https://aistudio.google.com/app/apikey"
        if "quota" in err.lower() or "429" in err:
            return "**Rate limit hit.** Wait a moment and try again."
        if "not found" in err.lower() or "404" in err:
            return "**Model not available.** Ensure your key has access to Gemini 2.0 Flash."
        return f"**AI error:** {err[:200]}"

def biz_context() -> str:
    """Build a compact data context string for LLM calls."""
    if not st.session_state.ready:
        return "No business data loaded."
    df   = st.session_state.data
    btyp = st.session_state.biz_type
    b    = BENCHMARKS[btyp]
    last = df.iloc[-1]
    return f"""
Business: {st.session_state.biz_name or 'Unnamed'} | Sector: {btyp} | {len(df)} months data
Annual Revenue (run-rate): {fmt(df['Revenue'].mean()*12)}
Gross Margin: {df['Gross_Margin'].mean():.1f}% (industry avg: {b['gross']}%)
EBITDA Margin: {df['EBITDA_Margin'].mean():.1f}% (industry avg: {b['ebitda']}%)
Net Margin: {df['Net_Margin'].mean():.1f}% (industry avg: {b['net']}%)
Profitable months: {(df['Net_Profit']>0).sum()}/{len(df)}
Latest month ({last['Month']}): Revenue {fmt(last['Revenue'])} | Net Profit {fmt(last['Net_Profit'])}
Avg monthly burn rate: {fmt(df['Burn_Rate'].mean()) if df['Burn_Rate'].mean()>0 else 'None'}
Total net profit (period): {fmt(df['Net_Profit'].sum())}
""".strip()

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style="padding:20px 16px 14px;">
        <div class="vs-logo">Venture<span>Scope</span></div>
        <div style="font-size:10px;color:{C['muted']};margin-top:4px;letter-spacing:1px;">
            BUSINESS INTELLIGENCE
        </div>
    </div>
    <div style="height:1px;background:{C['border']};margin:0 0 10px;"></div>
    """, unsafe_allow_html=True)

    # ── Quick data loader in sidebar ──────────────────────────
    if not st.session_state.ready:
        with st.expander("Load Data", expanded=True):
            method = st.radio("", ["Sample Data","Upload CSV"], label_visibility="collapsed")
            if method == "Sample Data":
                if st.button("Load Sample Dataset", use_container_width=True):
                    st.session_state.data     = calc_metrics(sample_data())
                    st.session_state.ready    = True
                    st.session_state.biz_name = "Sample Retail Co."
                    st.session_state.biz_type = "Retail"
                    if st.session_state.page == "Home":
                        st.session_state.page = "Overview"
                    st.rerun()
            else:
                up = st.file_uploader("CSV/Excel", type=["csv","xlsx"], label_visibility="collapsed")
                if up:
                    try:
                        raw = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
                        # normalise columns
                        needed = ["Revenue","Rent","Salaries","Utilities","Marketing","COGS","Misc"]
                        for col in needed:
                            if col not in raw.columns: raw[col] = 0
                        if "Total_Expenses" not in raw.columns:
                            raw["Total_Expenses"] = raw[needed[1:]].sum(axis=1)
                        if "Month" not in raw.columns:
                            raw.insert(0,"Month",[MONTHS[i%12] for i in range(len(raw))])
                        for col in needed+["Total_Expenses","Revenue"]:
                            raw[col] = pd.to_numeric(raw[col],errors="coerce").fillna(0)
                        st.session_state.data  = calc_metrics(raw)
                        st.session_state.ready = True
                        if st.session_state.page == "Home":
                            st.session_state.page = "Overview"
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Error: {ex}")
    else:
        # Business card when data loaded
        df_s = st.session_state.data
        tot_r = df_s["Revenue"].sum()
        tot_p = df_s["Net_Profit"].sum()
        pc = C["green"] if tot_p >= 0 else C["red"]
        st.markdown(f"""
        <div style="background:{C['bg']};border:1px solid {C['border']};border-radius:12px;
                    padding:14px 16px;margin:4px 0 16px;">
            <div style="font-size:9.5px;color:{C['muted']};letter-spacing:1.2px;
                        text-transform:uppercase;margin-bottom:5px;">
                {st.session_state.biz_name or 'My Business'}
            </div>
            <div style="font-size:12px;font-weight:600;color:{C['text']};margin-bottom:10px;">
                {st.session_state.biz_type}
            </div>
            <div style="display:flex;justify-content:space-between;">
                <div>
                    <div style="font-size:9px;color:{C['muted']}">Revenue</div>
                    <div style="font-size:13px;font-weight:700;color:{C['text']};
                                font-family:'JetBrains Mono'">{fmt(tot_r)}</div>
                </div>
                <div>
                    <div style="font-size:9px;color:{C['muted']}">Net Profit</div>
                    <div style="font-size:13px;font-weight:700;color:{pc};
                                font-family:'JetBrains Mono'">{fmt(tot_p)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────
    st.markdown(f'<div style="font-size:9px;font-weight:700;letter-spacing:2px;color:{C["dim"]};margin:6px 0 8px 4px;">ANALYTICS</div>',
                unsafe_allow_html=True)
    pages = [
        ("Overview",           "Dashboard & KPIs"),
        ("Income Statement",   "Full P&L"),
        ("Shark Tank Metrics", "Investor scorecard"),
        ("Forecast",           "ML Revenue Prediction"),
        ("Risk Simulation",    "Monte Carlo"),
        ("Scenario Comparison","Saved scenarios"),
        ("Benchmark",          "Industry comparison"),
        ("AI Insights",        "Gemini analysis"),
    ]
    for pg, _ in pages:
        is_active = st.session_state.page == pg
        style = (f"background:{C['accent']}14;color:{C['accent']};font-weight:700;"
                 f"border-left:3px solid {C['accent']};border-radius:0 8px 8px 0;") if is_active else ""
        if st.button(pg, key=f"nav_{pg}", use_container_width=True):
            st.session_state.page = pg
            st.rerun()

    st.markdown(f'<div style="font-size:9px;font-weight:700;letter-spacing:2px;color:{C["dim"]};margin:16px 0 8px 4px;">CONFIG</div>',
                unsafe_allow_html=True)
    if st.button("Settings", key="nav_Settings", use_container_width=True):
        st.session_state.page = "Settings"
        st.rerun()

    # ── Gemini key input (hidden in sidebar, not a main page) ──
    st.markdown(f'<div style="height:1px;background:{C["border"]};margin:16px 0 12px;"></div>',
                unsafe_allow_html=True)
    with st.expander("Gemini AI Key", expanded=False):
        key_val = st.text_input("API Key", type="password",
            value=st.session_state.gemini_key,
            placeholder="AIzaSy...",
            label_visibility="collapsed",
            help="Get a free key at aistudio.google.com/app/apikey")
        if key_val != st.session_state.gemini_key:
            st.session_state.gemini_key = key_val
            st.session_state.llm_ok = None
        if st.session_state.llm_ok is True:
            st.markdown('<span class="pill pill-g">AI Connected</span>', unsafe_allow_html=True)
        elif st.session_state.llm_ok is False:
            st.markdown('<span class="pill pill-r">AI Error — check key</span>', unsafe_allow_html=True)
        elif st.session_state.gemini_key:
            st.markdown('<span class="pill pill-a">Key loaded</span>', unsafe_allow_html=True)
        else:
            st.caption("No key — AI features disabled")

    if st.session_state.ready:
        st.markdown(f'<div style="height:1px;background:{C["border"]};margin:12px 0 8px;"></div>',
                    unsafe_allow_html=True)
        if st.button("Clear Data", key="sb_clear", use_container_width=True):
            st.session_state.data    = None
            st.session_state.ready   = False
            st.session_state.page    = "Home"
            st.rerun()

    st.markdown(f'<div style="font-size:9.5px;color:{C["dim"]};padding:16px 4px 0;">VentureScope v2.0</div>',
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: HOME  (landing — default when no data)
# ═══════════════════════════════════════════════════════════
if st.session_state.page == "Home":
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">
            Financial intelligence<br>for your <span>growing business</span>
        </div>
        <div class="hero-sub">
            Enter your monthly revenue and expense data. VentureScope predicts your future revenue,
            simulates risk scenarios, benchmarks you against your industry, and gives you AI-driven
            insights — the kind of analysis that gets businesses funded.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature grid
    features = [
        (C["blue"],   "Revenue Forecast",     "ML models predict your next 6–12 months using Linear Regression, Ridge, Gradient Boosting and Random Forest."),
        (C["accent"], "Shark Tank Metrics",   "ARR, EBITDA, operating leverage, break-even point, and an investor readiness scorecard — exactly what judges ask."),
        (C["green"],  "Risk Simulation",      "Monte Carlo analysis runs 10,000+ scenarios and gives you a probability distribution of future profits."),
        (C["purple"], "AI Insights",          "Gemini analyses your numbers and writes a health report, answers your questions, and builds a goal action plan."),
        (C["teal"],   "Industry Benchmark",   "Compare every margin and cost ratio against your sector average with a gap analysis and radar chart."),
        (C["orange"], "Scenario Comparison",  "Save named scenarios (High Rent, Revenue Drop, etc.) and compare Bear/Base/Bull cases side by side."),
    ]
    cols = st.columns(3)
    for i, (clr, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon" style="background:{clr}1A;">
                    <div style="width:10px;height:10px;border-radius:50%;background:{clr};
                                box-shadow:0 0 8px {clr}88;"></div>
                </div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        if (i + 1) % 3 == 0 and i < len(features) - 1:
            st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

    # Workflow diagram
    st.markdown('<div style="margin-top:32px;"></div>', unsafe_allow_html=True)
    sec("How It Works")
    steps = [
        ("1", C["blue"],   "Input Data",       "Enter monthly revenue, rent, salaries, COGS, marketing, utilities"),
        ("2", C["accent"], "Data Processing",  "Automatic metric calculation — Gross Profit, EBITDA, Margins, Burn Rate"),
        ("3", C["green"],  "ML Analysis",      "4 regression models train on your data and forecast future revenue"),
        ("4", C["purple"], "Risk Simulation",  "Monte Carlo generates 10,000 profit scenarios with probability bands"),
        ("5", C["teal"],   "AI Interpretation","Gemini reads your numbers and writes plain-English insights + action plan"),
    ]
    step_cols = st.columns(5)
    for i, (num, clr, title, desc) in enumerate(steps):
        with step_cols[i]:
            st.markdown(f"""
            <div style="text-align:center;padding:16px 10px;">
                <div style="width:44px;height:44px;border-radius:50%;background:{clr}22;
                            border:2px solid {clr}55;display:flex;align-items:center;
                            justify-content:center;margin:0 auto 12px;
                            font-family:'JetBrains Mono';font-size:16px;font-weight:700;
                            color:{clr};">{num}</div>
                <div style="font-size:12.5px;font-weight:700;color:{C['text']};margin-bottom:6px;">{title}</div>
                <div style="font-size:11px;color:{C['muted']};line-height:1.55;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if i < 4:
                pass  # arrow handled by layout

    # CTA
    st.markdown("<div style='margin:28px 0 10px;text-align:center;'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,1,2])
    with c2:
        if st.button("Load Sample Data & Explore", use_container_width=True, key="hero_cta"):
            st.session_state.data     = calc_metrics(sample_data())
            st.session_state.ready    = True
            st.session_state.biz_name = "Sample Retail Co."
            st.session_state.biz_type = "Retail"
            st.session_state.page     = "Overview"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "Overview":
    ph("Overview", "Key performance indicators and financial health summary")
    if need_data(): st.stop()

    df   = st.session_state.data
    b    = BENCHMARKS[st.session_state.biz_type]
    n    = len(df)

    # Row 1 KPIs
    r1 = st.columns(5)
    kpi_card(r1[0],"Total Revenue",      fmt(df["Revenue"].sum()),     delta=mom_delta(df,"Revenue"),      color=C["ch1"])
    kpi_card(r1[1],"Gross Profit",       fmt(df["Gross_Profit"].sum()),delta=mom_delta(df,"Gross_Profit"), color=C["ch3"])
    kpi_card(r1[2],"EBITDA",             fmt(df["EBITDA"].sum()),      delta=mom_delta(df,"EBITDA"),       color=C["ch5"])
    kpi_card(r1[3],"Net Profit",         fmt(df["Net_Profit"].sum()),  delta=mom_delta(df,"Net_Profit"),
             color=C["green"] if df["Net_Profit"].sum()>=0 else C["red"])
    kpi_card(r1[4],"Profitable Months",  f"{(df['Net_Profit']>0).sum()} / {n}",color=C["accent"])

    # Row 2 KPIs
    r2 = st.columns(5)
    avg_gm  = df["Gross_Margin"].mean()
    avg_eb  = df["EBITDA_Margin"].mean()
    avg_nm  = df["Net_Margin"].mean()
    burns   = df[df["Net_Profit"]<0]["Burn_Rate"]
    avg_brn = burns.mean() if len(burns) else 0
    kpi_card(r2[0],"Gross Margin",  pct(avg_gm),  color=C["ch2"], sub=f"Industry: {b['gross']}%")
    kpi_card(r2[1],"EBITDA Margin", pct(avg_eb),  color=C["ch6"], sub=f"Industry: {b['ebitda']}%")
    kpi_card(r2[2],"Net Margin",    pct(avg_nm),
             color=C["green"] if avg_nm>=b["net"] else C["red"], sub=f"Industry: {b['net']}%")
    kpi_card(r2[3],"Avg Burn Rate", fmt(avg_brn) if avg_brn>0 else "None",
             color=C["red"] if avg_brn>0 else C["green"])
    kpi_card(r2[4],"OpEx Ratio",    pct(df["OpEx_Pct"].mean()), color=C["ch7"])

    sec("Revenue & Profitability")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Month"],y=df["Revenue"],name="Revenue",
            mode="lines+markers",line=dict(color=C["ch1"],width=2.5),marker=dict(size=6),
            fill="tozeroy",fillcolor=C["ch1"]+"12"))
        fig.add_trace(go.Scatter(x=df["Month"],y=df["Total_Expenses"],name="Expenses",
            mode="lines+markers",line=dict(color=C["ch4"],width=2.5),marker=dict(size=6),
            fill="tozeroy",fillcolor=C["ch4"]+"12"))
        cb(fig,"Revenue vs Total Expenses")
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        clrs = [C["green"] if p>=0 else C["red"] for p in df["Net_Profit"]]
        fig  = go.Figure(go.Bar(x=df["Month"],y=df["Net_Profit"],marker_color=clrs,
            text=[fmt(p) for p in df["Net_Profit"]],
            textposition="outside",textfont=dict(size=9,color=C["muted"])))
        fig.add_hline(y=0,line_dash="dash",line_color=C["border2"],line_width=1)
        cb(fig,"Monthly Net Profit / Loss")
        st.plotly_chart(fig,use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        for col,name,clr in [("Gross_Margin","Gross Margin",C["ch3"]),
                              ("EBITDA_Margin","EBITDA Margin",C["ch5"]),
                              ("Net_Margin","Net Margin",C["ch2"])]:
            fig.add_trace(go.Scatter(x=df["Month"],y=df[col],name=name,
                mode="lines+markers",line=dict(color=clr,width=2.2),marker=dict(size=5)))
        fig.add_hline(y=b["net"],line_dash="dot",line_color=C["muted"],
            annotation_text="Net Margin Benchmark",annotation_position="top right",
            annotation_font=dict(color=C["muted"],size=10))
        cb(fig,"Margin Trends (%)"); fig.update_layout(yaxis_title="%")
        st.plotly_chart(fig,use_container_width=True)

    with c4:
        lm    = df.iloc[-1]
        labs  = ["COGS","Salaries","Rent","Marketing","Utilities","Misc"]
        vals  = [lm["COGS"],lm["Salaries"],lm["Rent"],lm["Marketing"],lm["Utilities"],lm["Misc"]]
        fig   = go.Figure(go.Pie(labels=labs,values=vals,hole=0.55,
            marker=dict(colors=PAL),textinfo="label+percent",
            insidetextorientation="radial",textfont=dict(size=11,color=C["text"])))
        fig.update_layout(height=360,paper_bgcolor=C["card"],showlegend=False,
            title=dict(text=f"Cost Mix — {lm['Month']}",
                font=dict(size=12.5,color=C["text"],family="Plus Jakarta Sans"),x=0.01),
            margin=dict(l=10,r=10,t=46,b=10))
        st.plotly_chart(fig,use_container_width=True)

    sec("Cash Flow Bridge — Latest Month")
    lm  = df.iloc[-1]
    wfx = ["Revenue","COGS","Salaries","Rent","Marketing","Utilities","Misc","Net Profit"]
    wfy = [lm["Revenue"],-lm["COGS"],-lm["Salaries"],-lm["Rent"],
           -lm["Marketing"],-lm["Utilities"],-lm["Misc"],lm["Net_Profit"]]
    fig = go.Figure(go.Waterfall(
        orientation="v",measure=["absolute"]+["relative"]*6+["total"],
        x=wfx,y=wfy,
        text=[fmt(abs(v)) for v in wfy],textposition="outside",
        textfont=dict(size=9.5,color=C["muted"]),
        connector={"line":{"color":C["border"]}},
        increasing={"marker":{"color":C["green"]}},
        decreasing={"marker":{"color":C["ch4"]}},
        totals={"marker":{"color":C["accent"]}}))
    cb(fig,f"Cash Flow Bridge — {lm['Month']}",h=330)
    st.plotly_chart(fig,use_container_width=True)

    # AI auto-summary
    if st.session_state.gemini_key:
        sec("AI Quick Read")
        if st.button("Generate Auto-Summary", key="ov_ai"):
            with st.spinner("Analysing your numbers…"):
                ctx = biz_context()
                out = call_llm(
                    f"Given this business data:\n{ctx}\n\n"
                    "Write a 3-sentence executive summary: (1) Overall health assessment with score /10. "
                    "(2) The single biggest strength with a specific number. "
                    "(3) The single biggest risk and what to do about it. Be direct, use ₹."
                )
            st.markdown(f'<div class="ai-box"><div class="ai-header"><div class="ai-dot"></div>'
                        f'<strong style="color:{C["blue"]}">AI Summary</strong></div>{out}</div>',
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: INCOME STATEMENT
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "Income Statement":
    ph("Income Statement", "Full P&L — from Revenue to Profit After Tax")
    if need_data(): st.stop()

    df   = st.session_state.data
    last = df.iloc[-1]
    rev  = last["Revenue"]

    # Compute IS line items
    cogs  = last["COGS"]
    gp    = last["Gross_Profit"]
    opex  = last["Rent"]+last["Salaries"]+last["Utilities"]+last["Marketing"]+last["Misc"]
    ebitda= last["EBITDA"]
    depr  = last["Rent"]*0.05
    ebit  = ebitda - depr
    pbt   = max(ebit, 0)
    tax   = pbt * 0.25
    pat   = pbt - tax

    # Render IS table
    sec("Income Statement — Latest Month")
    rows = [
        ("Revenue",                     rev,    True,  False),
        ("Cost of Goods Sold (COGS)",   -cogs,  False, False),
        ("Gross Profit",                gp,     False, True),
        ("Operating Expenses",          -opex,  False, False),
        ("EBITDA",                      ebitda, False, True),
        ("Depreciation (est. 5% rent)", -depr,  False, False),
        ("EBIT (Operating Profit)",     ebit,   False, True),
        ("Interest Expense",            0,      False, False),
        ("Profit Before Tax (PBT)",     pbt,    False, True),
        ("Tax — 25% (India est.)",      -tax,   False, False),
        ("Profit After Tax (PAT)",      pat,    False, True),
    ]
    for item, val, is_rev, is_sub in rows:
        clr = (C["accent"] if is_rev else
               (C["green"] if is_sub and val>=0 else
                C["red"]   if is_sub and val<0 else C["text"]))
        wt  = "700" if (is_sub or is_rev) else "400"
        bg  = f"background:{C['border']}22;" if is_sub else ""
        pct_html = (f'<span style="font-size:11.5px;color:{C["muted"]};margin-left:8px;">'
                    f'({val/rev*100:.1f}%)</span>') if rev and not is_rev and val!=0 else ""
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:10px 18px;border-bottom:1px solid {C['border']};
                    {bg}border-radius:6px;margin-bottom:2px;">
            <span style="color:{clr};font-weight:{wt};font-size:14px;">{item}</span>
            <span style="color:{clr};font-weight:{wt};font-family:'JetBrains Mono';font-size:14px;">
                {fmt(val) if val!=0 else "—"}{pct_html}
            </span>
        </div>""", unsafe_allow_html=True)

    # Monthly trend charts
    sec("Monthly Trend")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Month"],y=df["Gross_Profit"],name="Gross Profit",marker_color=C["ch3"]))
        fig.add_trace(go.Bar(x=df["Month"],y=df["EBITDA"],       name="EBITDA",      marker_color=C["ch5"]))
        fig.add_trace(go.Bar(x=df["Month"],y=df["Net_Profit"],   name="Net Profit",
            marker_color=[C["green"] if v>=0 else C["red"] for v in df["Net_Profit"]]))
        cb(fig,"Gross Profit / EBITDA / Net Profit",h=340)
        fig.update_layout(barmode="group")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig = go.Figure()
        for col,name,clr in [("Gross_Margin","Gross Margin",C["ch3"]),
                              ("EBITDA_Margin","EBITDA",C["ch5"]),("Net_Margin","Net Margin",C["ch2"])]:
            fig.add_trace(go.Scatter(x=df["Month"],y=df[col],name=name,
                mode="lines+markers",line=dict(color=clr,width=2.2),marker=dict(size=5)))
        cb(fig,"Margin Trends (%)",h=340); fig.update_layout(yaxis_title="%")
        st.plotly_chart(fig,use_container_width=True)

    # AI IS analysis
    if st.session_state.gemini_key:
        sec("AI Analysis")
        if st.button("Analyse Income Statement with AI", key="is_ai"):
            with st.spinner("Reading your P&L…"):
                out = call_llm(
                    f"{biz_context()}\n\n"
                    f"Latest month P&L: Revenue {fmt(rev)}, COGS {fmt(cogs)}, "
                    f"Gross Profit {fmt(gp)} ({gp/rev*100:.1f}%), "
                    f"EBITDA {fmt(ebitda)} ({ebitda/rev*100:.1f}%), "
                    f"PAT {fmt(pat)} ({pat/rev*100:.1f}% if >0).\n\n"
                    "Give: (1) Is gross margin healthy for the sector? (2) Where is the biggest cost leak? "
                    "(3) One specific way to improve PAT by at least 5%. Be concise, use ₹."
                )
            st.markdown(f'<div class="ai-box"><div class="ai-header"><div class="ai-dot"></div>'
                        f'<strong style="color:{C["blue"]}">AI P&L Analysis</strong></div>{out}</div>',
                        unsafe_allow_html=True)

    # Download
    sec("Export")
    csv = df[["Month","Revenue","COGS","Gross_Profit","Gross_Margin","EBITDA",
              "EBITDA_Margin","Net_Profit","Net_Margin","Total_Expenses"]].to_csv(index=False).encode()
    st.download_button("Download Income Statement CSV", csv,
                       f"{st.session_state.biz_name or 'business'}_IS.csv","text/csv")

# ═══════════════════════════════════════════════════════════
#  PAGE: SHARK TANK METRICS
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "Shark Tank Metrics":
    ph("Shark Tank Metrics",
       "The metrics investors actually look at before writing a cheque")
    if need_data(): st.stop()

    df   = st.session_state.data
    b    = BENCHMARKS[st.session_state.biz_type]
    n    = len(df)
    avg_rev = df["Revenue"].mean()

    st.markdown(f"""<div class="alert alert-info">
        These are the exact metrics asked on Shark Tank India, Y Combinator applications,
        and seed fund due diligence. Each metric below tells a different part of your business story.
    </div>""", unsafe_allow_html=True)

    # Core calculations
    ann_rev = avg_rev * 12
    ann_pro = df["Net_Profit"].mean() * 12

    # Revenue CAGR
    cagr = ((df["Revenue"].iloc[-1]/df["Revenue"].iloc[0])**(12/n)-1)*100 if n>=2 else 0

    # Operating leverage
    if n>1 and df["Revenue"].iloc[-2]!=0 and df["EBITDA"].iloc[-2]!=0:
        rc = (df["Revenue"].iloc[-1]-df["Revenue"].iloc[-2])/abs(df["Revenue"].iloc[-2])*100
        ec = (df["EBITDA"].iloc[-1]-df["EBITDA"].iloc[-2])/abs(df["EBITDA"].iloc[-2])*100
        op_lev = round(ec/rc,2) if rc!=0 else 0
    else:
        op_lev = 0

    # Break-even
    fc   = df["Rent"].mean()+df["Utilities"].mean()+df["Salaries"].mean()*0.7
    vc_r = max((df["COGS"].mean()+df["Marketing"].mean()+df["Misc"].mean()+df["Salaries"].mean()*0.3)/avg_rev,0)
    be   = fc/(1-vc_r) if vc_r<1 else fc*2
    safety = (avg_rev-be)/avg_rev*100

    # Burn / runway
    burns   = df[df["Net_Profit"]<0]["Burn_Rate"]
    avg_brn = burns.mean() if len(burns) else 0
    runway  = round(avg_rev*3/avg_brn) if avg_brn>0 else 999

    # Rev per employee
    est_emp = max(round(df["Salaries"].mean()/35_000),1)
    rev_emp = avg_rev / est_emp

    r1 = st.columns(4)
    kpi_card(r1[0],"ARR (Run-Rate)",     fmt(ann_rev),        color=C["ch1"],sub="Monthly avg × 12")
    kpi_card(r1[1],"Monthly EBITDA",     fmt(df["EBITDA"].mean()),color=C["ch5"])
    kpi_card(r1[2],"Revenue CAGR",       f"{cagr:.1f}%/yr",   color=C["ch3"])
    kpi_card(r1[3],"Operating Leverage", f"{op_lev:.2f}x",    color=C["ch2"],
             sub="EBITDA sensitivity to revenue change")

    r2 = st.columns(4)
    kpi_card(r2[0],"Break-Even",         fmt(be),             color=C["ch7"])
    kpi_card(r2[1],"Margin of Safety",   pct(safety),
             color=C["green"] if safety>0 else C["red"])
    kpi_card(r2[2],"Cash Runway",        f"{runway} mo." if runway<999 else "Healthy",
             color=C["green"] if runway>6 else C["red"])
    kpi_card(r2[3],"Rev / Employee",     fmt(rev_emp),        color=C["ch6"],
             sub=f"~{est_emp} employees est.")

    sec("Break-Even Analysis")
    c1, c2 = st.columns(2)
    with c1:
        rng  = np.linspace(0, avg_rev*1.6, 200)
        cost = fc + vc_r*rng
        fig  = go.Figure()
        fig.add_trace(go.Scatter(x=rng,y=rng,  name="Revenue",
            line=dict(color=C["ch1"],width=2.5)))
        fig.add_trace(go.Scatter(x=rng,y=cost, name="Total Cost",
            line=dict(color=C["ch4"],width=2.5)))
        fig.add_vline(x=be,line_dash="dash",line_color=C["muted"],line_width=1,
            annotation_text=f"BE {fmt(be)}",annotation_position="top left",
            annotation_font=dict(color=C["muted"],size=10))
        fig.add_trace(go.Scatter(x=[be],y=[be],mode="markers",name="Break-Even",
            marker=dict(size=14,color=C["accent"],symbol="star",
                        line=dict(color=C["bg"],width=1.5))))
        cb(fig,"Break-Even Chart")
        fig.update_layout(xaxis_title="Revenue (₹)",yaxis_title="Amount (₹)")
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        fixed = df["Rent"].mean()+df["Salaries"].mean()+df["Utilities"].mean()
        var   = df["COGS"].mean()+df["Marketing"].mean()+df["Misc"].mean()
        fv    = fixed/(fixed+var)*100
        fig   = go.Figure(go.Pie(
            labels=["Fixed","Variable"],values=[fixed,var],hole=0.58,
            marker=dict(colors=[C["ch2"],C["ch1"]]),
            textinfo="label+percent",textfont=dict(size=12,color=C["text"])))
        fig.update_layout(height=360,paper_bgcolor=C["card"],showlegend=False,
            title=dict(text="Fixed vs Variable Costs",
                font=dict(size=12.5,color=C["text"],family="Plus Jakarta Sans"),x=0.01),
            margin=dict(l=10,r=10,t=46,b=10))
        fig.add_annotation(text=f"Fixed<br>{fv:.0f}%",x=0.5,y=0.5,showarrow=False,
            font=dict(size=15,color=C["text"],family="JetBrains Mono"))
        st.plotly_chart(fig,use_container_width=True)

    # Investor scorecard
    sec("Investor Readiness Scorecard")
    checks = [
        ("Revenue Growth",    n>1 and df["Revenue"].iloc[-1]>df["Revenue"].iloc[0],
                              f"{'Trending up' if n>1 and df['Revenue'].iloc[-1]>df['Revenue'].iloc[0] else 'Flat/declining'}"),
        ("Gross Margin",      df["Gross_Margin"].mean()>=b["gross"]*0.85,
                              f"{df['Gross_Margin'].mean():.1f}% vs {b['gross']}% benchmark"),
        ("EBITDA Positive",   (df["EBITDA"]>0).sum()>=n*0.75,
                              f"{(df['EBITDA']>0).sum()}/{n} months positive"),
        ("Net Profitable",    (df["Net_Profit"]>0).sum()>=n*0.5,
                              f"{(df['Net_Profit']>0).sum()}/{n} months profitable"),
        ("Above Break-Even",  avg_rev>=be,
                              f"Avg revenue {fmt(avg_rev)} vs BE {fmt(be)}"),
        ("Cash Runway",       avg_brn==0 or runway>6,
                              "No burn" if avg_brn==0 else f"{runway} months"),
        ("OpEx Discipline",   df["OpEx_Pct"].mean()<95,
                              f"{df['OpEx_Pct'].mean():.0f}% expenses-to-revenue"),
        ("Margin of Safety",  safety>15,
                              f"{safety:.1f}% above break-even"),
    ]
    passed = sum(1 for _,p,_ in checks if p)
    sc_clr = C["green"] if passed>=6 else (C["accent"] if passed>=4 else C["red"])
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
        <div style="font-size:42px;font-weight:800;font-family:'JetBrains Mono';
                    color:{sc_clr};">{passed}/8</div>
        <div>
            <div style="font-size:15px;font-weight:700;color:{C['text']};">
                {'Investor Ready' if passed>=6 else 'Growing Business' if passed>=4 else 'Needs Attention'}
            </div>
            <div style="font-size:12px;color:{C['muted']}">{passed} of 8 criteria met</div>
        </div>
        <div style="flex:1;background:{C['border']};border-radius:6px;height:8px;margin-left:8px;">
            <div style="width:{passed/8*100:.0f}%;height:100%;background:{sc_clr};
                        border-radius:6px;"></div>
        </div>
    </div>""", unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    for i,(name,ok,detail) in enumerate(checks):
        col = sc1 if i%2==0 else sc2
        cls = "panel panel-green" if ok else "panel panel-gold"
        clr = C["green"] if ok else C["accent"]
        with col:
            st.markdown(f"""
            <div class="{cls}" style="display:flex;justify-content:space-between;
                         align-items:center;padding:12px 16px;margin-bottom:6px;">
                <div>
                    <strong style="color:{C['text']};font-size:13px;">{name}</strong>
                    <div style="font-size:11px;color:{C['muted']};margin-top:2px;">{detail}</div>
                </div>
                <span class="pill {'pill-g' if ok else 'pill-a'}">{'PASS' if ok else 'REVIEW'}</span>
            </div>""", unsafe_allow_html=True)

    # AI Shark Tank readiness
    if st.session_state.gemini_key:
        sec("AI Investor Assessment")
        if st.button("Get AI Investor Feedback", key="st_ai"):
            with st.spinner("Thinking like an investor…"):
                out = call_llm(
                    f"{biz_context()}\n\n"
                    f"Investor scorecard: {passed}/8 criteria passed.\n"
                    f"ARR: {fmt(ann_rev)}, EBITDA margin: {df['EBITDA_Margin'].mean():.1f}%, "
                    f"Break-even margin of safety: {safety:.1f}%\n\n"
                    "You are a Shark Tank India investor. Give: "
                    "(1) Would you invest? One sentence verdict with the main reason. "
                    "(2) The one number you'd want improved before investing. "
                    "(3) The valuation multiple this business could reasonably claim and why. "
                    "Be direct, no sugarcoating, use ₹."
                )
            st.markdown(f'<div class="ai-box"><div class="ai-header"><div class="ai-dot"></div>'
                        f'<strong style="color:{C["blue"]}">Investor Perspective</strong></div>{out}</div>',
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: FORECAST
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "Forecast":
    ph("Revenue Forecast & Prediction",
       "ML-powered forecasting with 4 models, confidence bands, and model comparison")
    if need_data(): st.stop()

    df = st.session_state.data
    n  = len(df)

    c1, c2 = st.columns([1,3])
    with c1:
        fwd   = st.slider("Forecast months", 1, 12, 6)
        model_name = st.selectbox("Model", [
            "Linear Regression","Ridge Regression",
            "Gradient Boosting","Random Forest"])
    with c2:
        st.markdown(f"""<div class="alert alert-info">
            <strong>How it works:</strong> Your monthly revenue history is the training data.
            The model learns the trend and projects it forward. The ±15% band reflects real-world
            uncertainty — use the lower band for conservative planning.
        </div>""", unsafe_allow_html=True)

    # Train all 4 models
    X  = np.arange(n).reshape(-1,1)
    Xf = np.arange(n,n+fwd).reshape(-1,1)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xfs= sc.transform(Xf)

    flabels = [MONTHS_S[(n+i)%12]+" (F)" for i in range(fwd)]

    def fit(name, Xs, y):
        m = {"Linear Regression": LinearRegression(),
             "Ridge Regression":  Ridge(alpha=1.0),
             "Gradient Boosting": GradientBoostingRegressor(n_estimators=80,random_state=42),
             "Random Forest":     RandomForestRegressor(n_estimators=80,random_state=42)}[name]
        m.fit(Xs,y); return m

    rm = fit(model_name,Xs,df["Revenue"].values)
    em = fit(model_name,Xs,df["Total_Expenses"].values)
    rf = rm.predict(Xfs); ef = em.predict(Xfs); pf = rf-ef
    rhi= rf*1.15; rlo= rf*0.85

    tr  = rm.predict(Xs)
    r2_ = r2_score(df["Revenue"].values,tr)
    mae_= mean_absolute_error(df["Revenue"].values,tr)

    st.markdown(f"""
    <div style="display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap;">
        <span class="pill pill-b">{model_name}</span>
        <span class="pill pill-g">R² = {r2_:.3f}</span>
        <span class="pill pill-a">MAE = {fmt(mae_)}</span>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Month"].tolist(),y=df["Revenue"].tolist(),
            name="Actual",mode="lines+markers",
            line=dict(color=C["ch1"],width=2.5),marker=dict(size=6)))
        fig.add_trace(go.Scatter(
            x=flabels+flabels[::-1],y=rhi.tolist()+rlo.tolist()[::-1],
            fill="toself",fillcolor=C["ch5"]+"18",
            line=dict(color="rgba(0,0,0,0)"),name="±15% Band"))
        fig.add_trace(go.Scatter(x=flabels,y=rf.tolist(),name="Forecast",
            mode="lines+markers",line=dict(color=C["ch5"],width=2.5,dash="dash"),
            marker=dict(size=8,symbol="diamond")))
        cb(fig,"Revenue Forecast with Confidence Band")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Month"].tolist(),y=df["Net_Profit"].tolist(),
            name="Actual",marker_color=[C["green"] if p>=0 else C["red"] for p in df["Net_Profit"]]))
        fig.add_trace(go.Bar(x=flabels,y=pf.tolist(),name="Forecast",
            marker_color=[C["ch3"]+"99" if p>=0 else C["ch4"]+"99" for p in pf],
            marker_line=dict(color=C["border2"],width=1)))
        fig.add_hline(y=0,line_dash="dash",line_color=C["border2"])
        cb(fig,"Profit Forecast")
        st.plotly_chart(fig,use_container_width=True)

    # Forecast table
    with st.expander("Forecast Detail Table"):
        fc_df = pd.DataFrame({
            "Month":flabels,
            "Revenue (₹)":rf.round(0),
            "Expenses (₹)":ef.round(0),
            "Net Profit (₹)":pf.round(0),
            "Net Margin %":np.where(rf>0,pf/rf*100,0).round(1)
        })
        st.dataframe(fc_df.style.format({
            "Revenue (₹)":"₹{:,.0f}","Expenses (₹)":"₹{:,.0f}",
            "Net Profit (₹)":"₹{:,.0f}","Net Margin %":"{:.1f}%"
        }).background_gradient(subset=["Net Profit (₹)"],cmap="RdYlGn"),
        use_container_width=True)

    # Model comparison
    sec("Model Performance Comparison")
    comp = []
    for mn in ["Linear Regression","Ridge Regression","Gradient Boosting","Random Forest"]:
        mm = fit(mn,Xs,df["Revenue"].values)
        pp = mm.predict(Xs)
        comp.append({"Model":mn,
                     "R²":round(r2_score(df["Revenue"].values,pp),4),
                     "MAE (₹)":int(mean_absolute_error(df["Revenue"].values,pp))})
    comp_df = pd.DataFrame(comp).sort_values("R²",ascending=False)
    fig = go.Figure()
    clrs_c = [C["accent"] if r["Model"]==model_name else C["ch1"] for _,r in comp_df.iterrows()]
    fig.add_trace(go.Bar(x=comp_df["Model"],y=comp_df["R²"],marker_color=clrs_c,
        text=[f"{v:.4f}" for v in comp_df["R²"]],
        textposition="outside",textfont=dict(size=11,color=C["muted"])))
    cb(fig,"R² Score by Model — Higher is Better",h=300)
    fig.update_layout(yaxis=dict(range=[0,1.12],title="R² Score"),showlegend=False)
    st.plotly_chart(fig,use_container_width=True)

    # AI forecast interpretation
    if st.session_state.gemini_key:
        sec("AI Forecast Interpretation")
        if st.button("Interpret Forecast with AI", key="fc_ai"):
            forecast_sum = f"Next {fwd} months: avg forecast revenue {fmt(rf.mean())}/mo, avg profit {fmt(pf.mean())}/mo"
            with st.spinner("Interpreting forecast…"):
                out = call_llm(
                    f"{biz_context()}\n\n"
                    f"Forecast ({model_name}, R²={r2_:.3f}): {forecast_sum}\n\n"
                    "Give: (1) Is this forecast realistic given the trend? "
                    "(2) What assumptions could make it wrong? "
                    "(3) One action to improve the forecast outcome. Use ₹, be specific."
                )
            st.markdown(f'<div class="ai-box"><div class="ai-header"><div class="ai-dot"></div>'
                        f'<strong style="color:{C["blue"]}">AI Forecast Analysis</strong></div>{out}</div>',
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: RISK SIMULATION
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "Risk Simulation":
    ph("Risk Simulation", "Monte Carlo analysis — probability distribution of future profits")
    if need_data(): st.stop()

    df = st.session_state.data

    st.markdown(f"""<div class="alert alert-info">
        Monte Carlo runs thousands of scenarios by randomly sampling your revenue and cost
        distributions. The result is a probability curve — far more honest than a single number.
        <strong>A profit probability above 50% means more scenarios are profitable than not.</strong>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1: rv    = st.slider("Revenue volatility %", 5,40,15)
    with c2: ev    = st.slider("Expense volatility %", 5,30,10)
    with c3: nsims = st.select_slider("Simulations",[1_000,5_000,10_000,50_000],10_000)

    st.markdown("**What-if stress tests**")
    w1,w2,w3,w4 = st.columns(4)
    with w1: ri = st.number_input("Rent increase %",   0,100,0,5)
    with w2: rd = st.number_input("Revenue drop %",    0,80, 0,5)
    with w3: si = st.number_input("Salary hike %",     0,50, 0,5)
    with w4: mi = st.number_input("Marketing hike %",  0,100,0,10)

    if st.button("Run Monte Carlo Simulation", use_container_width=True):
        with st.spinner(f"Running {nsims:,} scenarios…"):
            _ar = df["Revenue"].mean() * (1 - rd/100)
            _ae = (df["Total_Expenses"].mean()
                   + df["Rent"].mean()     * (ri/100)
                   + df["Salaries"].mean() * (si/100)
                   + df["Marketing"].mean()* (mi/100))
            _r  = np.maximum(np.random.normal(_ar, _ar*rv/100, nsims), 0)
            _e  = np.maximum(np.random.normal(_ae, _ae*ev/100, nsims), 0)
            st.session_state.mc_results = {
                "profits": _r - _e,
                "params": dict(rv=rv,ev=ev,ri=ri,rd=rd,si=si,mi=mi,nsims=nsims)
            }

    if st.session_state.mc_results:
        profits = st.session_state.mc_results["profits"]
        pp  = (profits>0).mean()*100
        ev_ = profits.mean()
        p5,p25,p50,p75,p95 = np.percentile(profits,[5,25,50,75,95])

        mc1,mc2,mc3,mc4,mc5 = st.columns(5)
        kpi_card(mc1,"Profit Probability", f"{pp:.1f}%",    color=C["green"] if pp>50 else C["red"])
        kpi_card(mc2,"Expected Profit",    fmt(ev_),         color=C["ch1"])
        kpi_card(mc3,"Best Case (P95)",    fmt(p95),         color=C["ch3"])
        kpi_card(mc4,"Worst Case (P5)",    fmt(p5),          color=C["ch4"])
        kpi_card(mc5,"Loss Probability",   f"{100-pp:.1f}%", color=C["red"] if (100-pp)>30 else C["accent"])

        c1,c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=profits,nbinsx=80,
                marker_color=C["ch5"],marker_line_color=C["bg"],
                marker_line_width=0.4,opacity=0.9,name="Scenarios"))
            fig.add_vline(x=0,  line_dash="dash",line_color=C["red"],  line_width=1.5,
                annotation_text="Zero",annotation_font=dict(color=C["red"],size=10))
            fig.add_vline(x=ev_,line_dash="dot", line_color=C["accent"],line_width=1.5,
                annotation_text="Expected",annotation_position="top right",
                annotation_font=dict(color=C["accent"],size=10))
            cb(fig,f"Profit Distribution — {nsims:,} Scenarios",h=360)
            fig.update_layout(xaxis_title="Monthly Profit (₹)",showlegend=False)
            st.plotly_chart(fig,use_container_width=True)

        with c2:
            # Tornado chart
            base = df["Revenue"].mean() - df["Total_Expenses"].mean()
            ar,as_,ac,am,art = (df["Revenue"].mean(),df["Salaries"].mean(),
                                 df["COGS"].mean(),df["Marketing"].mean(),df["Rent"].mean())
            sens = {
                "Revenue ±10%":      (base+ar*.10,  base-ar*.10),
                "COGS ±15%":         (base+ac*.15,  base-ac*.15),
                "Salaries ±15%":     (base+as_*.15, base-as_*.15),
                "Rent ±20%":         (base+art*.20, base-art*.20),
                "Marketing ±50%":    (base+am*.50,  base-am*.50),
            }
            ss = sorted(sens.items(),key=lambda x:abs(x[1][0]-x[1][1]),reverse=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(y=[s[0] for s in ss],x=[s[1][0] for s in ss],
                orientation="h",name="Upside",marker_color=C["ch3"]))
            fig.add_trace(go.Bar(y=[s[0] for s in ss],x=[s[1][1] for s in ss],
                orientation="h",name="Downside",marker_color=C["ch4"]))
            fig.add_vline(x=0,line_color=C["border2"],line_width=1)
            fig.update_layout(barmode="relative",height=360,
                paper_bgcolor=C["card"],plot_bgcolor=C["bg"],
                font=dict(family="Plus Jakarta Sans",color=C["muted"]),
                xaxis_title="Profit Impact (₹)",margin=dict(l=10,r=10,t=46,b=16),
                title=dict(text="Tornado — Sensitivity Chart",
                    font=dict(size=12.5,color=C["text"],family="Plus Jakarta Sans"),x=0.01),
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=C["text"])))
            st.plotly_chart(fig,use_container_width=True)

        # Scenario boxes
        sec("Scenario Summary")
        s1,s2,s3 = st.columns(3)
        for col,ttl,val,desc,clr in [
            (s1,"Bear Case",  p5,  "5th percentile — plan for this", C["red"]),
            (s2,"Base Case",  ev_, "Expected (mean outcome)",        C["accent"]),
            (s3,"Bull Case",  p95, "95th percentile — best case",    C["green"]),
        ]:
            with col:
                st.markdown(f"""
                <div class="panel" style="text-align:center;border-top:3px solid {clr};">
                    <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
                                color:{C['muted']};margin-bottom:10px;">{ttl}</div>
                    <div style="font-size:28px;font-weight:700;font-family:'JetBrains Mono';
                                color:{clr};">{fmt(val)}</div>
                    <div style="font-size:11px;color:{C['muted']};margin-top:6px;">{desc}</div>
                </div>""", unsafe_allow_html=True)

        # Save scenario
        st.markdown("---")
        sname = st.text_input("Save this as a named scenario",placeholder="e.g. High Rent Stress Test")
        if st.button("Save Scenario") and sname:
            st.session_state.scenarios[sname] = {
                "bear":p5,"base":ev_,"bull":p95,"prob":pp,
                "params":st.session_state.mc_results["params"]}
            st.success(f"Saved '{sname}'. View in Scenario Comparison.")

        # AI risk analysis
        if st.session_state.gemini_key:
            sec("AI Risk Assessment")
            if st.button("Analyse Risk with AI", key="mc_ai"):
                params = st.session_state.mc_results["params"]
                with st.spinner("Assessing risk profile…"):
                    out = call_llm(
                        f"{biz_context()}\n\n"
                        f"Monte Carlo results ({params['nsims']:,} simulations):\n"
                        f"Profit probability: {pp:.1f}%\n"
                        f"Expected profit: {fmt(ev_)}/month\n"
                        f"Bear case (P5): {fmt(p5)}/month\n"
                        f"Bull case (P95): {fmt(p95)}/month\n"
                        f"Stress tests applied: rent +{params['ri']}%, revenue -{params['rd']}%, "
                        f"salaries +{params['si']}%, marketing +{params['mi']}%\n\n"
                        "Give: (1) Is this risk profile acceptable for the sector? "
                        "(2) What is the single biggest risk driver? "
                        "(3) One specific hedge strategy to narrow the Bear-Bull gap. "
                        "Use ₹, be direct."
                    )
                st.markdown(f'<div class="ai-box"><div class="ai-header"><div class="ai-dot"></div>'
                            f'<strong style="color:{C["blue"]}">AI Risk Assessment</strong></div>{out}</div>',
                            unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert alert-warn" style="text-align:center;padding:28px;">Configure parameters above and click <strong>Run Monte Carlo Simulation</strong>.</div>',
                    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: SCENARIO COMPARISON
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "Scenario Comparison":
    ph("Scenario Comparison","Side-by-side comparison of saved Monte Carlo scenarios")

    if not st.session_state.scenarios:
        st.markdown(f'<div class="alert alert-warn">No scenarios saved yet. Run a simulation in <strong>Risk Simulation</strong> and click Save.</div>',
                    unsafe_allow_html=True)
        st.stop()

    sc = st.session_state.scenarios
    names = list(sc.keys())

    rows = [{"Scenario":n,"Profit Prob %":round(sc[n]["prob"],1),
             "Bear (₹)":int(sc[n]["bear"]),"Base (₹)":int(sc[n]["base"]),
             "Bull (₹)":int(sc[n]["bull"]),"Range (₹)":int(sc[n]["bull"]-sc[n]["bear"])}
            for n in names]
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    fig = go.Figure()
    for name,clr in [("Bear","bear",C["ch4"]),("Base","base",C["accent"]),("Bull","bull",C["ch3"])]:
        fig.add_trace(go.Bar(name=f"{name} Case",x=names,
            y=[sc[n][clr] for n in names],marker_color=clr))
    cb(fig,"Bear / Base / Bull by Scenario",h=380)
    fig.update_layout(barmode="group",yaxis_title="Monthly Profit (₹)")
    st.plotly_chart(fig,use_container_width=True)

    fig2 = go.Figure(go.Bar(x=names,y=[sc[n]["prob"] for n in names],
        marker_color=[C["green"] if sc[n]["prob"]>50 else C["red"] for n in names],
        text=[f"{sc[n]['prob']:.1f}%" for n in names],
        textposition="outside",textfont=dict(size=12,color=C["muted"])))
    fig2.add_hline(y=50,line_dash="dash",line_color=C["muted"],line_width=1)
    cb(fig2,"Profit Probability by Scenario",h=300)
    fig2.update_layout(yaxis=dict(title="%",range=[0,115]))
    st.plotly_chart(fig2,use_container_width=True)

    if st.button("Clear All Scenarios"):
        st.session_state.scenarios = {}
        st.rerun()

# ═══════════════════════════════════════════════════════════
#  PAGE: BENCHMARK
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "Benchmark":
    ph("Industry Benchmark","Your metrics vs sector averages with gap analysis")
    if need_data(): st.stop()

    df    = st.session_state.data
    btyp  = st.session_state.biz_type
    b     = BENCHMARKS[btyp]
    avg_r = df["Revenue"].mean()

    um = {
        "net":    df["Net_Margin"].mean(),
        "gross":  df["Gross_Margin"].mean(),
        "ebitda": df["EBITDA_Margin"].mean(),
        "rent":   df["Rent_Pct"].mean(),
        "salary": df["Salary_Pct"].mean(),
        "mkt":    df["Mkt_Pct"].mean(),
    }
    labels = {"net":"Net Margin","gross":"Gross Margin","ebitda":"EBITDA Margin",
              "rent":"Rent Ratio","salary":"Salary Ratio","mkt":"Marketing Ratio"}

    # Badge row
    cols = st.columns(6)
    for i,(key,label) in enumerate(labels.items()):
        uv,bv = um[key],b[key]
        is_m  = key in ["net","gross","ebitda"]
        better= (uv>=bv) if is_m else (uv<=bv)
        clr   = C["green"] if better else C["red"]
        with cols[i]:
            st.markdown(f"""
            <div class="kpi" style="border-top:3px solid {clr};">
                <div class="kpi-accent" style="background:{clr};"></div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-val" style="font-size:20px;">{uv:.1f}%</div>
                <div style="font-size:10px;color:{clr};margin-top:4px;font-weight:600;">
                    {'Above' if better else 'Below'} benchmark
                </div>
                <div class="kpi-sub">Industry: {bv:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    # Charts
    cats  = [labels[k] for k in labels]
    u_vals= [um[k] for k in labels]
    b_vals= [b[k]  for k in labels]

    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=u_vals+[u_vals[0]],theta=cats+[cats[0]],
            fill="toself",name="Your Business",
            line=dict(color=C["ch1"],width=2),fillcolor=C["ch1"]+"1A"))
        fig.add_trace(go.Scatterpolar(r=b_vals+[b_vals[0]],theta=cats+[cats[0]],
            fill="toself",name="Industry Avg",
            line=dict(color=C["ch4"],width=2,dash="dash"),fillcolor=C["ch4"]+"0D"))
        fig.update_layout(height=420,paper_bgcolor=C["card"],
            polar=dict(bgcolor=C["bg"],
                radialaxis=dict(visible=True,range=[0,max(max(u_vals),max(b_vals))*1.4],
                    gridcolor=C["border"],tickfont=dict(color=C["muted"])),
                angularaxis=dict(gridcolor=C["border"],tickfont=dict(color=C["text"],size=11))),
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=C["text"])),
            font=dict(family="Plus Jakarta Sans"),
            title=dict(text="Radar — You vs Industry Benchmark",
                font=dict(size=12.5,color=C["text"],family="Plus Jakarta Sans"),x=0.01),
            margin=dict(l=30,r=30,t=52,b=30))
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Your Business",x=cats,y=u_vals,marker_color=C["ch1"],
            text=[f"{v:.1f}%" for v in u_vals],textposition="outside",
            textfont=dict(color=C["muted"],size=10)))
        fig.add_trace(go.Bar(name="Industry Avg", x=cats,y=b_vals,marker_color=C["ch4"]+"AA",
            text=[f"{v:.1f}%" for v in b_vals],textposition="outside",
            textfont=dict(color=C["muted"],size=10)))
        cb(fig,"Metric-by-Metric Comparison",h=420)
        fig.update_layout(barmode="group",yaxis_title="%",xaxis=dict(tickangle=-20))
        st.plotly_chart(fig,use_container_width=True)

    # Gap analysis
    sec("Gap Analysis & Recommendations")
    for key,label in labels.items():
        uv,bv  = um[key],b[key]
        is_m   = key in ["net","gross","ebitda"]
        better = (uv>=bv) if is_m else (uv<=bv)
        gap    = abs(uv-bv)
        impact = gap*avg_r/100
        if better:
            st.markdown(f'<div class="alert alert-ok"><strong>{label}:</strong> {uv:.1f}% — outperforming ({bv:.1f}% benchmark). '
                        f'{gap:.1f}pp edge. Protect this as you scale.</div>', unsafe_allow_html=True)
        else:
            direction = "below" if is_m else "above"
            action    = f"Closing this gap could add ≈ {fmt(impact)}/month." if is_m else f"Optimising this could save ≈ {fmt(impact)}/month."
            st.markdown(f'<div class="alert alert-bad"><strong>{label}:</strong> {uv:.1f}% is {gap:.1f}pp {direction} the {btyp} benchmark of {bv:.1f}%. {action}</div>',
                        unsafe_allow_html=True)

    # AI benchmark analysis
    if st.session_state.gemini_key:
        sec("AI Benchmark Analysis")
        if st.button("Get AI Benchmark Insights", key="bm_ai"):
            gaps = [(labels[k],um[k],b[k]) for k in labels if
                    ((um[k]<b[k]) if k in ["net","gross","ebitda"] else (um[k]>b[k]))]
            with st.spinner("Benchmarking…"):
                out = call_llm(
                    f"{biz_context()}\n\n"
                    f"Underperforming vs {btyp} benchmark in: "
                    f"{', '.join([f'{n} ({uv:.1f}% vs {bv:.1f}%)' for n,uv,bv in gaps[:3]])}\n\n"
                    "Give: (1) Which gap is most urgent to fix and why? "
                    "(2) A concrete 3-step plan to fix the most critical gap. "
                    "(3) Realistic timeline. Use ₹, be specific."
                )
            st.markdown(f'<div class="ai-box"><div class="ai-header"><div class="ai-dot"></div>'
                        f'<strong style="color:{C["blue"]}">AI Benchmark Analysis</strong></div>{out}</div>',
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: AI INSIGHTS
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "AI Insights":
    ph("AI Insights", "Gemini-powered financial analysis, Q&A, and strategic planning")
    if need_data(): st.stop()

    if not st.session_state.gemini_key:
        st.markdown(f"""<div class="alert alert-warn">
            <strong>No API key configured.</strong>
            Open the <strong>Gemini AI Key</strong> section in the sidebar,
            paste your key, and return here.<br>
            Get a free key at
            <a href="https://aistudio.google.com/app/apikey" style="color:{C['accent']};">
            aistudio.google.com/app/apikey</a>
        </div>""", unsafe_allow_html=True)
        st.stop()

    df   = st.session_state.data
    btyp = st.session_state.biz_type
    b    = BENCHMARKS[btyp]

    t1, t2, t3, t4 = st.tabs([
        "Business Health Report",
        "Ask Your Data",
        "Goal Planner",
        "AI Chat"
    ])

    # ── Tab 1: Health Report ──────────────────────────────────
    with t1:
        sec("AI Business Health Report")
        st.markdown(f"""<div class="alert alert-info">
            Generates a full analysis of your business — health score, risks, strengths,
            one high-impact action, and an investor readiness verdict. ~400 words.
        </div>""", unsafe_allow_html=True)
        if st.button("Generate Health Report", use_container_width=True, key="hr_btn"):
            with st.spinner("Writing your report…"):
                out = call_llm(
                    f"{biz_context()}\n\n"
                    "Write a structured business health report:\n\n"
                    "## Executive Summary (2 sentences)\n"
                    "## Health Score (rate 1–10 with specific justification)\n"
                    "## Top 3 Risks (each with a specific mitigation tactic and ₹ impact)\n"
                    "## Top 2 Strengths (cite actual numbers)\n"
                    "## One High-Impact Action (most ROI, with estimated ₹ gain)\n"
                    "## Investor Readiness (honest seed/angel fund assessment)\n\n"
                    "Under 420 words. Use ₹. Be direct, no filler.",
                    system="You are a senior financial advisor specialising in Indian SMEs. "
                           "You give honest, data-driven assessments using ₹."
                )
            st.markdown(f'<div class="ai-box">{out}</div>', unsafe_allow_html=True)
            st.download_button("Download Report",out,
                               f"{st.session_state.biz_name or 'business'}_health_report.txt","text/plain")

    # ── Tab 2: Q&A ────────────────────────────────────────────
    with t2:
        sec("Ask Your Financial Data")
        st.markdown(f"""<div class="alert alert-info">
            Try: "What was my worst month and why?" &nbsp;&middot;&nbsp;
            "Is my salary ratio too high?" &nbsp;&middot;&nbsp;
            "How close am I to Series A metrics?" &nbsp;&middot;&nbsp;
            "Where should I cut costs first?"
        </div>""", unsafe_allow_html=True)
        q = st.text_area("Your question", height=72,
            placeholder="e.g. Which cost category is growing fastest and what should I do about it?")
        if st.button("Get Answer", use_container_width=True, key="qa_btn"):
            if q.strip():
                data_str = df[["Month","Revenue","Net_Profit","Net_Margin",
                               "EBITDA","Gross_Margin"]].to_string(index=False)
                with st.spinner("Analysing…"):
                    out = call_llm(
                        f"{biz_context()}\n\nMonthly data:\n{data_str}\n\n"
                        f"Industry benchmarks: Gross {b['gross']}%, EBITDA {b['ebitda']}%, Net {b['net']}%\n\n"
                        f"Question: {q}\n\n"
                        "Answer in 4–6 sentences. Be specific, cite numbers, use ₹."
                    )
                st.markdown(f'<div class="ai-box"><div class="ai-header"><div class="ai-dot"></div>'
                            f'<strong style="color:{C["blue"]}">Answer</strong></div>{out}</div>',
                            unsafe_allow_html=True)
            else:
                st.warning("Please enter a question.")

    # ── Tab 3: Goal Planner ───────────────────────────────────
    with t3:
        sec("Strategic Goal Planner")
        curr = df["Net_Profit"].mean()
        g1,g2,g3 = st.columns(3)
        with g1: target = st.number_input("Monthly Profit Target (₹)",0,10_000_000,
                                           int(max(curr*1.3,50_000)),5_000)
        with g2: tframe = st.selectbox("Timeframe",["1 month","3 months","6 months","12 months"])
        with g3: risk_a = st.selectbox("Risk Appetite",["Conservative","Moderate","Aggressive"])

        if st.button("Build Action Plan", use_container_width=True, key="gp_btn"):
            gap = max(target-curr,0)
            with st.spinner("Building strategy…"):
                out = call_llm(
                    f"{biz_context()}\n\n"
                    f"Goal: reach {fmt(target)}/month profit in {tframe}.\n"
                    f"Current avg: {fmt(curr)}/month. Gap: {fmt(gap)}/month.\n"
                    f"Risk appetite: {risk_a}.\n\n"
                    "Provide:\n"
                    "## Feasibility (is this gap closable in the timeframe? honest assessment)\n"
                    "## Action 1 (highest ROI — specific, measurable, with ₹ impact)\n"
                    "## Action 2\n"
                    "## Action 3\n"
                    "## Key Risk to Watch\n"
                    "## One Metric to Track Weekly\n\n"
                    "Under 320 words. Use ₹, be precise."
                )
            st.markdown(f'<div class="ai-box">{out}</div>', unsafe_allow_html=True)

    # ── Tab 4: AI Chat ────────────────────────────────────────
    with t4:
        sec("Conversational AI Chat")
        st.markdown(f"""<div class="alert alert-info">
            Have a back-and-forth conversation about your business finances.
            The AI has full context of your data throughout the conversation.
        </div>""", unsafe_allow_html=True)

        # Display history
        for msg in st.session_state.chat_hist:
            role_clr = C["blue"] if msg["role"]=="user" else C["green"]
            role_lbl = "You" if msg["role"]=="user" else "AI"
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:12px;
                        {'flex-direction:row-reverse;' if msg['role']=='user' else ''}">
                <div style="width:32px;height:32px;border-radius:50%;background:{role_clr}22;
                            border:1px solid {role_clr}44;flex-shrink:0;display:flex;
                            align-items:center;justify-content:center;
                            font-size:11px;font-weight:700;color:{role_clr};">{role_lbl}</div>
                <div style="background:{C['card']};border:1px solid {C['border']};
                            border-radius:12px;padding:12px 16px;max-width:80%;
                            font-size:13.5px;line-height:1.7;color:{C['text']};">
                    {msg['content']}
                </div>
            </div>""", unsafe_allow_html=True)

        # Input
        user_msg = st.chat_input("Ask anything about your finances…")
        if user_msg:
            st.session_state.chat_hist.append({"role":"user","content":user_msg})
            history_str = "\n".join([f"{m['role'].upper()}: {m['content']}"
                                     for m in st.session_state.chat_hist[-8:]])
            with st.spinner("Thinking…"):
                reply = call_llm(
                    f"Business context:\n{biz_context()}\n\n"
                    f"Conversation so far:\n{history_str}\n\n"
                    "Continue the conversation naturally. Be concise, use ₹."
                )
            st.session_state.chat_hist.append({"role":"assistant","content":reply})
            st.rerun()

        if st.session_state.chat_hist:
            if st.button("Clear Chat", key="clear_chat"):
                st.session_state.chat_hist = []
                st.rerun()

# ═══════════════════════════════════════════════════════════
#  PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "Settings":
    ph("Settings","Business profile and manual data entry")

    sec("Business Profile")
    pc1,pc2,pc3 = st.columns(3)
    with pc1:
        st.session_state.biz_name = st.text_input("Business Name",
            st.session_state.biz_name, placeholder="e.g. Riya's Café")
    with pc2:
        idx = list(BENCHMARKS.keys()).index(st.session_state.biz_type)
        st.session_state.biz_type = st.selectbox("Industry",list(BENCHMARKS.keys()),index=idx)
    with pc3:
        st.session_state.founded  = st.number_input("Year Founded",1990,2025,
                                                      st.session_state.founded)

    sec("Manual Data Entry")
    n_months = st.slider("Number of months", 3, 24, 6)
    rows = []
    for i in range(n_months):
        label = MONTHS[i % 12]
        with st.expander(f"Month {i+1} — {label}", expanded=(i==0)):
            ca,cb_ = st.columns(2)
            with ca:
                rev  = st.number_input("Revenue (₹)",       0,10_000_000,150_000,1_000,key=f"r{i}")
                rent = st.number_input("Rent (₹)",           0, 1_000_000, 15_000,  500,key=f"rn{i}")
                sal  = st.number_input("Salaries (₹)",       0, 5_000_000, 35_000,1_000,key=f"s{i}")
                util = st.number_input("Utilities (₹)",      0,   500_000,  4_000,  500,key=f"u{i}")
            with cb_:
                mkt  = st.number_input("Marketing (₹)",      0, 1_000_000,  7_000,  500,key=f"m{i}")
                cogs = st.number_input("COGS / Raw Mat (₹)", 0, 5_000_000, 55_000,1_000,key=f"cg{i}")
                misc = st.number_input("Miscellaneous (₹)",  0,   500_000,  3_500,  500,key=f"mc{i}")
            tot    = rent+sal+util+mkt+cogs+misc
            profit = rev-tot
            cls    = "alert-ok" if profit>=0 else "alert-bad"
            st.markdown(f'<div class="alert {cls}">Expenses: {fmt(tot)} &nbsp;|&nbsp; {"Profit" if profit>=0 else "Loss"}: {fmt(abs(profit))}</div>',
                        unsafe_allow_html=True)
            rows.append(dict(Month=label,Revenue=rev,Rent=rent,Salaries=sal,
                             Utilities=util,Marketing=mkt,COGS=cogs,
                             Misc=misc,Total_Expenses=tot))

    if st.button("Save & Analyse", use_container_width=True):
        st.session_state.data  = calc_metrics(pd.DataFrame(rows))
        st.session_state.ready = True
        st.session_state.page  = "Overview"
        st.rerun()

    sec("Upload CSV / Excel")
    with st.expander("Required columns"):
        tpl = pd.DataFrame({"Month":["January","February"],
            "Revenue":[150000,165000],"Rent":[15000,15000],
            "Salaries":[35000,36000],"Utilities":[4000,4200],
            "Marketing":[6000,7000],"COGS":[55000,60000],
            "Misc":[3000,3500],"Total_Expenses":[118000,125700]})
        st.dataframe(tpl,use_container_width=True)
        st.download_button("Download Template",tpl.to_csv(index=False).encode(),
                           "venturescope_template.csv","text/csv")

    uploaded = st.file_uploader("Upload CSV or Excel",type=["csv","xlsx","xls"])
    if uploaded:
        try:
            raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.dataframe(raw.head(),use_container_width=True)
            needed = ["Revenue","Rent","Salaries","Utilities","Marketing","COGS","Misc"]
            for col in needed:
                if col not in raw.columns:
                    raw[col] = raw.get("Raw_Materials",0) if col=="COGS" else 0
            if "Total_Expenses" not in raw.columns:
                raw["Total_Expenses"] = raw[needed[1:]].sum(axis=1)
            if "Month" not in raw.columns:
                raw.insert(0,"Month",[MONTHS[i%12] for i in range(len(raw))])
            for col in needed+["Total_Expenses","Revenue"]:
                raw[col] = pd.to_numeric(raw[col],errors="coerce").fillna(0)
            st.session_state.data  = calc_metrics(raw[["Month","Revenue","Rent","Salaries",
                "Utilities","Marketing","COGS","Misc","Total_Expenses"]].copy())
            st.session_state.ready = True
            st.success(f"Loaded {len(st.session_state.data)} months successfully. Go to Overview.")
        except Exception as ex:
            st.error(f"Error reading file: {ex}")

    if st.session_state.ready:
        sec("Loaded Data")
        st.dataframe(
            st.session_state.data[["Month","Revenue","Total_Expenses","Gross_Profit",
                                   "EBITDA","Net_Profit","Net_Margin"]].style.format({
                "Revenue":"₹{:,.0f}","Total_Expenses":"₹{:,.0f}",
                "Gross_Profit":"₹{:,.0f}","EBITDA":"₹{:,.0f}",
                "Net_Profit":"₹{:,.0f}","Net_Margin":"{:.1f}%",
            }).background_gradient(subset=["Net_Profit"],cmap="RdYlGn"),
            use_container_width=True)