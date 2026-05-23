"""
Stock Screener — Streamlit web frontend.

Reads daily CSV + brief outputs from data/ and displays them with
conviction scoring, top-pick cards, a CIO daily brief, and interactive filters.

Streamlit secrets required for the trigger button:
  GH_TOKEN       — GitHub personal access token (scope: workflow)
  GITHUB_OWNER   — GitHub username / org that owns the private screener repo
  GITHUB_REPO    — private repo name (default: stock-screener)
"""

import glob
import os
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

INSIDER_COLORS = {
    "BUYING":  "#4caf91",
    "SELLING": "#e05c6a",
    "NEUTRAL": "#9aa0aa",
    "—":       "#555a63",
}

SCREEN_META = {
    "below30Q":  {"label": "Below 30Q SMA",  "color": "#4da6ff", "sma_col": "30Q SMA"},
    "below50Q":  {"label": "Below 50Q SMA",  "color": "#fd7e14", "sma_col": "50Q SMA"},
    "below100Q": {"label": "Below 100Q SMA", "color": "#e05c6a", "sma_col": "100Q SMA"},
    "momentum":  {"label": "Momentum",       "color": "#4caf91", "sma_col": "20Q SMA"},
}

VALUE_COLS = [
    "Score", "Ticker", "Company", "Price", "% Off 52W High", "% Below SMA",
    "P/B Ratio", "Div Yield", "ROE",
    "Short % Float", "Short Ratio (Days)",
    "Insider Signal", "Insider Net Value (6mo)", "Insider Buys", "Insider Sells",
    "Market Cap", "Sector", "Exchange",
]

MOM_COLS = [
    "Score", "Ticker", "Company", "Price", "% Above 20Q SMA",
    "20Q SMA", "30Q SMA", "50Q SMA",
    "P/B Ratio", "Div Yield", "ROE",
    "Short % Float", "Short Ratio (Days)",
    "Insider Signal", "Insider Net Value (6mo)", "Insider Buys", "Insider Sells",
    "Market Cap", "Sector", "Exchange",
]

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border-radius: 10px;
    padding: 12px 16px;
  }
  .brief-card {
    border-radius: 12px;
    padding: 20px 24px;
    border-left: 4px solid #4da6ff;
    background: var(--secondary-background-color);
    margin-bottom: 4px;
  }
  .brief-label {
    font-size: 0.68rem;
    color: #4da6ff;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
  }
  .brief-text {
    font-size: 0.95rem;
    line-height: 1.7;
  }
  .overlap-badge {
    display: inline-block;
    background: #4caf9133;
    color: #4caf91;
    border: 1px solid #4caf9155;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-left: 6px;
    vertical-align: middle;
  }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────

def available_dates() -> list[str]:
    files = glob.glob(os.path.join(DATA_DIR, "screen_*_below30Q.csv"))
    dates = []
    for f in files:
        m = re.search(r"screen_(\d{4}-\d{2}-\d{2})_", os.path.basename(f))
        if m:
            dates.append(m.group(1))
    return sorted(set(dates), reverse=True)


@st.cache_data(ttl=300)
def load_csv(date_str: str, screen_type: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"screen_{date_str}_{screen_type}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype=str).fillna("—")


@st.cache_data(ttl=300)
def load_brief(date_str: str) -> str | None:
    path = os.path.join(DATA_DIR, f"brief_{date_str}.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


@st.cache_data(ttl=300)
def load_reports(date_str: str) -> dict[str, str]:
    paths = glob.glob(os.path.join(DATA_DIR, f"report_{date_str}_*.txt"))
    result: dict[str, str] = {}
    for path in sorted(paths):
        m = re.search(r"report_\d{4}-\d{2}-\d{2}_(.+)\.txt$", os.path.basename(path))
        if m:
            with open(path, encoding="utf-8") as f:
                result[m.group(1)] = f.read()
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

import math as _math

def _f(s) -> float:
    """Parse a formatted string like '1.23%' or '$5.2B' to a float."""
    try:
        return float(str(s).replace("%", "").replace("+", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return float("nan")


def _parse_insider_val(s) -> float:
    """Parse '+$5.6M' → 5_600_000, '-$1.2K' → -1_200, '—' → 0."""
    if not s or str(s) in ("—", "", "nan"):
        return 0.0
    s = str(s).strip()
    neg = s.startswith("-")
    s = s.replace("$", "").replace("+", "").replace("-", "").strip()
    try:
        if "T" in s:
            val = float(s.replace("T", "")) * 1e12
        elif "B" in s:
            val = float(s.replace("B", "")) * 1e9
        elif "M" in s:
            val = float(s.replace("M", "")) * 1e6
        elif "K" in s:
            val = float(s.replace("K", "")) * 1e3
        else:
            val = float(s)
        return -val if neg else val
    except (ValueError, AttributeError):
        return 0.0


def _insider_score(net_val: float) -> float:
    """Log-scaled 0–3 pts on net buying amount. Selling = −1."""
    if net_val < 0:
        return -1.0
    if net_val < 50_000:          # noise / no activity
        return 0.0
    # $50K → ~0 pts, $500K → ~1.3 pts, $5M → ~2.6 pts, $10M+ → 3 pts
    return min(_math.log10(net_val / 50_000) / _math.log10(200) * 3, 3.0)


def compute_conviction(df: pd.DataFrame, is_momentum: bool = False) -> pd.DataFrame:
    """
    Score each stock 0–10 based on:
      - Insider net value  (0–3 pts, log-scaled on $ amount; selling = −1)
      - P/B ratio          (0–2 pts, value screens only — lower is better)
      - ROE                (0–3 pts — higher is better, capped at 30%)
      - Dividend yield     (0–2 pts — scales with yield up to 5%+)

    Momentum stocks skip the P/B factor (max raw = 8 vs 10 for value).
    Both are normalised to 0–10.
    """
    df = df.copy()

    ins_score  = df["Insider Net Value (6mo)"].apply(_parse_insider_val).apply(_insider_score)
    roe_score  = df["ROE"].apply(_f).fillna(0).clip(0, 30) / 30 * 3
    div_score  = df["Div Yield"].apply(_f).fillna(0).clip(0, 10) / 5 * 2

    if is_momentum:
        raw     = ins_score + roe_score + div_score
        max_raw = 8.0
    else:
        pb_score = (2 - df["P/B Ratio"].apply(_f).clip(0, 2)) / 2 * 2
        raw      = ins_score + pb_score + roe_score + div_score
        max_raw  = 10.0

    df.insert(0, "Score", (raw / max_raw * 10).clip(0, 10).round(1))
    return df.sort_values("Score", ascending=False)


def _style_insider(val: str) -> str:
    return f"color: {INSIDER_COLORS.get(str(val), INSIDER_COLORS['—'])}; font-weight: 600"


def _apply_filters(df: pd.DataFrame, search: str, sector: str, insider: str, pb_max: float) -> pd.DataFrame:
    if search:
        mask = (
            df["Ticker"].str.contains(search, case=False, na=False)
            | df["Company"].str.contains(search, case=False, na=False)
        )
        df = df[mask]
    if sector != "All":
        df = df[df["Sector"] == sector]
    if insider != "All":
        df = df[df["Insider Signal"] == insider]
    if pb_max < 2.0 and "P/B Ratio" in df.columns:
        df = df[df["P/B Ratio"].apply(_f) <= pb_max]
    return df


def _cnt(df) -> int:
    return len(df) if df is not None else 0


def _kpi(df, metric) -> str:
    if df is None or df.empty:
        return "—"
    if metric == "buying_pct":
        return f"{(df['Insider Signal'] == 'BUYING').sum() / len(df) * 100:.0f}%"
    if metric == "avg_pb":
        v = df["P/B Ratio"].apply(_f).dropna()
        return f"{v.mean():.2f}×" if len(v) else "—"
    if metric == "with_div":
        return f"{(df['Div Yield'].apply(_f) > 0).sum() / len(df) * 100:.0f}%"
    return "—"


def render_top_cards(df: pd.DataFrame, overlap_tickers: set, n: int = 5) -> None:
    """Render top-N conviction cards. df must already have a Score column and _screen column."""
    if df is None or df.empty:
        return
    top = df.head(n)
    cols = st.columns(len(top))
    for i, (_, row) in enumerate(top.iterrows()):
        screen   = row.get("_screen", "")
        color    = next((v["color"] for k, v in SCREEN_META.items() if v["label"] == screen), "#4da6ff")
        ins      = row.get("Insider Signal", "—")
        ins_col  = INSIDER_COLORS.get(ins, INSIDER_COLORS["—"])
        ticker   = row.get("Ticker", "")
        company  = row.get("Company", "")[:22]
        score    = row.get("Score", 0)
        overlap  = "⭐ " if ticker in overlap_tickers else ""
        with cols[i]:
            st.markdown(f"""
<div style="background:var(--secondary-background-color);border-radius:12px;
            padding:16px 14px;border-top:3px solid {color};text-align:center;height:100%">
  <div style="font-size:1.45rem;font-weight:800;letter-spacing:-0.02em">{overlap}{ticker}</div>
  <div style="font-size:0.7rem;opacity:0.55;margin-bottom:10px;overflow:hidden;
              white-space:nowrap;text-overflow:ellipsis">{company}</div>
  <div style="font-size:1.25rem;font-weight:700;color:{color}">{score}/10</div>
  <div style="font-size:0.6rem;opacity:0.45;letter-spacing:0.08em;
              text-transform:uppercase;margin-bottom:10px">Conviction</div>
  <hr style="opacity:0.12;margin:8px 0">
  <table style="width:100%;font-size:0.72rem;border-collapse:collapse">
    <tr><td style="opacity:0.5;text-align:left">P/B</td>
        <td style="font-weight:600;text-align:right">{row.get('P/B Ratio','—')}</td></tr>
    <tr><td style="opacity:0.5;text-align:left">Insider</td>
        <td style="font-weight:600;text-align:right;color:{ins_col}">{ins}</td></tr>
    <tr><td style="opacity:0.5;text-align:left">ROE</td>
        <td style="font-weight:600;text-align:right">{row.get('ROE','—')}</td></tr>
    <tr><td style="opacity:0.5;text-align:left">Div</td>
        <td style="font-weight:600;text-align:right">{row.get('Div Yield','—')}</td></tr>
  </table>
  <div style="margin-top:10px;font-size:0.62rem;border-radius:4px;padding:3px 6px;
              background:{color}22;color:{color};font-weight:600">{screen}</div>
</div>""", unsafe_allow_html=True)


def render_table(df: pd.DataFrame, col_order: list[str]) -> None:
    cols = [c for c in col_order if c in df.columns]
    styled = df[cols].style.map(_style_insider, subset=["Insider Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=460)


def render_charts(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    c1, c2 = st.columns(2)
    with c1:
        sd = df["Sector"].value_counts().reset_index()
        sd.columns = ["Sector", "Count"]
        fig = px.pie(sd, values="Count", names="Sector", title="Sector Breakdown",
                     hole=0.4, template="plotly_dark",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(t=40,b=0,l=0,r=0), height=300,
                          paper_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pb = df["P/B Ratio"].apply(_f).dropna()
        if len(pb):
            fig = px.histogram(pb, title="P/B Distribution", nbins=15, template="plotly_dark",
                               color_discrete_sequence=["#4da6ff"],
                               labels={"value": "P/B Ratio", "count": "Stocks"})
            fig.update_layout(margin=dict(t=40,b=0,l=0,r=0), height=300, showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


def render_highlights(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🟢 Insider Buying**")
        b = df[df["Insider Signal"] == "BUYING"][["Ticker","Company","Insider Net Value (6mo)","Sector"]].head(5)
        if not b.empty:
            st.dataframe(b, hide_index=True, use_container_width=True, height=215)
        else:
            st.caption("None this run")
    with c2:
        st.markdown("**💰 Highest Dividend**")
        d = df.copy()
        d["_d"] = d["Div Yield"].apply(_f)
        top = d[d["_d"] > 0].nlargest(5, "_d")[["Ticker","Company","Div Yield","Sector"]]
        if not top.empty:
            st.dataframe(top, hide_index=True, use_container_width=True, height=215)
        else:
            st.caption("None paying dividends")
    with c3:
        st.markdown("**📉 Lowest P/B**")
        p = df.copy()
        p["_p"] = p["P/B Ratio"].apply(_f)
        top = p[p["_p"] > 0].nsmallest(5, "_p")[["Ticker","Company","P/B Ratio","Sector"]]
        if not top.empty:
            st.dataframe(top, hide_index=True, use_container_width=True, height=215)
        else:
            st.caption("No P/B data")


def trigger_workflow(force: bool) -> None:
    try:
        token = st.secrets["GH_TOKEN"]
        owner = st.secrets["GITHUB_OWNER"]
        repo  = st.secrets.get("GITHUB_REPO", "stock-screener")
    except KeyError as exc:
        st.error(f"Missing secret: {exc}")
        return
    resp = requests.post(
        f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/daily_screener.yml/dispatches",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"},
        json={"ref": "master", "inputs": {"force_reports": str(force).lower()}},
        timeout=10,
    )
    if resp.status_code == 204:
        st.success("Run triggered! Results update in ~30–45 min.")
    else:
        st.error(f"GitHub API {resp.status_code}: {resp.text[:200]}")


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Stock Screener")
    st.caption("NYSE + NASDAQ · Market Cap ≥ $2B")

    dates = available_dates()
    if not dates:
        st.warning("No data yet.")
        selected_date = prev_date = None
    else:
        selected_date = st.selectbox(
            "Date", options=dates, index=0,
            format_func=lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%b %d, %Y"),
        )
        prev_date = dates[1] if len(dates) > 1 else None

    if selected_date:
        df_30q  = load_csv(selected_date, "below30Q")
        df_50q  = load_csv(selected_date, "below50Q")
        df_100q = load_csv(selected_date, "below100Q")
        df_mom  = load_csv(selected_date, "momentum")
        p30  = load_csv(prev_date, "below30Q")  if prev_date else None
        p50  = load_csv(prev_date, "below50Q")  if prev_date else None
        p100 = load_csv(prev_date, "below100Q") if prev_date else None
        pmom = load_csv(prev_date, "momentum")  if prev_date else None

        st.divider()
        st.subheader("Counts")
        st.metric("Below 30Q SMA",  _cnt(df_30q),  (_cnt(df_30q) -_cnt(p30))  if prev_date else None)
        st.metric("Below 50Q SMA",  _cnt(df_50q),  (_cnt(df_50q) -_cnt(p50))  if prev_date else None)
        st.metric("Below 100Q SMA", _cnt(df_100q), (_cnt(df_100q)-_cnt(p100)) if prev_date else None)
        st.metric("Momentum",       _cnt(df_mom),  (_cnt(df_mom) -_cnt(pmom)) if prev_date else None)

    st.divider()
    st.subheader("Run Screener")
    force = st.toggle("Force research reports", value=False)
    if st.button("Run Now", type="primary", use_container_width=True):
        trigger_workflow(force)
    st.caption("Needs GH_TOKEN + GITHUB_OWNER in Streamlit secrets.")

# ── Guard ──────────────────────────────────────────────────────────────────────

if not selected_date:
    st.info("No data found. Trigger a run or wait for the daily 7 AM ET job.")
    st.stop()

# ── Load & score data ──────────────────────────────────────────────────────────

df_30q  = compute_conviction(df_30q)                    if df_30q  is not None else None
df_50q  = compute_conviction(df_50q)                    if df_50q  is not None else None
df_100q = compute_conviction(df_100q)                   if df_100q is not None else None
df_mom  = compute_conviction(df_mom, is_momentum=True)  if df_mom  is not None else None

# Combined ranked list across all screens (deepest value wins on dedup)
_frames = []
for _df, _key in [(df_100q,"below100Q"),(df_50q,"below50Q"),(df_30q,"below30Q"),(df_mom,"momentum")]:
    if _df is not None and not _df.empty:
        _d = _df.copy()
        _d["_screen"] = SCREEN_META[_key]["label"]
        _frames.append(_d)
combined = pd.concat(_frames).drop_duplicates("Ticker", keep="first").sort_values("Score", ascending=False) if _frames else pd.DataFrame()

# Cross-screen overlap: tickers in any value screen AND momentum
value_tickers = set()
for _df in [df_30q, df_50q, df_100q]:
    if _df is not None:
        value_tickers.update(_df["Ticker"].tolist())
mom_tickers = set(df_mom["Ticker"].tolist()) if df_mom is not None else set()
overlap_tickers = value_tickers & mom_tickers

# ── Daily Brief ────────────────────────────────────────────────────────────────

fmt_date = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%B %d, %Y")
st.markdown(f"### {fmt_date}")

brief = load_brief(selected_date)
if brief:
    st.markdown(f"""
<div class="brief-card">
  <div class="brief-label">📋 Daily Brief</div>
  <div class="brief-text">{brief}</div>
</div>""", unsafe_allow_html=True)
else:
    st.caption("No brief for this date — generated automatically from the next screener run.")

st.divider()

# ── Top Conviction Picks ───────────────────────────────────────────────────────

if overlap_tickers:
    overlap_list = ", ".join(sorted(overlap_tickers))
    st.markdown(
        f"**Top Conviction Picks** &nbsp;"
        f'<span class="overlap-badge">⭐ {len(overlap_tickers)} cross-screen: {overlap_list}</span>',
        unsafe_allow_html=True,
    )
    st.caption("⭐ = appears in both a value screen and momentum — highest conviction signal")
else:
    st.markdown("**Top Conviction Picks** — ranked across all screens")

render_top_cards(combined, overlap_tickers, n=5)
st.markdown("")
with st.expander("How is the conviction score calculated?"):
    st.markdown("""
| Factor | Max pts | Logic |
|---|---|---|
| **Insider net buying** | 3 | Log-scaled on $ amount — $50K ≈ 0, $500K ≈ 1.3, $5M ≈ 2.6, $10M+ = 3. Net selling = −1 |
| **P/B ratio** | 2 | Value screens only (not momentum) — P/B 0 = 2 pts, P/B 2 = 0 pts |
| **Return on equity** | 3 | Scales linearly — ROE 15% = 1.5 pts, ROE 30%+ = 3 pts |
| **Dividend yield** | 2 | Scales with yield — 2.5% = 1 pt, 5%+ = 2 pts |

Value screen max = 10 pts · Momentum max = 8 pts (P/B excluded) — both normalised to **0–10**. ⭐ = appears in both a value screen and momentum.
    """)
st.divider()

# ── KPI bar ────────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Unique Tickers",   f"{combined['Ticker'].nunique():,}" if not combined.empty else "—")
k2.metric("Value Hits",       f"{_cnt(df_30q)+_cnt(df_50q)+_cnt(df_100q):,}")
k3.metric("Momentum Hits",    f"{_cnt(df_mom):,}")
k4.metric("Insider Buying",   _kpi(combined, "buying_pct"))
k5.metric("Avg P/B",          _kpi(combined, "avg_pb"))

st.divider()

# ── Global filters ─────────────────────────────────────────────────────────────

sector_opts = ["All"] + sorted(
    s for s in combined["Sector"].dropna().unique() if s and s != "—"
) if not combined.empty else ["All"]

fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
search_q       = fc1.text_input("🔍 Search", placeholder="Ticker or company", label_visibility="collapsed")
sector_filter  = fc2.selectbox("Sector", sector_opts, label_visibility="collapsed")
insider_filter = fc3.selectbox("Insider", ["All","BUYING","NEUTRAL","SELLING","—"], label_visibility="collapsed")
pb_max         = fc4.slider("Max P/B", min_value=0.0, max_value=2.0, value=2.0, step=0.1)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_ov, tab1, tab2, tab3, tab4, tab_diag = st.tabs([
    "📊 Overview", "🔵 Below 30Q", "🟠 Below 50Q", "🔴 Below 100Q", "🚀 Momentum", "🔍 Diagnostics"
])

# Overview ─────────────────────────────────────────────────────────────────────

with tab_ov:
    bar_data = pd.DataFrame({
        "Screen": ["Below 30Q","Below 50Q","Below 100Q","Momentum"],
        "Count":  [_cnt(df_30q),_cnt(df_50q),_cnt(df_100q),_cnt(df_mom)],
        "Color":  ["#4da6ff","#fd7e14","#e05c6a","#4caf91"],
    })
    fig = px.bar(bar_data, x="Screen", y="Count", color="Screen", text="Count",
                 color_discrete_map=dict(zip(bar_data["Screen"], bar_data["Color"])),
                 title="Stocks per Screen", template="plotly_dark")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=280, margin=dict(t=40,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highlights Across All Screens")
    render_highlights(combined)

    st.divider()
    st.subheader("Sector Exposure")
    if not combined.empty:
        frames_heat = []
        for key, df in [("below30Q",df_30q),("below50Q",df_50q),
                         ("below100Q",df_100q),("momentum",df_mom)]:
            if df is not None:
                sc = df["Sector"].value_counts().reset_index()
                sc.columns = ["Sector","Count"]
                sc["Screen"] = SCREEN_META[key]["label"]
                frames_heat.append(sc)
        if frames_heat:
            ss = pd.concat(frames_heat)
            pivot = ss.pivot_table(index="Sector", columns="Screen", values="Count", fill_value=0)
            fig = px.imshow(pivot, text_auto=True, aspect="auto",
                            color_continuous_scale="Blues", template="plotly_dark",
                            title="Stock count by Sector × Screen")
            fig.update_layout(height=420, margin=dict(t=40,b=0),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

# Value tab helper ─────────────────────────────────────────────────────────────

def value_tab(df, label, sma_col):
    if df is None or df.empty:
        st.info(f"No {label} data for {selected_date}.")
        return
    filtered = _apply_filters(df, search_q, sector_filter, insider_filter, pb_max)
    st.caption(f"{len(filtered):,} of {len(df):,} stocks")

    st.markdown(f"**Top picks — {label}**")
    render_top_cards(filtered, overlap_tickers, n=5)
    st.markdown("")

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Shown",          f"{len(filtered):,}")
    m2.metric("Insider Buying", _kpi(filtered,"buying_pct"))
    m3.metric("Avg P/B",        _kpi(filtered,"avg_pb"))
    m4.metric("With Dividend",  _kpi(filtered,"with_div"))

    cols = [sma_col if c == "% Below SMA" else c for c in VALUE_COLS]
    render_table(filtered, cols)
    render_charts(filtered)
    st.subheader("Highlights"); render_highlights(filtered)


with tab1:
    st.markdown("Price below its **30-quarter (7.5 yr) SMA** — trading below long-run average.")
    value_tab(df_30q, "Below 30Q", "30Q SMA")

with tab2:
    st.markdown("Price below its **50-quarter (12.5 yr) SMA** — moderate undervaluation.")
    value_tab(df_50q, "Below 50Q", "50Q SMA")

with tab3:
    st.markdown("Price below its **100-quarter (25 yr) SMA** — deep undervaluation vs 25-year history.")
    value_tab(df_100q, "Below 100Q", "100Q SMA")

# Momentum tab ─────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("**Bull-aligned quarterly SMAs**, near 20Q support, insider buying or neutral.")
    if df_mom is None or df_mom.empty:
        st.info(f"No momentum data for {selected_date}.")
    else:
        filtered_mom = _apply_filters(df_mom, search_q, sector_filter, insider_filter, pb_max)
        st.caption(f"{len(filtered_mom):,} of {len(df_mom):,} stocks")

        st.markdown("**Top picks — Momentum**")
        render_top_cards(filtered_mom, overlap_tickers, n=5)
        st.markdown("")

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Shown",          f"{len(filtered_mom):,}")
        m2.metric("Insider Buying", _kpi(filtered_mom,"buying_pct"))
        m3.metric("Avg P/B",        _kpi(filtered_mom,"avg_pb"))
        m4.metric("With Dividend",  _kpi(filtered_mom,"with_div"))

        render_table(filtered_mom, MOM_COLS)
        render_charts(filtered_mom)
        st.subheader("Highlights"); render_highlights(filtered_mom)

        reports = load_reports(selected_date)
        if reports:
            st.divider()
            st.subheader(f"📄 Research Reports ({len(reports)})")
            st.caption("Claude-generated investment memos for new momentum names.")
            for ticker, text in reports.items():
                with st.expander(ticker):
                    st.text(text)
        else:
            st.caption("No research reports for this date — generated on Mondays or when forced.")

# ── Diagnostics tab ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_financials(ticker: str) -> dict:
    import yfinance as yf
    t   = yf.Ticker(ticker)
    return {"income": t.income_stmt, "info": t.info}


def _run_diagnostic(ticker: str) -> None:
    with st.spinner(f"Loading {ticker} financials…"):
        try:
            data = _fetch_financials(ticker)
        except Exception as exc:
            st.error(f"Could not load data for {ticker}: {exc}")
            return

    income = data["income"]
    info   = data["info"]

    if income is None or income.empty:
        st.error(f"No financial data found for {ticker}.")
        return

    # ── Extract series ─────────────────────────────────────────────────────────
    def _row(label):
        for l in [label, label.replace(" ", ""), label.lower()]:
            if l in income.index:
                return income.loc[l].sort_index()
        return None

    rev = _row("Total Revenue")
    if rev is None:
        st.error(f"No revenue data for {ticker}.")
        return

    rev_b   = rev / 1e9
    years   = [str(d.year) for d in rev_b.index]
    gp      = _row("Gross Profit")
    op      = _row("Operating Income")
    g_margin = (gp / rev * 100).clip(-100, 100) if gp is not None else None
    o_margin = (op / rev * 100).clip(-100, 100) if op is not None else None

    # ── Signals ────────────────────────────────────────────────────────────────
    signals    = []
    red_flags  = 0
    green_flags = 0

    def sig(icon, label, detail):
        nonlocal red_flags, green_flags
        signals.append((icon, label, detail))
        if icon == "🔴": red_flags  += 1
        if icon == "✅": green_flags += 1

    if len(rev_b) >= 3:
        delta = float(rev_b.iloc[-1] - rev_b.iloc[-3])
        pct   = delta / abs(float(rev_b.iloc[-3])) * 100 if rev_b.iloc[-3] != 0 else 0
        if pct > 5:
            sig("✅", "Revenue trend", f"Growing +{pct:.0f}% over last 2 years")
        elif pct < -10:
            sig("🔴", "Revenue trend", f"Down {pct:.0f}% — check if cycle-driven or structural")
        else:
            sig("🟡", "Revenue trend", f"Roughly flat ({pct:+.0f}%) — watch for inflection")

    if g_margin is not None and len(g_margin) >= 3:
        chg = float(g_margin.iloc[-1] - g_margin.iloc[-3])
        if chg > 1:
            sig("✅", "Gross margin", f"Expanding +{chg:.1f}pp over 2 years")
        elif chg < -2:
            sig("🔴", "Gross margin", f"Compressing {chg:.1f}pp — pricing power concern")
        else:
            sig("🟡", "Gross margin", f"Stable ({chg:+.1f}pp)")

    if o_margin is not None and len(o_margin) >= 3:
        chg = float(o_margin.iloc[-1] - o_margin.iloc[-3])
        if chg > 1:
            sig("✅", "Op. margin", f"Expanding +{chg:.1f}pp over 2 years")
        elif chg < -2:
            sig("🔴", "Op. margin", f"Compressing {chg:.1f}pp — structural pressure likely")
        else:
            sig("🟡", "Op. margin", f"Stable ({chg:+.1f}pp)")

    short_pct = info.get("shortPercentOfFloat")
    if short_pct is not None:
        sp = short_pct * 100
        if sp > 15:
            sig("🔴", "Short interest", f"{sp:.1f}% of float — smart money may see structural issue")
        elif sp < 5:
            sig("✅", "Short interest", f"{sp:.1f}% of float — low")
        else:
            sig("🟡", "Short interest", f"{sp:.1f}% of float — moderate")

    # In today's screen?
    if not combined.empty and ticker in combined["Ticker"].values:
        row    = combined[combined["Ticker"] == ticker].iloc[0]
        screen = row.get("_screen", "a screen")
        score  = row.get("Score", "—")
        signals.append(("📊", "In today's screen", f"{screen} · conviction score {score}/10"))

    # ── Verdict ────────────────────────────────────────────────────────────────
    if red_flags >= 2:
        verdict, vc = "🔴  Structural Risk", "#e05c6a"
        vdesc = "Multiple red flags. Investigate market-share trends and whether revenue peaks are getting lower before sizing."
    elif red_flags == 0 and green_flags >= 2:
        verdict, vc = "🟢  Likely Cyclical", "#4caf91"
        vdesc = "Fundamentals look intact. Decline appears externally driven — classic mean-reversion setup."
    else:
        verdict, vc = "🟡  Unclear — Do More Work", "#fd7e14"
        vdesc = "Mixed signals. Check 10-year revenue history, peer comparisons, and what management is doing with capital."

    # ── Render ─────────────────────────────────────────────────────────────────
    company = info.get("longName", ticker)
    sector  = info.get("sector", "—")
    st.markdown(f"#### {company} ({ticker}) · {sector}")

    st.markdown(f"""
<div style="background:var(--secondary-background-color);border-radius:12px;
            padding:16px 20px;border-left:4px solid {vc};margin-bottom:16px">
  <div style="font-size:1.05rem;font-weight:700;color:{vc};margin-bottom:4px">{verdict}</div>
  <div style="font-size:0.87rem;opacity:0.85">{vdesc}</div>
</div>""", unsafe_allow_html=True)

    # Charts
    cc1, cc2 = st.columns(2)
    with cc1:
        fig = px.bar(x=years, y=rev_b.values, title="Annual Revenue ($B)",
                     template="plotly_dark", color_discrete_sequence=["#4da6ff"],
                     labels={"x": "Year", "y": "$B"})
        fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with cc2:
        if g_margin is not None or o_margin is not None:
            fig = go.Figure()
            if g_margin is not None:
                fig.add_trace(go.Scatter(x=years, y=g_margin.values, name="Gross Margin %",
                                         line=dict(color="#4caf91", width=2)))
            if o_margin is not None:
                fig.add_trace(go.Scatter(x=years, y=o_margin.values, name="Op. Margin %",
                                         line=dict(color="#4da6ff", width=2)))
            fig.update_layout(title="Margin Trends (%)", template="plotly_dark", height=280,
                              margin=dict(t=40, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    # Signals
    st.markdown("**Signals**")
    for icon, label, detail in signals:
        st.markdown(f"{icon} &nbsp; **{label}** — {detail}", unsafe_allow_html=True)

    st.divider()
    with st.expander("Diagnostic framework reference"):
        st.markdown("""
**Type 1 — External cycle** · Whole industry down equally. Mean reversion is almost mechanical.
**Type 2 — Temporary self-inflicted wound** · Bad acquisition, CEO departure, guidance miss. Recoverable if franchise intact.
**Type 3 — Structural disruption** · New force permanently shrinking TAM or compressing margins. Often looks like Type 1 for years.
**Type 4 — Terminal decline** · Business model irreversibly broken. Rare in large caps.

---
**Key checks:**
- 10-year revenue: lower highs each cycle = structural red flag
- Gross/op margin trend: staircase down across cycles = structural
- Are all peers down equally? If yes → cyclical. If only this name → dig deeper
- Short interest >15%: smart money may have done the work
- Management actions vs words: buybacks = confidence, asset sales = distress
- Are your customers in shrinking industries?
        """)


with tab_diag:
    st.markdown("### Cycle vs. Structural Diagnostic")
    st.caption(
        "Paste any ticker to check whether its decline is a cyclical opportunity or a structural value trap."
    )

    dc1, dc2 = st.columns([3, 1])
    diag_ticker = dc1.text_input(
        "Ticker", placeholder="e.g. CAG, MKC, IP, GPN",
        label_visibility="collapsed", key="diag_ticker"
    )
    run_diag = dc2.button("Analyse →", type="primary", use_container_width=True)

    if run_diag and diag_ticker:
        _run_diagnostic(diag_ticker.strip().upper())
