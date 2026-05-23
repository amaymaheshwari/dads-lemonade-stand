"""
Stock Screener — Streamlit web frontend.

Reads daily CSV outputs from the data/ directory and displays them with
KPI cards, interactive filters, charts, and a manual trigger button.

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
    "BUYING":  "#198754",
    "SELLING": "#dc3545",
    "NEUTRAL": "#6c757d",
    "—":       "#adb5bd",
}

SCREENS = {
    "below30Q":  {"label": "Below 30Q SMA",  "color": "#0d6efd", "sma_col": "30Q SMA"},
    "below50Q":  {"label": "Below 50Q SMA",  "color": "#fd7e14", "sma_col": "50Q SMA"},
    "below100Q": {"label": "Below 100Q SMA", "color": "#dc3545", "sma_col": "100Q SMA"},
    "momentum":  {"label": "Momentum",       "color": "#198754", "sma_col": "20Q SMA"},
}

VALUE_COLS = [
    "Ticker", "Company", "Price", "% Off 52W High", "% Below SMA",
    "P/B Ratio", "Div Yield", "ROE",
    "Short % Float", "Short Ratio (Days)",
    "Insider Signal", "Insider Net Value (6mo)", "Insider Buys", "Insider Sells",
    "Market Cap", "Sector", "Exchange",
]

MOM_COLS = [
    "Ticker", "Company", "Price", "% Above 20Q SMA",
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
  .kpi-label { font-size: 0.75rem; opacity: 0.6; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi-value { font-size: 1.6rem; font-weight: 700; line-height: 1.2; }
  .kpi-sub   { font-size: 0.78rem; opacity: 0.6; margin-top: 2px; }
  .buying    { color: #4caf91; font-weight: 600; }
  .selling   { color: #e05c6a; font-weight: 600; }
  .neutral   { color: #9aa0aa; font-weight: 600; }
  div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border-radius: 10px;
    padding: 12px 16px;
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

def _to_float(s) -> float:
    try:
        return float(str(s).replace("%", "").replace("+", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return float("nan")


def _style_insider(val: str) -> str:
    color = INSIDER_COLORS.get(str(val), INSIDER_COLORS["—"])
    return f"color: {color}; font-weight: 600"


def _apply_filters(df: pd.DataFrame, search: str, sector: str, insider: str, pb_max: float) -> pd.DataFrame:
    if search:
        mask = (
            df["Ticker"].str.contains(search, case=False, na=False)
            | df["Company"].str.contains(search, case=False, na=False)
        )
        df = df[mask]
    if sector and sector != "All":
        df = df[df["Sector"] == sector]
    if insider and insider != "All":
        df = df[df["Insider Signal"] == insider]
    if pb_max < 2.0 and "P/B Ratio" in df.columns:
        df = df[df["P/B Ratio"].apply(_to_float) <= pb_max]
    return df


def _kpi_buying_pct(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "—"
    n = len(df)
    buying = (df["Insider Signal"] == "BUYING").sum()
    return f"{buying / n * 100:.0f}%"


def _kpi_avg_pb(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "—"
    vals = df["P/B Ratio"].apply(_to_float).dropna()
    return f"{vals.mean():.2f}x" if len(vals) else "—"


def _kpi_with_div(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "—"
    has_div = df["Div Yield"].apply(lambda x: _to_float(x) > 0).sum()
    return f"{has_div / len(df) * 100:.0f}%"


def _render_table(df: pd.DataFrame, col_order: list[str]) -> None:
    cols = [c for c in col_order if c in df.columns]
    styled = df[cols].style.map(_style_insider, subset=["Insider Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=460)


def _render_charts(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    col1, col2 = st.columns(2)
    with col1:
        sector_data = df["Sector"].value_counts().reset_index()
        sector_data.columns = ["Sector", "Count"]
        fig = px.pie(
            sector_data, values="Count", names="Sector",
            title="Sector Breakdown", hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_dark",
        )
        fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=300, showlegend=True,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        pb_vals = df["P/B Ratio"].apply(_to_float).dropna()
        if len(pb_vals):
            fig = px.histogram(
                pb_vals, title="P/B Ratio Distribution",
                nbins=15, color_discrete_sequence=["#4da6ff"],
                labels={"value": "P/B Ratio", "count": "# Stocks"},
                template="plotly_dark",
            )
            fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=300, showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


def _render_highlights(df: pd.DataFrame) -> None:
    """Top-5 cards for insider buying, highest dividend, lowest P/B."""
    if df is None or df.empty:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🟢 Insider Buying**")
        buying = df[df["Insider Signal"] == "BUYING"][["Ticker", "Company", "Insider Net Value (6mo)", "Sector"]].head(5)
        if buying.empty:
            st.caption("None this run")
        else:
            st.dataframe(buying, hide_index=True, use_container_width=True,
                         height=40 + 35 * len(buying))

    with col2:
        st.markdown("**💰 Highest Dividend Yield**")
        div_df = df.copy()
        div_df["_div"] = div_df["Div Yield"].apply(_to_float)
        top_div = div_df[div_df["_div"] > 0].nlargest(5, "_div")[["Ticker", "Company", "Div Yield", "Sector"]]
        if top_div.empty:
            st.caption("None with dividends")
        else:
            st.dataframe(top_div, hide_index=True, use_container_width=True,
                         height=40 + 35 * len(top_div))

    with col3:
        st.markdown("**📉 Lowest P/B Ratio**")
        pb_df = df.copy()
        pb_df["_pb"] = pb_df["P/B Ratio"].apply(_to_float)
        top_pb = pb_df[pb_df["_pb"] > 0].nsmallest(5, "_pb")[["Ticker", "Company", "P/B Ratio", "Sector"]]
        if top_pb.empty:
            st.caption("No P/B data")
        else:
            st.dataframe(top_pb, hide_index=True, use_container_width=True,
                         height=40 + 35 * len(top_pb))


def _trigger_workflow(force_reports: bool) -> None:
    try:
        token = st.secrets["GH_TOKEN"]
        owner = st.secrets["GITHUB_OWNER"]
        repo  = st.secrets.get("GITHUB_REPO", "stock-screener")
    except KeyError as exc:
        st.error(f"Missing secret: {exc}. Add it in Settings → Secrets.")
        return
    resp = requests.post(
        f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/daily_screener.yml/dispatches",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"},
        json={"ref": "master", "inputs": {"force_reports": str(force_reports).lower()}},
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
        selected_date = None
        prev_date = None
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

        if prev_date:
            p30q  = load_csv(prev_date, "below30Q")
            p50q  = load_csv(prev_date, "below50Q")
            p100q = load_csv(prev_date, "below100Q")
            pmom  = load_csv(prev_date, "momentum")
        else:
            p30q = p50q = p100q = pmom = None

        st.divider()
        st.subheader("Counts")
        def _cnt(df): return len(df) if df is not None else 0
        def _delta(cur, prev): return (_cnt(cur) - _cnt(prev)) if prev is not None else None

        st.metric("Below 30Q SMA",  _cnt(df_30q),  _delta(df_30q,  p30q))
        st.metric("Below 50Q SMA",  _cnt(df_50q),  _delta(df_50q,  p50q))
        st.metric("Below 100Q SMA", _cnt(df_100q), _delta(df_100q, p100q))
        st.metric("Momentum",       _cnt(df_mom),  _delta(df_mom,  pmom))

    st.divider()
    st.subheader("Run Screener")
    force = st.toggle("Force research reports", value=False)
    if st.button("Run Now", type="primary", use_container_width=True):
        _trigger_workflow(force)
    st.caption("Needs GH_TOKEN + GITHUB_OWNER in Streamlit secrets.")

# ── Guard ──────────────────────────────────────────────────────────────────────

if not selected_date:
    st.info("No data found. Trigger a run or wait for the daily 7 AM ET job.")
    st.stop()

# ── Top KPI bar ────────────────────────────────────────────────────────────────

all_dfs = [df for df in [df_30q, df_50q, df_100q, df_mom] if df is not None]
combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
unique_tickers = combined["Ticker"].nunique() if not combined.empty else 0

total_value = sum(_cnt(d) for d in [df_30q, df_50q, df_100q])
buying_pct  = _kpi_buying_pct(combined)
avg_pb      = _kpi_avg_pb(combined)
with_div    = _kpi_with_div(combined)
n_sectors   = combined["Sector"].nunique() if not combined.empty else 0

fmt_date = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%B %d, %Y")
st.markdown(f"### Results — {fmt_date}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Unique Tickers",    f"{unique_tickers:,}")
k2.metric("Value Screen Hits", f"{total_value:,}")
k3.metric("Momentum Hits",     f"{_cnt(df_mom):,}")
k4.metric("Insider Buying",    buying_pct)
k5.metric("Avg P/B Ratio",     avg_pb)

st.divider()

# ── Global filters ─────────────────────────────────────────────────────────────

sector_opts = ["All"] + sorted(
    s for s in combined["Sector"].dropna().unique() if s and s != "—"
) if not combined.empty else ["All"]

fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
search_q       = fc1.text_input("🔍 Search", placeholder="Ticker or company", label_visibility="collapsed")
sector_filter  = fc2.selectbox("Sector", sector_opts, label_visibility="collapsed")
insider_filter = fc3.selectbox("Insider", ["All", "BUYING", "NEUTRAL", "SELLING", "—"], label_visibility="collapsed")
pb_max         = fc4.slider("Max P/B", min_value=0.0, max_value=2.0, value=2.0, step=0.1)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_overview, tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🔵 Below 30Q", "🟠 Below 50Q", "🔴 Below 100Q", "🚀 Momentum"
])

# ── Overview ───────────────────────────────────────────────────────────────────

with tab_overview:
    st.subheader("Cross-Screen Summary")

    # Bar chart: count per screen
    bar_data = pd.DataFrame({
        "Screen":  ["Below 30Q", "Below 50Q", "Below 100Q", "Momentum"],
        "Count":   [_cnt(df_30q), _cnt(df_50q), _cnt(df_100q), _cnt(df_mom)],
        "Color":   ["#0d6efd",   "#fd7e14",    "#dc3545",     "#198754"],
    })
    fig = px.bar(
        bar_data, x="Screen", y="Count", color="Screen",
        color_discrete_map=dict(zip(bar_data["Screen"], bar_data["Color"])),
        title="Stocks per Screen", text="Count",
        template="plotly_dark",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highlights Across All Screens")
    _render_highlights(combined)

    # Sector heatmap across screens
    st.subheader("Sector Exposure")
    if not combined.empty:
        sector_screen = []
        labels = {"below30Q": "Below 30Q", "below50Q": "Below 50Q",
                  "below100Q": "Below 100Q", "momentum": "Momentum"}
        for key, df in [("below30Q", df_30q), ("below50Q", df_50q),
                         ("below100Q", df_100q), ("momentum", df_mom)]:
            if df is not None:
                counts = df["Sector"].value_counts().reset_index()
                counts.columns = ["Sector", "Count"]
                counts["Screen"] = labels[key]
                sector_screen.append(counts)
        if sector_screen:
            ss_df = pd.concat(sector_screen, ignore_index=True)
            pivot = ss_df.pivot_table(index="Sector", columns="Screen", values="Count", fill_value=0)
            fig = px.imshow(
                pivot, text_auto=True, aspect="auto",
                color_continuous_scale="Blues",
                title="Stock count by Sector × Screen",
                template="plotly_dark",
            )
            fig.update_layout(height=400, margin=dict(t=40, b=0),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

# ── Value screen tab helper ────────────────────────────────────────────────────

def _value_tab(df: pd.DataFrame | None, label: str, sma_col: str, color: str) -> None:
    if df is None or df.empty:
        st.info(f"No {label} data for {selected_date}.")
        return

    filtered = _apply_filters(df, search_q, sector_filter, insider_filter, pb_max)

    # Mini KPIs
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks shown",    f"{len(filtered):,}", f"{len(filtered) - len(df):+,}" if len(filtered) != len(df) else None)
    m2.metric("Insider Buying",  _kpi_buying_pct(filtered))
    m3.metric("Avg P/B",         _kpi_avg_pb(filtered))
    m4.metric("With Dividend",   _kpi_with_div(filtered))

    cols = [sma_col if c == "% Below SMA" else c for c in VALUE_COLS]
    _render_table(filtered, cols)
    _render_charts(filtered)

    st.subheader("Highlights")
    _render_highlights(filtered)


with tab1:
    st.markdown("Price below its **30-quarter (7.5 yr) SMA** — trading below long-run average.")
    _value_tab(df_30q, "Below 30Q", "30Q SMA", "#0d6efd")

with tab2:
    st.markdown("Price below its **50-quarter (12.5 yr) SMA** — moderate undervaluation.")
    _value_tab(df_50q, "Below 50Q", "50Q SMA", "#fd7e14")

with tab3:
    st.markdown("Price below its **100-quarter (25 yr) SMA** — deep undervaluation vs 25-year history.")
    _value_tab(df_100q, "Below 100Q", "100Q SMA", "#dc3545")

# ── Momentum tab ───────────────────────────────────────────────────────────────

with tab4:
    st.markdown("**Bull-aligned quarterly SMAs**, trading near 20Q support with insider buying or neutral.")

    if df_mom is None or df_mom.empty:
        st.info(f"No momentum data for {selected_date}.")
    else:
        filtered_mom = _apply_filters(df_mom, search_q, sector_filter, insider_filter, pb_max)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stocks shown",  f"{len(filtered_mom):,}")
        m2.metric("Insider Buying", _kpi_buying_pct(filtered_mom))
        m3.metric("Avg P/B",       _kpi_avg_pb(filtered_mom))
        m4.metric("With Dividend", _kpi_with_div(filtered_mom))

        _render_table(filtered_mom, MOM_COLS)
        _render_charts(filtered_mom)

        st.subheader("Highlights")
        _render_highlights(filtered_mom)

        # Research reports
        reports = load_reports(selected_date)
        if reports:
            st.divider()
            st.subheader(f"📄 Research Reports ({len(reports)})")
            st.caption("Claude-generated investment memos for new momentum names.")
            for ticker, text in reports.items():
                with st.expander(f"{ticker}"):
                    st.text(text)
        else:
            st.caption("No research reports for this date — generated on Mondays or when forced.")
