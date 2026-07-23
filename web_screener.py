#!/usr/bin/env python3
"""
IDX Momentum + Quality Screener v2 — Web Dashboard (Streamlit)

Run:
    ./.venv/bin/streamlit run web_screener.py
"""

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from screener import (
    TICKERS_FILE,
    analyze_ticker,
    classify_pattern,
    load_tickers,
    volatility_label,
)
from portfolio import (
    BASELINE_INVESTED,
    check_alerts,
    compute_days_held,
    compute_portfolio,
    fetch_prices_batch,
    flag_data_quality,
    has_data_quality_issue,
    load_cash_state,
    load_pending,
    load_positions,
    save_cash_state,
    summarize,
    time_exit_signal,
)
from history import (
    add_to_watchlist,
    get_days_tracked,
    get_history,
    get_latest_and_previous,
    get_score_changes,
    load_watchlist,
    remove_from_watchlist,
    save_watchlist,
    snapshot_scores,
)
from backtest import (
    benchmark_ihsg,
    run_backtest,
    summarize_backtest,
)
from news import (
    detect_corporate_action,
    enrich_news,
    fetch_news,
    format_published,
    has_high_impact_news,
)
from financials import (
    format_large,
    full_financial_summary,
    interpret_financials,
)
from valuation import full_valuation
from us_portfolio import (
    compute_us_portfolio,
    fetch_us_price,
    fetch_usd_idr_rate,
    load_us_positions,
    summarize_us,
)
from us_screener import (
    load_us_tickers,
    screen_us_all,
)

st.set_page_config(
    page_title="IDX Momentum + Quality Screener v2",
    page_icon="📊",
    layout="wide",
)


def _check_auth():
    """Password gate. Enabled when APP_PASSWORD env var is set."""
    required = os.getenv("APP_PASSWORD")
    if not required:
        return
    if st.session_state.get("auth_ok"):
        return
    st.title("🔒 Login")
    pw = st.text_input("Password", type="password", key="pw_input")
    if st.button("Enter") or pw:
        if pw == required:
            st.session_state.auth_ok = True
            st.rerun()
        elif pw:
            st.error("Wrong password")
    st.stop()


_check_auth()


def _safe_pct(val):
    """Normalize dividend yield / ROE which yfinance sometimes returns as fraction, sometimes as pct."""
    if val is None:
        return None
    if abs(val) < 1:
        return val * 100
    return val


def _cap_display(val, cap=1000):
    """Cap unreasonable values (yfinance IDX data glitches)."""
    if val is None or val > cap:
        return None
    return val


@st.cache_data(ttl=1800, show_spinner=False)  # 30 min cache (fundamentals fetch slow)
def fetch_all_v2(tickers_tuple):
    """Fetch + compute all metrics (technical + fundamental) for all tickers."""
    results = []
    progress = st.progress(0.0, text="Fetching data (may take 1-2 min for fundamentals)...")
    total = len(tickers_tuple)

    for i, (ticker, category) in enumerate(tickers_tuple):
        progress.progress((i + 1) / total, text=f"[{i+1}/{total}] {ticker}...")
        r = analyze_ticker(ticker, category)
        if r is None:
            continue
        m = r["metrics"]
        results.append({
            "Ticker": ticker,
            "Category": category,
            "Price": m["current"],
            "1Y%": m["return_1y"],
            "3M%": m["return_3m"],
            "1M%": m["return_1m"],
            "1W%": m["return_1w"],
            "vsMA50%": m["price_vs_ma50"],
            "HL": m["higher_lows"],
            "MTF": m["mtf_aligned"],
            "Divergence": m.get("divergence"),
            "RSI": m["rsi"],
            "ATR%": m["atr_pct"],
            "PER": _cap_display(m.get("pe"), cap=500),
            "PBV": _cap_display(m.get("pbv"), cap=100),
            "ROE%": _safe_pct(m.get("roe")),
            "DivY%": _safe_pct(m.get("div_yield")),
            "D/E": _cap_display(m.get("de_ratio"), cap=1000),
            "Momentum": r["momentum"],
            "Quality": r["quality"] if r["quality"] is not None else None,
            "Score": r["overall"],
            "Verdict": r["verdict"],
            "Vol": volatility_label(m["atr_pct"]),
        })

    progress.empty()
    return pd.DataFrame(results)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_history(ticker: str, period="1y", interval="1d"):
    yf_ticker = f"{ticker}.JK"
    return yf.Ticker(yf_ticker).history(period=period, interval=interval, auto_adjust=False)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_financials_global(ticker: str):
    return full_financial_summary(ticker, market="IDX")


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_valuation_global(ticker: str, target_pe: float = 15, target_pbv: float = 2, discount: float = 0.10):
    return full_valuation(ticker, market="IDX",
                          custom_target_pe=target_pe,
                          custom_target_pbv=target_pbv,
                          custom_discount=discount)


def color_score(val):
    if val is None or pd.isna(val): return ""
    if val >= 75: return "background-color: #16a34a; color: white; font-weight: bold"
    if val >= 60: return "background-color: #22c55e; color: white"
    if val >= 45: return "background-color: #eab308; color: black"
    if val >= 30: return "background-color: #f97316; color: white"
    return "background-color: #dc2626; color: white"


def render_chart(df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#16a34a",
        decreasing_line_color="#dc2626",
    ))
    if len(df) >= 20:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(), name="MA20", line=dict(color="orange", width=1)))
    if len(df) >= 50:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(50).mean(), name="MA50", line=dict(color="blue", width=1)))
    if len(df) >= 200:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(200).mean(), name="MA200", line=dict(color="purple", width=1)))

    fig.update_layout(
        title=f"{ticker}.JK — 1 Year",
        yaxis_title="Price (IDR)",
        xaxis_rangeslider_visible=False,
        height=500,
        template="plotly_white",
    )
    return fig


# =========================
# UI
# =========================

st.title("📊 IDX Momentum + Quality Screener v2")
st.caption("Data-driven analysis with technical + fundamental scoring. Powered by Yahoo Finance.")

with st.sidebar:
    st.header("🎛️ Filters")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    min_score = st.slider("Min Overall Score", 0, 100, 0, step=5)
    min_momentum = st.slider("Min Momentum Score", 0, 100, 0, step=5)
    min_quality = st.slider("Min Quality Score", 0, 100, 0, step=5)

    verdict_filter = st.multiselect(
        "Verdict",
        options=["🟢 STRONG BUY", "🟢 BUY", "🟡 WATCH", "🟠 WEAK", "🔴 AVOID"],
        default=[],
    )

    tickers = load_tickers(TICKERS_FILE)
    all_categories = sorted(set(c for _, c in tickers))
    category_filter = st.multiselect("Category", options=all_categories, default=[])

    volatility_filter = st.multiselect(
        "Volatility",
        options=["🧊 LOW", "⚡ MED", "🔥 HIGH"],
        default=[],
    )

    require_hl = st.checkbox("Higher Lows only")
    require_mtf = st.checkbox("Multi-timeframe aligned only")
    exclude_bearish_div = st.checkbox("Exclude bearish divergence")

    sort_by = st.selectbox("Sort by", options=["Score", "Momentum", "Quality", "1Y%", "3M%", "1M%", "DivY%"], index=0)
    top_n = st.number_input("Show Top N", 5, 100, 20, step=5)

    st.divider()
    st.caption(f"Total tickers: {len(tickers)}")
    st.caption("Cache: 30 min. Refresh for latest.")

tab_portfolio, tab1, tab_us, tab_watchlist, tab2, tab3, tab_backtest, tab4 = st.tabs(["💼 Portfolio", "📈 Screener IDX", "🇺🇸 Screener US", "📌 Watchlist", "🔍 Detail", "🏭 Sector Strength", "🧪 Backtest", "ℹ️ Methodology"])

# ======= PORTFOLIO MONITOR =======
with tab_portfolio:
    st.subheader("💼 Portfolio Monitor")

    positions_df = load_positions()
    pending_df = load_pending()

    if positions_df.empty:
        st.info("No positions in `02-position-tracker.csv`. Add positions to start monitoring.")
    else:
        # Fetch live prices
        with st.spinner("Fetching latest prices..."):
            tickers_list = positions_df["Ticker"].tolist()
            prices = fetch_prices_batch(tickers_list)

        # Compute P/L + days held
        port_df = compute_portfolio(positions_df, prices)
        port_df = compute_days_held(port_df)
        stats = summarize(port_df)

        # Load cash state
        cash = load_cash_state()
        trading_balance = cash.get("trading_balance", 0)
        reserved = cash.get("reserved_pending", 0)
        total_cash = trading_balance + reserved

        # Total equity = stocks market value + cash
        total_equity = stats['total_value'] + total_cash

        # Header metrics — row 1: portfolio value
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Total Equity", f"Rp {total_equity:,.0f}",
                   delta=f"Cash Rp {total_cash:,.0f}")
        col2.metric("📊 Market Value (Stocks)", f"Rp {stats['total_value']:,.0f}")
        col3.metric("💵 Cash Available", f"Rp {trading_balance:,.0f}",
                   delta=f"Reserved Rp {reserved:,.0f}" if reserved > 0 else None,
                   delta_color="off")
        col4.metric("📉 Realized Loss (This Session)", "Rp -2,780,580",
                   delta="IKAN sell", delta_color="inverse")

        st.divider()

        # Header metrics — row 2: recovery tracking
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Baseline (Original Invested)", f"Rp {BASELINE_INVESTED:,.0f}")

        recovery_pct = (total_equity / BASELINE_INVESTED) * 100
        col2.metric("📈 Recovery vs Baseline", f"{recovery_pct:.1f}%",
                   delta=f"{recovery_pct - 100:+.1f}% vs break-even")

        gap = BASELINE_INVESTED - total_equity
        col3.metric("🎳 Gap to Break-Even",
                   f"Rp {gap:,.0f}" if gap > 0 else "🎉 Break-even reached",
                   delta=f"Need +{gap/total_equity*100:.1f}%" if gap > 0 else None,
                   delta_color="off")

        col4.metric("📉 Total Loss vs Baseline",
                   f"Rp {total_equity - BASELINE_INVESTED:+,.0f}",
                   delta=f"{recovery_pct - 100:+.2f}%")

        # Recovery progress bar
        st.markdown("**Recovery Progress:**")
        st.progress(min(1.0, total_equity / BASELINE_INVESTED))

        # Editable cash state
        with st.expander("💰 Update Cash State"):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                new_balance = st.number_input("Trading Balance (Cash)", value=float(trading_balance), step=100000.0)
            with col2:
                new_reserved = st.number_input("Reserved (Pending Orders)", value=float(reserved), step=100000.0)
            with col3:
                if st.button("💾 Save"):
                    save_cash_state(new_balance, new_reserved)
                    st.success("Saved! Refresh to apply.")

        st.divider()

        # Alerts
        alerts = check_alerts(port_df)
        if alerts:
            st.subheader("🔔 Alerts")
            for a in alerts:
                if "CRITICAL" in a["level"]:
                    st.error(f"{a['level']} — **{a['ticker']}**: {a['msg']}")
                elif "WARNING" in a["level"]:
                    st.warning(f"{a['level']} — **{a['ticker']}**: {a['msg']}")
                else:
                    st.success(f"{a['level']} — **{a['ticker']}**: {a['msg']}")
            st.divider()

        # Layer breakdown
        st.subheader("📊 Layer Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧊 Layer 1 — Frozen Legacy")
            st.metric("Value", f"Rp {stats['layer1_value']:,.0f}",
                     delta=f"{(stats['layer1_pl']/stats['layer1_invested']*100 if stats['layer1_invested'] else 0):+.1f}%")
            st.caption(f"Invested: Rp {stats['layer1_invested']:,.0f} | P/L: Rp {stats['layer1_pl']:+,.0f}")
        with col2:
            st.markdown("### 🎯 Layer 2 — Active Trading")
            if stats['layer2_invested'] > 0:
                st.metric("Value", f"Rp {stats['layer2_value']:,.0f}",
                         delta=f"{(stats['layer2_pl']/stats['layer2_invested']*100):+.1f}%")
                st.caption(f"Invested: Rp {stats['layer2_invested']:,.0f} | P/L: Rp {stats['layer2_pl']:+,.0f}")
            else:
                st.info("No Layer 2 positions yet. Waiting for MAPI + ADRO orders to fill.")

        st.divider()

        # Color function for P/L values
        def color_pl(val):
            try:
                if pd.isna(val):
                    return ""
            except (TypeError, ValueError):
                pass
            if isinstance(val, (int, float)):
                if val > 0:
                    return "color: #22c55e; font-weight: bold"
                if val < 0:
                    return "color: #ef4444; font-weight: bold"
            return ""

        # Layer 1 table with color styling
        st.subheader("🧊 Layer 1 — Frozen Legacy Positions")
        l1_df = port_df[port_df["Layer"] == "Layer 1"].copy()
        if not l1_df.empty:
            l1_display = l1_df[["Ticker", "Lot", "Avg Price", "Current Price", "Cost Basis", "Market Value", "P/L Rp", "P/L %", "Stop Loss", "Take Profit 1", "Notes"]].copy()
            l1_styled = (
                l1_display.style
                .format({
                    "Avg Price": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "Current Price": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "Cost Basis": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "Market Value": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "P/L Rp": lambda x: f"{x:+,.0f}" if pd.notna(x) else "-",
                    "P/L %": lambda x: f"{x:+.1f}%" if pd.notna(x) else "-",
                    "Stop Loss": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "Take Profit 1": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                })
                .map(color_pl, subset=["P/L Rp", "P/L %"])
            )
            st.dataframe(l1_styled, use_container_width=True, hide_index=True)

        # Layer 2 table with color styling + time exit signal
        l2_df = port_df[port_df["Layer"] == "Layer 2"].copy()
        if not l2_df.empty:
            st.subheader("🎯 Layer 2 — Active Trading Positions")

            # ===== TIME EXIT ALERTS =====
            st.markdown("**⏰ Time Exit Signals:**")
            for _, r in l2_df.iterrows():
                signal = time_exit_signal(r.get("Days Held"), r.get("Layer"))
                ticker = r["Ticker"]
                msg = f"**{ticker}** — {signal['signal']}"
                action = signal["action"]
                color = signal["color"]
                if color == "green":
                    st.success(f"{msg} | 💡 {action}")
                elif color == "yellow":
                    st.info(f"{msg} | 💡 {action}")
                elif color == "orange":
                    st.warning(f"{msg} | 💡 {action}")
                elif color == "red" or color == "critical":
                    st.error(f"{msg} | 💡 {action}")
                else:
                    st.write(f"{msg} — {action}")

            l2_display = l2_df[["Ticker", "Lot", "Avg Price", "Current Price", "Cost Basis", "Market Value", "P/L Rp", "P/L %", "Stop Loss", "Take Profit 1", "Take Profit 2", "Entry Date", "Days Held"]].copy()
            l2_styled = (
                l2_display.style
                .format({
                    "Avg Price": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "Current Price": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "Cost Basis": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "Market Value": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "P/L Rp": lambda x: f"{x:+,.0f}" if pd.notna(x) else "-",
                    "P/L %": lambda x: f"{x:+.1f}%" if pd.notna(x) else "-",
                    "Stop Loss": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "Take Profit 1": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "Take Profit 2": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "Days Held": lambda x: f"{int(x)}d" if pd.notna(x) else "-",
                })
                .map(color_pl, subset=["P/L Rp", "P/L %"])
            )
            st.dataframe(l2_styled, use_container_width=True, hide_index=True)

        # Pending orders
        if not pending_df.empty:
            st.divider()
            st.subheader("⏳ Pending Orders")
            pending_display = pending_df[["Ticker", "Action", "Limit Price", "Lot", "Amount", "Status", "Expiry", "Layer", "Notes"]].copy()
            for col in ["Limit Price", "Amount"]:
                pending_display[col] = pending_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
            st.dataframe(pending_display, use_container_width=True, hide_index=True)

        st.divider()

        # ======= US Portfolio Section =======
        st.subheader("🇺🇸 US Portfolio (Gotrade)")

        us_positions = load_us_positions()
        if us_positions.empty:
            st.info("No US positions. Add to `us_positions.csv`.")
        else:
            with st.spinner("Fetching US prices..."):
                us_prices = {t: fetch_us_price(t) for t in us_positions["Ticker"]}
                usd_idr_rate = fetch_usd_idr_rate()

            us_port = compute_us_portfolio(us_positions, us_prices, usd_idr_rate)
            us_stats = summarize_us(us_port, usd_idr_rate)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💵 US Portfolio (USD)", f"${us_stats['total_value_usd']:,.2f}",
                       delta=f"{us_stats['total_pct']:+.2f}%")
            col2.metric("💰 US Portfolio (IDR)", f"Rp {us_stats['total_value_idr']:,.0f}")
            # Fix: put sign at start so streamlit auto-colors correctly
            pl_idr_val = us_stats['total_pl_idr']
            pl_idr_str = f"{pl_idr_val:+,.0f} IDR"
            col3.metric("📉 P/L", f"${us_stats['total_pl_usd']:+,.2f}",
                       delta=pl_idr_str)
            col4.metric("💱 USD/IDR Rate", f"Rp {usd_idr_rate:,.0f}",
                       delta="Live rate", delta_color="off")

            # US positions table with color styling
            us_display = us_port[["Ticker", "Name", "Shares", "Avg Cost USD", "Current Price USD",
                                   "Cost Basis USD", "Market Value USD", "P/L USD", "P/L %",
                                   "Market Value IDR"]].copy()

            us_styled = (
                us_display.style
                .format({
                    "Shares": lambda x: f"{x:.4f}",
                    "Avg Cost USD": lambda x: f"${x:.2f}",
                    "Current Price USD": lambda x: f"${x:.2f}" if pd.notna(x) else "-",
                    "Cost Basis USD": lambda x: f"${x:,.2f}",
                    "Market Value USD": lambda x: f"${x:,.2f}" if pd.notna(x) else "-",
                    "P/L USD": lambda x: f"${x:+,.2f}" if pd.notna(x) else "-",
                    "P/L %": lambda x: f"{x:+.2f}%" if pd.notna(x) else "-",
                    "Market Value IDR": lambda x: f"Rp {x:,.0f}" if pd.notna(x) else "-",
                })
                .map(color_pl, subset=["P/L USD", "P/L %"])
            )
            st.dataframe(us_styled, use_container_width=True, hide_index=True)

        st.divider()

        # ======= CONSOLIDATED PORTFOLIO =======
        st.subheader("🌍 Consolidated Portfolio (IDR)")

        # IDX total
        idx_stocks_value = stats['total_value']
        idx_cash = total_cash
        idx_total = idx_stocks_value + idx_cash

        # US total
        us_value_idr = us_stats.get('total_value_idr', 0) if not us_positions.empty else 0

        consolidated_total = idx_total + us_value_idr

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🇮🇩 IDX Total", f"Rp {idx_total:,.0f}",
                   delta=f"{(idx_total/consolidated_total*100):.1f}% of total")
        col2.metric("🇺🇸 US Total", f"Rp {us_value_idr:,.0f}",
                   delta=f"{(us_value_idr/consolidated_total*100):.1f}% of total" if consolidated_total else None)
        col3.metric("💎 Grand Total", f"Rp {consolidated_total:,.0f}")

        # Original baseline includes just IDX. Add US cost as separate line
        us_cost_idr = us_stats.get('total_cost_idr', 0) if not us_positions.empty else 0
        total_baseline_all = BASELINE_INVESTED + us_cost_idr
        recovery_all = (consolidated_total / total_baseline_all) * 100 if total_baseline_all else 0
        col4.metric("📊 Recovery vs All-Time Invested", f"{recovery_all:.1f}%",
                   delta=f"Total invested Rp {total_baseline_all:,.0f}")

        st.caption(f"💡 Consolidated view: IDX stocks + IDX cash + US stocks. USD/IDR rate: Rp {usd_idr_rate:,.0f} live from Yahoo Finance.")


# ======= TAB 1: Screener =======
with tab1:
    with st.spinner("Loading market data..."):
        df = fetch_all_v2(tuple(tickers))

    if df.empty:
        st.error("No data loaded.")
        st.stop()

    # Auto-snapshot to history DB (once per day per ticker)
    if "snapshot_taken" not in st.session_state:
        n = snapshot_scores(df.to_dict("records"))
        st.session_state["snapshot_taken"] = True
        if n > 0:
            st.toast(f"📸 History snapshot: {n} tickers recorded", icon="✅")

    filtered = df.copy()
    if min_score > 0: filtered = filtered[filtered["Score"] >= min_score]
    if min_momentum > 0: filtered = filtered[filtered["Momentum"] >= min_momentum]
    if min_quality > 0: filtered = filtered[(filtered["Quality"].notna()) & (filtered["Quality"] >= min_quality)]
    if verdict_filter: filtered = filtered[filtered["Verdict"].isin(verdict_filter)]
    if category_filter: filtered = filtered[filtered["Category"].isin(category_filter)]
    if volatility_filter: filtered = filtered[filtered["Vol"].isin(volatility_filter)]
    if require_hl: filtered = filtered[filtered["HL"] == True]
    if require_mtf: filtered = filtered[filtered["MTF"] == True]
    if exclude_bearish_div: filtered = filtered[filtered["Divergence"] != "bearish"]

    sort_col = {"Score": "Score", "Momentum": "Momentum", "Quality": "Quality", "1Y%": "1Y%", "3M%": "3M%", "1M%": "1M%", "DivY%": "DivY%"}[sort_by]
    filtered = filtered.sort_values(by=sort_col, ascending=False, na_position="last").head(top_n)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Screened", len(df))
    col2.metric("Matching", len(filtered))
    col3.metric("Avg Score", f"{filtered['Score'].mean():.0f}" if not filtered.empty else "-")
    col4.metric("Avg 1Y", f"{filtered['1Y%'].mean():+.1f}%" if not filtered.empty else "-")

    st.divider()

    display = filtered.copy()
    display["HL"] = display["HL"].map({True: "✓", False: "✗"})
    display["MTF"] = display["MTF"].map({True: "✓", False: "✗"})
    display["Divergence"] = display["Divergence"].map(lambda x: {"bearish": "⬇ Bear", "bullish": "⬆ Bull"}.get(x, "-"))

    # Add data quality flag column
    def _has_dq_issue(row_dict):
        return "⚠️" if has_data_quality_issue(row_dict) else ""
    display["DQ"] = filtered.apply(lambda row: _has_dq_issue(row.to_dict()), axis=1).values

    # Color function for returns
    def color_return(val):
        try:
            if pd.isna(val):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(val, (int, float)):
            if val > 0: return "color: #22c55e; font-weight: bold"
            if val < 0: return "color: #ef4444; font-weight: bold"
        return ""

    pct_cols = ["1Y%", "3M%", "1M%", "1W%", "vsMA50%"]
    styled = (
        display.style
        .format({
            "Price": lambda x: f"{x:,.0f}",
            "1Y%": lambda x: f"{x:+.1f}" if pd.notna(x) else "-",
            "3M%": lambda x: f"{x:+.1f}" if pd.notna(x) else "-",
            "1M%": lambda x: f"{x:+.1f}" if pd.notna(x) else "-",
            "1W%": lambda x: f"{x:+.1f}" if pd.notna(x) else "-",
            "vsMA50%": lambda x: f"{x:+.1f}" if pd.notna(x) else "-",
            "RSI": lambda x: f"{x:.0f}",
            "ATR%": lambda x: f"{x:.1f}",
            "PER": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
            "PBV": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
            "D/E": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
            "ROE%": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
            "DivY%": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
            "Quality": lambda x: f"{int(x)}" if pd.notna(x) else "N/A",
        })
        .map(color_score, subset=["Score", "Momentum", "Quality"])
        .map(color_return, subset=pct_cols)
        .set_properties(**{"text-align": "right"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Data quality warnings expander
    dq_flagged = filtered[filtered.apply(lambda row: has_data_quality_issue(row.to_dict()), axis=1)]
    if not dq_flagged.empty:
        with st.expander(f"⚠️ Data Quality Warnings ({len(dq_flagged)} tickers flagged)"):
            for _, row in dq_flagged.iterrows():
                warns = flag_data_quality(row.to_dict())
                st.markdown(f"**{row['Ticker']}** ({row['Category']}):")
                for w in warns:
                    st.caption(f"  • {w}")

    st.download_button(
        "📥 Export CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="screener_v2_result.csv",
        mime="text/csv",
    )

# ======= US SCREENER TAB =======
with tab_us:
    st.subheader("🇺🇸 US Market Momentum Screener")
    st.caption("Data-driven screening for NYSE/NASDAQ (mega caps + growth + ETFs). Powered by Yahoo Finance.")

    @st.cache_data(ttl=1800, show_spinner=False)
    def fetch_us_all(tickers_tuple):
        results = []
        progress = st.progress(0.0, text="Fetching US market data...")
        total = len(tickers_tuple)

        from us_screener import analyze_us_ticker
        for i, (ticker, category) in enumerate(tickers_tuple):
            progress.progress((i + 1) / total, text=f"[{i+1}/{total}] {ticker}...")
            r = analyze_us_ticker(ticker, category)
            if r is None:
                continue
            m = r["metrics"]
            results.append({
                "Ticker": ticker,
                "Category": category,
                "Price": m["current"],
                "1Y%": m["return_1y"],
                "3M%": m["return_3m"],
                "1M%": m["return_1m"],
                "1W%": m["return_1w"],
                "vsMA50%": m["price_vs_ma50"],
                "HL": m["higher_lows"],
                "MTF": m["mtf_aligned"],
                "RSI": m["rsi"],
                "ATR%": m["atr_pct"],
                "PER": _cap_display(m.get("pe"), cap=500),
                "PBV": _cap_display(m.get("pbv"), cap=100),
                "ROE%": _safe_pct(m.get("roe")),
                "DivY%": _safe_pct(m.get("div_yield")),
                "Momentum": r["momentum"],
                "Quality": r["quality"] if r["quality"] is not None else None,
                "Score": r["overall"],
                "Verdict": r["verdict"],
                "Vol": volatility_label(m["atr_pct"]),
            })
        progress.empty()
        return pd.DataFrame(results)

    us_tickers = load_us_tickers()

    with st.sidebar:
        st.divider()
        st.markdown("**🇺🇸 US Screener Filters**")
        us_min_score = st.slider("US Min Score", 0, 100, 0, step=5, key="us_min_score")
        us_verdict = st.multiselect("US Verdict",
            options=["🟢 STRONG BUY", "🟢 BUY", "🟡 WATCH", "🟠 WEAK", "🔴 AVOID"],
            default=[], key="us_verdict_filter")
        us_categories = sorted(set(c for _, c in us_tickers))
        us_cat_filter = st.multiselect("US Category", options=us_categories, default=[], key="us_cat_filter")
        us_top_n = st.number_input("US Show Top N", 5, 60, 20, step=5, key="us_top")

    with st.spinner("Loading US market data..."):
        us_df = fetch_us_all(tuple(us_tickers))

    if us_df.empty:
        st.error("No US data loaded. Check network.")
    else:
        # Filter
        us_filtered = us_df.copy()
        if us_min_score > 0: us_filtered = us_filtered[us_filtered["Score"] >= us_min_score]
        if us_verdict: us_filtered = us_filtered[us_filtered["Verdict"].isin(us_verdict)]
        if us_cat_filter: us_filtered = us_filtered[us_filtered["Category"].isin(us_cat_filter)]

        us_filtered = us_filtered.sort_values("Score", ascending=False).head(us_top_n)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Screened", len(us_df))
        col2.metric("Matching", len(us_filtered))
        col3.metric("Avg Score", f"{us_filtered['Score'].mean():.0f}" if not us_filtered.empty else "-")
        col4.metric("Avg 1Y", f"{us_filtered['1Y%'].mean():+.1f}%" if not us_filtered.empty else "-")

        st.divider()

        # Format display with color styling
        us_display = us_filtered.copy()
        us_display["HL"] = us_display["HL"].map({True: "✓", False: "✗"})
        us_display["MTF"] = us_display["MTF"].map({True: "✓", False: "✗"})

        def color_us_return(val):
            try:
                if pd.isna(val): return ""
            except (TypeError, ValueError):
                pass
            if isinstance(val, (int, float)):
                if val > 0: return "color: #22c55e; font-weight: bold"
                if val < 0: return "color: #ef4444; font-weight: bold"
            return ""

        us_pct_cols = ["1Y%", "3M%", "1M%", "1W%", "vsMA50%"]
        us_styled = (
            us_display.style
            .format({
                "Price": lambda x: f"${x:,.2f}",
                "1Y%": lambda x: f"{x:+.1f}" if pd.notna(x) else "-",
                "3M%": lambda x: f"{x:+.1f}" if pd.notna(x) else "-",
                "1M%": lambda x: f"{x:+.1f}" if pd.notna(x) else "-",
                "1W%": lambda x: f"{x:+.1f}" if pd.notna(x) else "-",
                "vsMA50%": lambda x: f"{x:+.1f}" if pd.notna(x) else "-",
                "RSI": lambda x: f"{x:.0f}",
                "ATR%": lambda x: f"{x:.1f}",
                "PER": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
                "PBV": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
                "ROE%": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
                "DivY%": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
                "Quality": lambda x: f"{int(x)}" if pd.notna(x) else "N/A",
            })
            .map(color_score, subset=["Score", "Momentum", "Quality"])
            .map(color_us_return, subset=us_pct_cols)
            .set_properties(**{"text-align": "right"})
        )
        st.dataframe(us_styled, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Export US CSV",
            data=us_filtered.to_csv(index=False).encode("utf-8"),
            file_name="us_screener_result.csv",
            mime="text/csv",
        )

        # Alert: recommend for AVGO (mas dewo's holding)
        st.divider()
        st.subheader("🎯 Your US Holdings — Current Signal")

        holdings_check = us_df[us_df["Ticker"].isin(["AVGO", "SLV"])]
        if not holdings_check.empty:
            for _, row in holdings_check.iterrows():
                score = row["Score"]
                # Match verdict thresholds exactly
                if score >= 75: color = "🟢"     # STRONG BUY
                elif score >= 60: color = "🟢"   # BUY
                elif score >= 45: color = "🟡"   # WATCH
                elif score >= 30: color = "🟠"   # WEAK
                else: color = "🔴"               # AVOID
                st.markdown(f"{color} **{row['Ticker']}** ({row['Category']}) — Score: {score} | Verdict: {row['Verdict']}")
                st.caption(f"1Y: {row['1Y%']:+.1f}% | 1M: {row['1M%']:+.1f}% | RSI: {row['RSI']:.0f} | Volatility: {row['Vol']}")

# ======= WATCHLIST TAB =======
with tab_watchlist:
    st.subheader("📌 Watchlist")

    watchlist = load_watchlist()

    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.metric("Watched Tickers", len(watchlist))
        st.metric("History Days Tracked", get_days_tracked())

    with col_b:
        st.caption("Track saham favorit + monitor perubahan score dari waktu ke waktu.")
        with st.expander("➕ Manage Watchlist"):
            add_ticker = st.text_input("Add ticker (uppercase):", "").strip().upper()
            add_col1, add_col2 = st.columns(2)
            with add_col1:
                if st.button("➕ Add") and add_ticker:
                    if add_to_watchlist(add_ticker):
                        st.success(f"Added {add_ticker}")
                        st.rerun()
                    else:
                        st.warning(f"{add_ticker} already in list")

            remove_ticker = st.selectbox("Remove ticker:", [""] + watchlist)
            with add_col2:
                if st.button("🗑️ Remove") and remove_ticker:
                    remove_from_watchlist(remove_ticker)
                    st.success(f"Removed {remove_ticker}")
                    st.rerun()

    st.divider()

    if not watchlist:
        st.info("Watchlist kosong. Tambah ticker via expander atau edit `watchlist.txt`.")
    else:
        # Show current + previous score comparison
        rows = []
        for ticker in watchlist:
            today, prev = get_latest_and_previous(ticker)
            row_data = {"Ticker": ticker}

            # Get current data from screener df if available
            in_screener = df[df["Ticker"] == ticker]
            if not in_screener.empty:
                sr = in_screener.iloc[0]
                row_data["Category"] = sr["Category"]
                row_data["Price"] = sr["Price"]
                row_data["Score Now"] = sr["Score"]
                row_data["Verdict"] = sr["Verdict"]
                row_data["1M%"] = sr["1M%"]
                row_data["1W%"] = sr["1W%"]
                row_data["RSI"] = sr["RSI"]
            else:
                row_data["Category"] = "-"
                row_data["Price"] = None
                row_data["Score Now"] = None
                row_data["Verdict"] = "N/A"

            if prev:
                row_data["Score Prev"] = prev[1]
                row_data["Δ Score"] = (row_data["Score Now"] or 0) - (prev[1] or 0)
                row_data["Prev Date"] = prev[0]
                row_data["Prev Price"] = prev[4]
                if row_data["Price"] and prev[4]:
                    row_data["Δ Price %"] = (row_data["Price"] - prev[4]) / prev[4] * 100
                else:
                    row_data["Δ Price %"] = None
            else:
                row_data["Score Prev"] = None
                row_data["Δ Score"] = None
                row_data["Prev Date"] = "-"
                row_data["Prev Price"] = None
                row_data["Δ Price %"] = None

            rows.append(row_data)

        wl_df = pd.DataFrame(rows)

        # Trend arrow
        def trend_arrow(delta):
            if pd.isna(delta) or delta is None: return "-"
            if delta > 5: return "🟢 ↗ +" + f"{delta:.0f}"
            if delta > 0: return "🟢 →↗ +" + f"{delta:.0f}"
            if delta == 0: return "⚪ →"
            if delta > -5: return "🟠 →↘ " + f"{delta:.0f}"
            return "🔴 ↘ " + f"{delta:.0f}"

        wl_display = wl_df.copy()
        wl_display["Δ Score Trend"] = wl_display["Δ Score"].apply(trend_arrow)

        cols = ["Ticker", "Category", "Verdict", "Score Now", "Score Prev", "Δ Score", "Δ Score Trend",
                "Price", "Prev Price", "Δ Price %", "Prev Date"]
        wl_display = wl_display[cols]

        def color_delta(val):
            try:
                if pd.isna(val):
                    return ""
            except (TypeError, ValueError):
                pass
            if isinstance(val, (int, float)):
                if val > 0: return "color: #22c55e; font-weight: bold"
                if val < 0: return "color: #ef4444; font-weight: bold"
            return ""

        wl_styled = (
            wl_display.style
            .format({
                "Price": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                "Prev Price": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                "Δ Price %": lambda x: f"{x:+.1f}%" if pd.notna(x) else "-",
                "Δ Score": lambda x: f"{x:+.0f}" if pd.notna(x) else "-",
            })
            .map(color_delta, subset=["Δ Score", "Δ Price %"])
        )
        st.dataframe(wl_styled, use_container_width=True, hide_index=True)

    # Score changes across ALL tickers (not just watchlist)
    st.divider()
    st.subheader("🚨 Big Score Changes (All Tickers)")
    changes = get_score_changes(min_change=5)
    if changes.empty:
        st.info("Belum ada data histori cukup (butuh minimal 2 hari snapshot). Lanjutkan pakai dashboard hari-hari berikutnya.")
    else:
        def color_delta_local(val):
            try:
                if pd.isna(val):
                    return ""
            except (TypeError, ValueError):
                pass
            if isinstance(val, (int, float)):
                if val > 0: return "color: #22c55e; font-weight: bold"
                if val < 0: return "color: #ef4444; font-weight: bold"
            return ""

        changes_display = changes[["ticker", "score_now", "score_prev", "score_change",
                                     "verdict_now", "verdict_prev", "price_now", "price_change_pct"]].copy()
        changes_styled = (
            changes_display.style
            .format({
                "score_change": lambda x: f"{x:+.0f}" if pd.notna(x) else "-",
                "price_change_pct": lambda x: f"{x:+.1f}%" if pd.notna(x) else "-",
                "price_now": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
            })
            .map(color_delta_local, subset=["score_change", "price_change_pct"])
        )
        st.dataframe(changes_styled, use_container_width=True, hide_index=True)


# ======= TAB 2: Detail =======
with tab2:
    selected = st.selectbox("Select Ticker", options=sorted(df["Ticker"].tolist()))
    if selected:
        row = df[df["Ticker"] == selected].iloc[0]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Price", f"Rp {row['Price']:,.0f}")
        col2.metric("Momentum", f"{row['Momentum']}/100")
        col3.metric("Quality", f"{row['Quality']}/100" if pd.notna(row['Quality']) else "N/A")
        col4.metric("Overall", f"{row['Score']}/100")
        with col5:
            watchlist = load_watchlist()
            if selected in watchlist:
                if st.button("★ Remove from Watchlist", key=f"wl_rm_{selected}"):
                    remove_from_watchlist(selected)
                    st.rerun()
            else:
                if st.button("☆ Add to Watchlist", key=f"wl_add_{selected}"):
                    add_to_watchlist(selected)
                    st.rerun()

        st.markdown(f"**Verdict:** {row['Verdict']} | **Volatility:** {row['Vol']} | **Category:** {row['Category']}")

        # === PATTERN CLASSIFICATION (BULLISH / RECOVERY / etc) ===
        # Fetch fresh metrics for pattern analysis
        with st.spinner("Analyzing price pattern..."):
            try:
                r_pattern = analyze_ticker(selected, row["Category"])
                if r_pattern:
                    m_p = r_pattern["metrics"]
                    pattern = classify_pattern(m_p)

                    # Banner
                    st.markdown("### 🎯 Price Pattern Analysis")
                    if pattern["color"] == "green":
                        st.success(f"**{pattern['pattern']}** — {pattern['desc']}\n\n💡 **Action:** {pattern['action']}")
                    elif pattern["color"] == "yellow":
                        st.info(f"**{pattern['pattern']}** — {pattern['desc']}\n\n💡 **Action:** {pattern['action']}")
                    elif pattern["color"] == "orange":
                        st.warning(f"**{pattern['pattern']}** — {pattern['desc']}\n\n💡 **Action:** {pattern['action']}")
                    elif pattern["color"] == "red":
                        st.error(f"**{pattern['pattern']}** — {pattern['desc']}\n\n💡 **Action:** {pattern['action']}")
                    else:
                        st.write(f"**{pattern['pattern']}** — {pattern['desc']}\n\n💡 **Action:** {pattern['action']}")

                    # ATH / Drawdown context
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("1Y ATH (Peak)", f"Rp {m_p['peak_1y']:,.0f}",
                               delta=f"{m_p['drawdown']:+.1f}% dari peak" if m_p.get('drawdown') else None)
                    col2.metric("1Y ATL (Trough)", f"Rp {m_p['trough_1y']:,.0f}",
                               delta=f"+{m_p['recovery_from_bottom']:.1f}% dari bottom" if m_p.get('recovery_from_bottom') else None)
                    col3.metric("Days from Peak", f"{m_p.get('days_from_peak', 0)}d ago" if m_p.get('days_from_peak') else "-")
                    col4.metric("Days from Trough", f"{m_p.get('days_from_trough', 0)}d ago" if m_p.get('days_from_trough') else "-")
            except Exception as e:
                st.caption(f"Pattern analysis unavailable: {e}")
        st.divider()

        with st.spinner(f"Loading {selected} chart..."):
            hist = fetch_price_history(selected)
        if hist is not None and not hist.empty:
            st.plotly_chart(render_chart(hist, selected), use_container_width=True)

        st.subheader("📊 Technical")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Returns**")
            st.write(f"1Y: {row['1Y%']:+.1f}%")
            st.write(f"3M: {row['3M%']:+.1f}%" if pd.notna(row['3M%']) else "3M: -")
            st.write(f"1M: {row['1M%']:+.1f}%" if pd.notna(row['1M%']) else "1M: -")
            st.write(f"1W: {row['1W%']:+.1f}%" if pd.notna(row['1W%']) else "1W: -")
        with col2:
            st.markdown("**Trend**")
            st.write(f"vs MA50: {row['vsMA50%']:+.1f}%")
            st.write(f"Higher Lows: {'✓' if row['HL'] else '✗'}")
            st.write(f"Multi-Timeframe: {'✓ Aligned' if row['MTF'] else '✗ Diverged'}")
            div = row['Divergence']
            div_str = {"bearish": "⬇ Bearish", "bullish": "⬆ Bullish"}.get(div, "None")
            st.write(f"Divergence: {div_str}")
        with col3:
            st.markdown("**Volatility & RSI**")
            st.write(f"RSI: {row['RSI']:.0f}")
            st.write(f"ATR: {row['ATR%']:.1f}%")
            st.write(f"Volatility: {row['Vol']}")
            if row['RSI'] >= 70:
                st.warning("⚠️ Overbought")
            elif row['RSI'] <= 30:
                st.info("💡 Oversold")

        st.subheader("💼 Fundamental")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Valuation**")
            per = row.get('PER')
            pbv = row.get('PBV')
            st.write(f"PER: {per:.1f}" if pd.notna(per) else "PER: -")
            st.write(f"PBV: {pbv:.1f}" if pd.notna(pbv) else "PBV: -")
        with col2:
            st.markdown("**Profitability**")
            roe = row.get('ROE%')
            st.write(f"ROE: {roe:.1f}%" if pd.notna(roe) else "ROE: -")
        with col3:
            st.markdown("**Income**")
            dy = row.get('DivY%')
            st.write(f"Dividend Yield: {dy:.1f}%" if pd.notna(dy) else "DivY: -")
            de = row.get('D/E')
            st.write(f"D/E: {de:.1f}" if pd.notna(de) else "D/E: -")

        # Entry suggestion
        st.divider()
        st.subheader("💡 Entry Suggestion")
        price = row["Price"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Buy Levels:**
            - 🟡 Aggressive: Rp {price:,.0f}
            - 🟢 Moderate: Rp {price*0.97:,.0f} (-3%)
            - 🔵 Conservative: Rp {price*0.92:,.0f} (-8%)
            """)
        with col2:
            st.markdown(f"""
            **Risk Management:**
            - 🔴 Stop Loss: Rp {price*0.90:,.0f} (-10%)
            - 🎯 Target 1: Rp {price*1.12:,.0f} (+12%)
            - 🎯 Target 2: Rp {price*1.25:,.0f} (+25%)
            """)

        # Financial Summary section
        st.divider()
        st.subheader(f"📑 Financial Summary — {selected}")

        with st.spinner("Fetching financial statements..."):
            try:
                fin = _cached_financials_global(selected)
            except Exception as e:
                st.error(f"Failed to fetch financials: {e}")
                fin = {}

        income_annual = fin.get("income_annual", [])
        income_q = fin.get("income_quarterly", [])
        balance = fin.get("balance", {})
        cashflow = fin.get("cashflow", {})

        if not income_annual:
            st.info("Financial data tidak tersedia untuk ticker ini.")
        else:
            # === AI INTERPRETATION ===
            interpretation = interpret_financials(fin)

            # === COMBINED SIGNAL: Momentum + Fundamental ===
            screener_score = row["Score"]
            momentum_score = row["Momentum"]
            fin_verdict = interpretation["verdict"]

            fundamental_ok = ("🟢" in fin_verdict)
            momentum_ok = (momentum_score >= 45)

            if momentum_ok and fundamental_ok:
                combined = ("🟢 **BUY NOW** — Momentum + Fundamental both positive. Ideal entry, high conviction.",
                           "success")
            elif fundamental_ok and not momentum_ok:
                combined = (f"🟡 **VALUE TRAP WARNING** — Fundamental strong (perusahaan sehat) TAPI chart lagi turun (momentum {momentum_score}). "
                           "Wait for momentum reversal atau siap hold jangka panjang (12+ bulan). Ada risiko bagholder kalau salah timing.",
                           "warning")
            elif momentum_ok and not fundamental_ok:
                combined = (f"🟠 **MOMENTUM CHASE** — Chart naik tapi fundamental lemah. Short-term speculation only. "
                           "Set tight stop loss. Bisa reverse cepat kalau sentiment berubah.",
                           "warning")
            else:
                combined = ("🔴 **AVOID** — Momentum + Fundamental both negative. Multiple red flags. Skip untuk sekarang.",
                           "error")

            # Combined signal banner
            st.markdown("### 🎯 Combined Investment Signal")
            msg, level = combined
            if level == "success":
                st.success(msg)
            elif level == "warning":
                st.warning(msg)
            else:
                st.error(msg)

            # Comparison metrics
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            with comp_col1:
                mom_emoji = "🟢" if momentum_ok else "🔴"
                st.metric(f"{mom_emoji} Momentum Signal", f"{momentum_score}/100",
                         delta="Chart / Trend / MA")
            with comp_col2:
                fund_emoji = "🟢" if fundamental_ok else "🔴"
                st.metric(f"{fund_emoji} Fundamental Signal", fin_verdict.split(" — ")[0].replace("🟢 ", "").replace("🟡 ", "").replace("🟠 ", "").replace("🔴 ", ""),
                         delta="Revenue / Margin / D/E")
            with comp_col3:
                st.metric("Screener Overall", f"{screener_score}/100",
                         delta="60% mom + 40% quality")

            st.divider()

            # Verdict banner
            verdict = interpretation["verdict"]
            if "🟢" in verdict:
                st.success(f"**Overall:** {verdict}")
            elif "🟡" in verdict or "⚪" in verdict:
                st.info(f"**Overall:** {verdict}")
            elif "🟠" in verdict:
                st.warning(f"**Overall:** {verdict}")
            else:
                st.error(f"**Overall:** {verdict}")

            # Narrative sections
            with st.expander("📝 Financial Narrative — Apa yang Terjadi", expanded=True):
                narratives = interpretation["narratives"]
                labels = {
                    "revenue": "💰 Revenue Growth",
                    "profitability": "📊 Profitability",
                    "net_income": "💵 Net Income",
                    "quarterly": "📅 Quarterly Momentum",
                    "balance_sheet": "🏦 Balance Sheet Health",
                    "cash_flow": "💧 Cash Flow",
                }
                for section, text in narratives.items():
                    label = labels.get(section, section.title())
                    st.markdown(f"**{label}:** {text}")

            # Flags summary
            col_g, col_y, col_r = st.columns(3)
            with col_g:
                st.markdown(f"**🟢 Green Flags ({len(interpretation['green_flags'])}):**")
                for f in interpretation["green_flags"]:
                    st.caption(f"✓ {f}")
            with col_y:
                st.markdown(f"**🟡 Yellow Flags ({len(interpretation['yellow_flags'])}):**")
                for f in interpretation["yellow_flags"]:
                    st.caption(f"⚠ {f}")
            with col_r:
                st.markdown(f"**🔴 Red Flags ({len(interpretation['red_flags'])}):**")
                for f in interpretation["red_flags"]:
                    st.caption(f"✗ {f}")

            st.divider()

            # Latest annual highlight
            latest = income_annual[0]
            st.markdown(f"**Latest Annual: {latest['period']}**")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Revenue", format_large(latest["revenue"]),
                       delta=f"{latest.get('revenue_yoy', 0):+.1f}% YoY" if latest.get("revenue_yoy") is not None else None)
            col2.metric("Net Income", format_large(latest["net_income"]),
                       delta=f"{latest.get('net_income_yoy', 0):+.1f}% YoY" if latest.get("net_income_yoy") is not None else None)
            col3.metric("EBITDA", format_large(latest["ebitda"]))
            col4.metric("Net Margin", f"{latest['net_margin']:.1f}%" if latest.get("net_margin") else "-")

            # Color helper for financial growth
            def color_growth_fin(val):
                try:
                    if pd.isna(val): return ""
                except (TypeError, ValueError):
                    pass
                if isinstance(val, (int, float)):
                    if val > 10: return "background-color: #16a34a33; color: #22c55e; font-weight: bold"
                    if val > 0: return "color: #22c55e; font-weight: bold"
                    if val > -10: return "color: #f97316; font-weight: bold"
                    return "background-color: #dc262633; color: #ef4444; font-weight: bold"
                return ""

            def trend_arrow_fin(val):
                if pd.isna(val) or val is None: return "-"
                if val > 20: return f"🚀 {val:+.1f}%"
                if val > 10: return f"📈 {val:+.1f}%"
                if val > 0: return f"↗ {val:+.1f}%"
                if val == 0: return "→ 0.0%"
                if val > -10: return f"↘ {val:.1f}%"
                if val > -20: return f"📉 {val:.1f}%"
                return f"⚠️ {val:.1f}%"

            # Annual trend table with color + trend arrows
            st.markdown("**Annual Income Trend (5 Years):**")
            ann_rows = []
            for p in income_annual[:5]:
                ann_rows.append({
                    "Period": p["period"],
                    "Revenue": format_large(p["revenue"]),
                    "Revenue YoY": p.get("revenue_yoy"),  # numeric for coloring
                    "Trend Rev": trend_arrow_fin(p.get("revenue_yoy")),
                    "Net Income": format_large(p["net_income"]),
                    "Net Income YoY": p.get("net_income_yoy"),  # numeric
                    "Trend NI": trend_arrow_fin(p.get("net_income_yoy")),
                    "Net Margin": p.get("net_margin"),  # numeric
                    "EPS": f"{p['eps']:.2f}" if p.get("eps") else "-",
                })
            ann_df = pd.DataFrame(ann_rows)

            ann_styled = (
                ann_df.style
                .format({
                    "Revenue YoY": lambda x: f"{x:+.1f}%" if pd.notna(x) else "-",
                    "Net Income YoY": lambda x: f"{x:+.1f}%" if pd.notna(x) else "-",
                    "Net Margin": lambda x: f"{x:.1f}%" if pd.notna(x) else "-",
                })
                .map(color_growth_fin, subset=["Revenue YoY", "Net Income YoY"])
            )
            st.dataframe(ann_styled, use_container_width=True, hide_index=True)

            # Quarterly trend (last 4 quarters) with color + trend arrows
            if income_q:
                st.markdown("**Quarterly Trend (Last 4 Quarters):**")
                q_rows = []
                for p in income_q[:4]:
                    q_rows.append({
                        "Period": p["period"],
                        "Revenue": format_large(p["revenue"]),
                        "Rev QoQ": p.get("revenue_qoq"),
                        "Trend QoQ": trend_arrow_fin(p.get("revenue_qoq")),
                        "Rev YoY": p.get("revenue_yoy"),
                        "Trend YoY": trend_arrow_fin(p.get("revenue_yoy")),
                        "Net Income": format_large(p["net_income"]),
                        "NI QoQ": p.get("net_income_qoq"),
                        "NI YoY": p.get("net_income_yoy"),
                    })
                q_df = pd.DataFrame(q_rows)

                q_styled = (
                    q_df.style
                    .format({
                        "Rev QoQ": lambda x: f"{x:+.1f}%" if pd.notna(x) else "-",
                        "Rev YoY": lambda x: f"{x:+.1f}%" if pd.notna(x) else "-",
                        "NI QoQ": lambda x: f"{x:+.1f}%" if pd.notna(x) else "-",
                        "NI YoY": lambda x: f"{x:+.1f}%" if pd.notna(x) else "-",
                    })
                    .map(color_growth_fin, subset=["Rev QoQ", "Rev YoY", "NI QoQ", "NI YoY"])
                )
                st.dataframe(q_styled, use_container_width=True, hide_index=True)

            # Balance Sheet
            if balance:
                st.markdown(f"**Balance Sheet — {balance.get('period', 'N/A')}:**")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Assets", format_large(balance.get("total_assets")))
                col2.metric("Total Equity", format_large(balance.get("total_equity")))
                col3.metric("Total Debt", format_large(balance.get("total_debt")))
                col4.metric("Cash", format_large(balance.get("cash")))

                if balance.get("debt_to_equity") is not None:
                    st.caption(f"💡 Debt/Equity: {balance['debt_to_equity']:.2f} | Equity Ratio: {balance.get('equity_ratio', 0):.1f}%")

            # Cash Flow
            if cashflow:
                st.markdown(f"**Cash Flow — {cashflow.get('period', 'N/A')}:**")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Operating CF", format_large(cashflow.get("operating_cash")))
                col2.metric("Investing CF", format_large(cashflow.get("investing_cash")))
                col3.metric("Financing CF", format_large(cashflow.get("financing_cash")))
                col4.metric("Free Cash Flow", format_large(cashflow.get("free_cash_flow")))

        # === Fair Value / Valuation section ===
        st.divider()
        st.subheader(f"💰 Fair Value Analysis — {selected}")

        # Customize target multipliers
        with st.expander("⚙️ Custom Valuation Assumptions"):
            v_col1, v_col2, v_col3 = st.columns(3)
            with v_col1:
                v_target_pe = st.number_input("Target PE Multiple", 5.0, 50.0, 15.0, step=1.0, key=f"pe_{selected}")
            with v_col2:
                v_target_pbv = st.number_input("Target PBV Multiple", 0.5, 10.0, 2.0, step=0.5, key=f"pbv_{selected}")
            with v_col3:
                v_discount = st.number_input("Discount Rate (WACC)", 0.05, 0.20, 0.10, step=0.01, key=f"dr_{selected}")

        with st.spinner("Computing fair value..."):
            try:
                val = _cached_valuation_global(selected, v_target_pe, v_target_pbv, v_discount)
            except Exception as e:
                st.error(f"Failed to compute valuation: {e}")
                val = {}

        summary = val.get("summary")
        valuations = val.get("valuations", {})

        if not summary:
            st.info("Fair value analysis tidak tersedia (missing EPS/BVPS data).")
        else:
            # Verdict banner
            verdict = summary["verdict"]
            if "🟢" in verdict:
                st.success(f"**{verdict}**")
            elif "🟡" in verdict:
                st.info(f"**{verdict}**")
            elif "🟠" in verdict:
                st.warning(f"**{verdict}**")
            else:
                st.error(f"**{verdict}**")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"Rp {summary['current_price']:,.0f}")
            col2.metric("Avg Fair Value", f"Rp {summary['avg_fair_value']:,.0f}")
            col3.metric("Median Fair Value", f"Rp {summary['median_fair_value']:,.0f}")

            upside = summary["upside_pct"]
            delta_color = "normal" if upside >= 0 else "inverse"
            col4.metric("Upside/Downside",
                       f"{upside:+.1f}%",
                       delta=f"vs current" if upside > 0 else "vs current")

            # Per-method breakdown
            st.markdown("**Valuation Methods Breakdown:**")
            val_rows = []
            for name, m in valuations.items():
                fv = m["fair_value"]
                upside_m = (fv - summary["current_price"]) / summary["current_price"] * 100
                val_rows.append({
                    "Method": name,
                    "Fair Value": fv,
                    "Upside %": upside_m,
                    "Formula": m["formula"],
                    "Note": m["note"],
                })
            val_df = pd.DataFrame(val_rows)

            def color_upside(val):
                try:
                    if pd.isna(val): return ""
                except (TypeError, ValueError):
                    pass
                if isinstance(val, (int, float)):
                    if val > 20: return "color: #22c55e; font-weight: bold"
                    if val > 0: return "color: #22c55e"
                    if val > -20: return "color: #f97316"
                    return "color: #ef4444; font-weight: bold"
                return ""

            val_styled = (
                val_df.style
                .format({
                    "Fair Value": lambda x: f"Rp {x:,.0f}",
                    "Upside %": lambda x: f"{x:+.1f}%",
                })
                .map(color_upside, subset=["Upside %"])
            )
            st.dataframe(val_styled, use_container_width=True, hide_index=True)

            # Interpretation
            st.caption(f"💡 **Interpretation:** Based on {summary['n_methods']} valuation methods. "
                      f"Average fair value **Rp {summary['avg_fair_value']:,.0f}** vs current **Rp {summary['current_price']:,.0f}**. "
                      f"Cross-check dengan chart pattern + fundamental sebelum decide.")

            st.caption("⚠️ **Warning:** Valuation ≠ jaminan harga akan bergerak ke fair value. Market bisa remain irrational lebih lama dari mas dewo tetap solvent.")

        # News section
        st.divider()
        st.subheader(f"📰 Latest News — {selected}")

        @st.cache_data(ttl=3600, show_spinner=False)  # cache 1 hour
        def _cached_news(ticker):
            items = fetch_news(ticker, count=8)
            return enrich_news(items)

        with st.spinner("Fetching news..."):
            news_items = _cached_news(selected)

        if not news_items:
            st.info("No recent news found for this ticker.")
        else:
            # High-impact alert
            if has_high_impact_news(news_items):
                impact_items = [n for n in news_items if n["is_high_impact"]]
                st.warning(f"⚠️ **{len(impact_items)} high-impact corporate action detected** — click to expand below.")

            for i, item in enumerate(news_items):
                sentiment = item.get("sentiment", "neutral")
                sent_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}[sentiment]
                actions = item.get("corporate_actions", [])
                impact_badge = " 🚨 **HIGH IMPACT**" if item.get("is_high_impact") else ""

                title = item.get("title", "").strip()
                link = item.get("link", "#")
                source = item.get("source", "")
                pub = format_published(item.get("published", ""))

                with st.container():
                    st.markdown(f"{sent_emoji} **[{title}]({link})**{impact_badge}")
                    caption_parts = []
                    if source: caption_parts.append(f"📰 {source}")
                    if pub: caption_parts.append(f"🕐 {pub}")
                    if actions: caption_parts.append(f"🏷️ {', '.join(actions)}")
                    st.caption(" | ".join(caption_parts))
                    st.divider()

# ======= TAB 3: Sector Strength =======
with tab3:
    st.subheader("🏭 Sector Strength Analysis")
    if not df.empty:
        sector_df = df.groupby("Category").agg({
            "Score": "mean",
            "Momentum": "mean",
            "Quality": "mean",
            "1Y%": "mean",
            "3M%": "mean",
            "1M%": "mean",
            "Ticker": "count",
        }).round(1).reset_index()
        sector_df.columns = ["Sector", "Avg Score", "Avg Momentum", "Avg Quality", "Avg 1Y%", "Avg 3M%", "Avg 1M%", "# Stocks"]
        sector_df = sector_df.sort_values("Avg Score", ascending=False)

        st.markdown("**Sector ranking by average score:**")
        sector_styled = sector_df.style.map(color_score, subset=["Avg Score", "Avg Momentum", "Avg Quality"])
        st.dataframe(sector_styled, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**Top sector currently:** " + sector_df.iloc[0]["Sector"] + f" (Score {sector_df.iloc[0]['Avg Score']:.0f})")

# ======= TAB BACKTEST =======
with tab_backtest:
    st.subheader("🧪 Backtest Engine")
    st.caption("Validate momentum scoring system against historical data. Uses momentum score only (fundamentals not available historically).")

    col1, col2, col3 = st.columns(3)
    with col1:
        bt_min_score = st.slider("Min Score to Buy", 40, 90, 60, step=5, key="bt_score")
    with col2:
        bt_hold_days = st.slider("Hold Days", 20, 120, 60, step=10, key="bt_hold")
    with col3:
        bt_step_days = st.slider("Signal Sampling (days)", 7, 60, 30, step=7, key="bt_step")

    st.caption(f"Setup: Beli kalau score ≥ {bt_min_score}, hold {bt_hold_days} hari, sampling tiap {bt_step_days} hari.")

    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest... (takes 1-3 minutes for full ticker list)"):
            progress_bar = st.progress(0.0)
            status = st.empty()

            def cb(current, total, ticker):
                progress_bar.progress(current / total, text=f"[{current}/{total}] {ticker}")

            tickers_only = [t for t, _ in load_tickers(TICKERS_FILE)]
            trades_df = run_backtest(
                tickers=tickers_only,
                min_score=bt_min_score,
                hold_days=bt_hold_days,
                step_days=bt_step_days,
                progress_callback=cb,
            )

            progress_bar.empty()
            status.empty()

            st.session_state["backtest_result"] = trades_df

            # Benchmark
            bench_df = benchmark_ihsg(hold_days=bt_hold_days, step_days=bt_step_days)
            st.session_state["backtest_benchmark"] = bench_df

    if "backtest_result" in st.session_state:
        trades_df = st.session_state["backtest_result"]

        if trades_df.empty:
            st.warning("No trades matched criteria. Try lowering Min Score.")
        else:
            summary = summarize_backtest(trades_df)

            # Top metrics
            st.divider()
            st.markdown("### 📊 Summary Metrics")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", summary["total_trades"])
            col2.metric("Win Rate", f"{summary['win_rate']:.1f}%")
            col3.metric("Avg Return per Trade", f"{summary['avg_return']:+.2f}%")
            col4.metric("Risk/Reward", f"{summary['risk_reward_ratio']:.2f}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Win", f"{summary['avg_win']:+.2f}%")
            col2.metric("Avg Loss", f"{summary['avg_loss']:.2f}%")
            col3.metric("Best Trade", f"{summary['best_trade']:+.2f}%")
            col4.metric("Worst Trade", f"{summary['worst_trade']:.2f}%")

            # Benchmark comparison
            if "backtest_benchmark" in st.session_state:
                bench = st.session_state["backtest_benchmark"]
                if not bench.empty:
                    st.divider()
                    st.markdown("### 📉 vs IHSG Buy-and-Hold Benchmark")
                    bench_avg = bench["return_pct"].mean()
                    bench_win = (bench["return_pct"] > 0).sum() / len(bench) * 100
                    col1, col2, col3 = st.columns(3)
                    col1.metric("IHSG Avg Return", f"{bench_avg:+.2f}%",
                               delta=f"{summary['avg_return'] - bench_avg:+.2f}% vs strategy")
                    col2.metric("IHSG Win Rate", f"{bench_win:.1f}%",
                               delta=f"{summary['win_rate'] - bench_win:+.1f}% vs strategy")
                    col3.metric("IHSG Std Dev", f"{bench['return_pct'].std():.2f}%")

            # Return distribution
            st.divider()
            st.markdown("### 📈 Return Distribution")
            import plotly.express as px

            fig_hist = px.histogram(
                trades_df, x="return_pct", nbins=30,
                title="Trade Return Distribution",
                labels={"return_pct": "Return %", "count": "Number of Trades"},
                color_discrete_sequence=["#3b82f6"],
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
            fig_hist.add_vline(x=summary["avg_return"], line_dash="dash", line_color="green",
                              annotation_text=f"Avg {summary['avg_return']:.1f}%")
            st.plotly_chart(fig_hist, use_container_width=True)

            # Backtest color helper
            def color_bt(val):
                try:
                    if pd.isna(val):
                        return ""
                except (TypeError, ValueError):
                    pass
                if isinstance(val, (int, float)):
                    if val > 0: return "color: #22c55e; font-weight: bold"
                    if val < 0: return "color: #ef4444; font-weight: bold"
                return ""

            def bt_format(df, cols_pct=None):
                cols_pct = cols_pct or []
                fmts = {
                    "entry_price": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                    "exit_price": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
                }
                for c in cols_pct:
                    fmts[c] = lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
                return df.style.format(fmts).map(color_bt, subset=cols_pct)

            # Top winners
            st.divider()
            st.markdown("### 🏆 Top 10 Winning Trades")
            top10 = trades_df.sort_values("return_pct", ascending=False).head(10).copy()
            st.dataframe(
                bt_format(top10, cols_pct=["return_pct", "max_drawdown_pct", "max_upside_pct"]),
                use_container_width=True, hide_index=True,
            )

            st.markdown("### 💀 Worst 10 Losing Trades")
            worst10 = trades_df.sort_values("return_pct").head(10).copy()
            st.dataframe(
                bt_format(worst10, cols_pct=["return_pct", "max_drawdown_pct", "max_upside_pct"]),
                use_container_width=True, hide_index=True,
            )

            # Ticker performance summary
            st.divider()
            st.markdown("### 🎯 Per-Ticker Performance")
            per_ticker = trades_df.groupby("ticker").agg(
                trades=("return_pct", "count"),
                avg_return=("return_pct", "mean"),
                win_rate=("return_pct", lambda x: (x > 0).sum() / len(x) * 100),
                best=("return_pct", "max"),
                worst=("return_pct", "min"),
            ).round(2).reset_index()
            per_ticker = per_ticker.sort_values("avg_return", ascending=False)

            per_ticker_styled = (
                per_ticker.style
                .format({
                    "avg_return": lambda x: f"{x:+.2f}%",
                    "best": lambda x: f"{x:+.2f}%",
                    "worst": lambda x: f"{x:+.2f}%",
                    "win_rate": lambda x: f"{x:.1f}%",
                })
                .map(color_bt, subset=["avg_return", "best", "worst"])
            )
            st.dataframe(per_ticker_styled, use_container_width=True, hide_index=True)

            # Export
            st.download_button(
                "📥 Export Trades CSV",
                data=trades_df.to_csv(index=False).encode("utf-8"),
                file_name=f"backtest_score{bt_min_score}_hold{bt_hold_days}.csv",
                mime="text/csv",
            )

    else:
        st.info("Klik **Run Backtest** untuk memulai. Hasil akan menampilkan:\n"
                "- Total trades matching criteria\n"
                "- Win rate + avg return\n"
                "- Risk/reward ratio\n"
                "- IHSG benchmark comparison\n"
                "- Return distribution histogram\n"
                "- Top/worst trades + per-ticker breakdown")


# ======= TAB 4: Methodology =======
with tab4:
    st.markdown("""
    ### Scoring System v2 — Two Scores

    #### Momentum Score (0-100) — Technical
    | Category | Max Pts |
    |----------|---------|
    | 1Y Return | 20 |
    | 3M Return | 20 |
    | 1M Return | 15 |
    | MA20/50/200 alignment | 15 |
    | Higher Lows | 10 |
    | Volume Trend | 5 |
    | RSI Health (40-70) | 5 |
    | Multi-Timeframe Alignment | 5 |
    | Divergence bonus/penalty | ±5 |

    #### Quality Score (0-100) — Fundamental
    | Category | Max Pts |
    |----------|---------|
    | PER (<10 best) | 20 |
    | PBV (<1 best) | 15 |
    | ROE (>20% best) | 20 |
    | Earnings Growth (>20% best) | 15 |
    | Dividend Yield (>7% best) | 15 |
    | D/E Ratio (<30 best) | 15 |

    Requires min 3 valid metrics for reliable quality score.

    #### Overall Score
    `Overall = 0.6 * Momentum + 0.4 * Quality`
    (If quality unavailable, uses momentum only)

    ### Verdict Thresholds
    | Score | Verdict |
    |-------|---------|
    | ≥75 | 🟢 STRONG BUY |
    | 60-74 | 🟢 BUY |
    | 45-59 | 🟡 WATCH |
    | 30-44 | 🟠 WEAK |
    | <30 | 🔴 AVOID |

    ### New in v2
    1. **Fundamental data** — PER, PBV, ROE, Dividend Yield, D/E
    2. **ATR-based volatility** — for position sizing (🧊 LOW / ⚡ MED / 🔥 HIGH)
    3. **Multi-timeframe** — daily + weekly bullish alignment
    4. **Divergence detection** — RSI vs Price divergence (bearish warning / bullish opportunity)
    5. **Sector strength** — aggregate per category (see Sector tab)

    ### Data Quality Notes
    - Yahoo Finance data for IDX stocks sometimes has scale issues
    - Values are capped at reasonable levels in display
    - Cross-reference with official sources (IDX.co.id, broker fundamental data)

    ### Disclaimer
    Screening tool only. Not investment advice. DYOR. Always cross-check chart + news before trade.
    """)
