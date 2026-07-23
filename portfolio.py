#!/usr/bin/env python3
"""
Portfolio helper module — reads position CSVs, fetches live prices, computes P/L.
"""

import os
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR)))
POSITIONS_CSV = DATA_DIR / "02-position-tracker.csv"
PENDING_CSV = DATA_DIR / "pending_orders.csv"

# Portfolio baseline (starting point for recovery tracking)
BASELINE_INVESTED = 73_666_205  # Total invested before recovery started
BASELINE_EQUITY_START = 49_580_501  # Equity when recovery started (22 Jul 2026)

# Cash tracking (updated after each session)
TRADING_BALANCE_FILE = DATA_DIR / "cash_state.txt"


def load_cash_state():
    """Load current cash state: trading balance + reserved pending."""
    if not TRADING_BALANCE_FILE.exists():
        return {"trading_balance": 0, "reserved_pending": 0}
    try:
        lines = TRADING_BALANCE_FILE.read_text().strip().splitlines()
        state = {}
        for line in lines:
            k, v = line.split("=", 1)
            state[k.strip()] = float(v.strip())
        return state
    except Exception:
        return {"trading_balance": 0, "reserved_pending": 0}


def save_cash_state(trading_balance: float, reserved_pending: float):
    TRADING_BALANCE_FILE.write_text(
        f"trading_balance={trading_balance}\n"
        f"reserved_pending={reserved_pending}\n"
    )


def load_positions() -> pd.DataFrame:
    if not POSITIONS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(POSITIONS_CSV)


def load_pending() -> pd.DataFrame:
    if not PENDING_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(PENDING_CSV)


def fetch_current_price(ticker: str) -> float:
    """Fetch latest close price. Returns None on failure."""
    try:
        yf_ticker = f"{ticker}.JK"
        data = yf.Ticker(yf_ticker).history(period="5d", interval="1d", auto_adjust=False)
        if data.empty:
            return None
        return float(data["Close"].iloc[-1])
    except Exception:
        return None


def fetch_prices_batch(tickers: list) -> dict:
    """Fetch prices for multiple tickers. Returns dict {ticker: price}."""
    prices = {}
    for t in tickers:
        p = fetch_current_price(t)
        if p is not None:
            prices[t] = p
    return prices


def compute_portfolio(positions_df: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """Add computed columns: Current Price, Market Value, P/L Rp, P/L %."""
    if positions_df.empty:
        return positions_df

    df = positions_df.copy()
    df["Current Price"] = df["Ticker"].map(prices)
    df["Market Value"] = df["Lot"] * 100 * df["Current Price"]
    df["P/L Rp"] = df["Market Value"] - df["Cost Basis"]
    df["P/L %"] = (df["P/L Rp"] / df["Cost Basis"]) * 100
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate stats: total invested, market value, P/L, per layer."""
    if df.empty or "Market Value" not in df.columns:
        return {
            "total_invested": 0, "total_value": 0, "total_pl": 0, "total_pct": 0,
            "layer1_invested": 0, "layer1_value": 0, "layer1_pl": 0,
            "layer2_invested": 0, "layer2_value": 0, "layer2_pl": 0,
        }

    total_invested = df["Cost Basis"].sum()
    total_value = df["Market Value"].sum()
    total_pl = df["P/L Rp"].sum()
    total_pct = (total_pl / total_invested * 100) if total_invested else 0

    l1 = df[df["Layer"] == "Layer 1"]
    l2 = df[df["Layer"] == "Layer 2"]

    return {
        "total_invested": total_invested,
        "total_value": total_value,
        "total_pl": total_pl,
        "total_pct": total_pct,
        "layer1_invested": l1["Cost Basis"].sum() if not l1.empty else 0,
        "layer1_value": l1["Market Value"].sum() if not l1.empty else 0,
        "layer1_pl": l1["P/L Rp"].sum() if not l1.empty else 0,
        "layer2_invested": l2["Cost Basis"].sum() if not l2.empty else 0,
        "layer2_value": l2["Market Value"].sum() if not l2.empty else 0,
        "layer2_pl": l2["P/L Rp"].sum() if not l2.empty else 0,
    }


def compute_days_held(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'Days Held' column based on Entry Date."""
    from datetime import datetime
    if df.empty or "Entry Date" not in df.columns:
        return df
    df = df.copy()
    today = datetime.now().date()

    def days(d):
        try:
            dt = datetime.strptime(str(d), "%Y-%m-%d").date()
            return (today - dt).days
        except Exception:
            return None

    df["Days Held"] = df["Entry Date"].apply(days)
    return df


def time_exit_signal(days: int, layer: str) -> dict:
    """Compute time-based exit recommendation."""
    if pd.isna(days) or days is None:
        return {"stage": "unknown", "signal": "-", "action": "-", "color": "gray"}

    # Layer 1 = passive hold, ga apply time exit
    if layer == "Layer 1":
        return {
            "stage": "legacy",
            "signal": f"Layer 1 Legacy ({days}d)",
            "action": "Passive hold, quarterly review",
            "color": "gray",
        }

    # Layer 2 time exit rules (swing 1-3 bulan)
    if days <= 30:
        return {
            "stage": "trust",
            "signal": f"✅ Day {days}/90 — TRUST PHASE",
            "action": "Trust the plan. Weekly check only.",
            "color": "green",
        }
    if days <= 60:
        return {
            "stage": "watch",
            "signal": f"👀 Day {days}/90 — WATCH PHASE",
            "action": "Weekly review chart + score. No panic.",
            "color": "yellow",
        }
    if days <= 75:
        return {
            "stage": "evaluate",
            "signal": f"🔍 Day {days}/90 — EVALUATE PHASE",
            "action": "Serious review. Check thesis intact. Prepare decision.",
            "color": "orange",
        }
    if days <= 90:
        return {
            "stage": "decision",
            "signal": f"⚠️ Day {days}/90 — DECISION WEEK",
            "action": "MUST decide: HOLD (with catalyst) / EXTEND (max +30d) / CUT (rotate).",
            "color": "red",
        }
    # Beyond 90 days
    return {
        "stage": "overdue",
        "signal": f"🚨 Day {days}/90 — OVERDUE",
        "action": "Time exit rule violated. Cut or set new deadline immediately.",
        "color": "critical",
    }


def check_alerts(df: pd.DataFrame) -> list:
    """Return list of alert dicts for positions near stop loss / take profit."""
    alerts = []
    if df.empty or "Current Price" not in df.columns:
        return alerts

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        cur = row.get("Current Price")
        if pd.isna(cur):
            continue

        sl = row.get("Stop Loss")
        tp1 = row.get("Take Profit 1")
        tp2 = row.get("Take Profit 2")

        # Stop loss near / hit
        if pd.notna(sl) and sl > 0:
            dist_sl_pct = (cur - sl) / cur * 100
            if cur <= sl:
                alerts.append({
                    "level": "🚨 CRITICAL", "ticker": ticker,
                    "msg": f"Stop loss HIT (current {cur:,.0f} ≤ SL {sl:,.0f})"
                })
            elif dist_sl_pct < 3:
                alerts.append({
                    "level": "⚠️ WARNING", "ticker": ticker,
                    "msg": f"Near stop loss ({dist_sl_pct:.1f}% away, cur {cur:,.0f} vs SL {sl:,.0f})"
                })

        # Take profit tercapai
        if pd.notna(tp1) and tp1 > 0:
            if cur >= tp1:
                alerts.append({
                    "level": "🎯 OPPORTUNITY", "ticker": ticker,
                    "msg": f"Take Profit 1 HIT (current {cur:,.0f} ≥ TP1 {tp1:,.0f})"
                })

        if pd.notna(tp2) and tp2 > 0:
            if cur >= tp2:
                alerts.append({
                    "level": "🎯 OPPORTUNITY", "ticker": ticker,
                    "msg": f"Take Profit 2 HIT (current {cur:,.0f} ≥ TP2 {tp2:,.0f})"
                })

    return alerts


# ==============================================================
# Data Quality Validation
# ==============================================================

def flag_data_quality(row: dict) -> list:
    """Return list of data quality warnings for a ticker."""
    warnings = []

    per = row.get("PER")
    if per is not None and not pd.isna(per):
        if per > 100:
            warnings.append(f"PER {per:.0f} unusually high — verify at Stockbit/IDX")
        if per < 0:
            warnings.append(f"PER {per:.1f} negative (loss company)")

    pbv = row.get("PBV")
    if pbv is not None and not pd.isna(pbv):
        if pbv > 20:
            warnings.append(f"PBV {pbv:.0f} unusually high — likely yfinance error")
        if pbv < 0:
            warnings.append(f"PBV {pbv:.1f} negative — data error")

    roe = row.get("ROE%")
    if roe is not None and not pd.isna(roe):
        if roe > 100:
            warnings.append(f"ROE {roe:.0f}% unusually high — verify")
        if roe < -50:
            warnings.append(f"ROE {roe:.0f}% deeply negative (loss company)")

    dy = row.get("DivY%")
    if dy is not None and not pd.isna(dy):
        if dy > 15:
            warnings.append(f"DivY {dy:.1f}% unusually high — likely yfinance error, verify at IDX/Stockbit")

    de = row.get("D/E")
    if de is not None and not pd.isna(de):
        if de > 500:
            warnings.append(f"D/E {de:.0f} unusually high — verify")

    return warnings


def has_data_quality_issue(row: dict) -> bool:
    return len(flag_data_quality(row)) > 0
