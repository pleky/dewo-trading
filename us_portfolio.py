#!/usr/bin/env python3
"""
US Portfolio module — handle US stock positions via Gotrade or similar brokers.
Uses yfinance without .JK suffix.
"""

import pandas as pd
import yfinance as yf

from db import get_engine


def load_us_positions() -> pd.DataFrame:
    try:
        df = pd.read_sql_query("""
            SELECT ticker AS "Ticker", name AS "Name", shares AS "Shares",
                   avg_cost_usd AS "Avg Cost USD", cost_basis_usd AS "Cost Basis USD",
                   notes AS "Notes"
            FROM us_positions
        """, get_engine())
        for col in ["Shares", "Avg Cost USD", "Cost Basis USD"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def fetch_us_price(ticker: str) -> float:
    """Fetch latest US stock/ETF price (no .JK suffix)."""
    try:
        data = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        if data.empty:
            return None
        return float(data["Close"].iloc[-1])
    except Exception:
        return None


def fetch_usd_idr_rate() -> float:
    """Fetch current USD/IDR exchange rate."""
    try:
        # USDIDR=X is the yfinance symbol for USD/IDR forex
        data = yf.Ticker("USDIDR=X").history(period="5d", interval="1d")
        if data.empty:
            return 16300  # fallback
        return float(data["Close"].iloc[-1])
    except Exception:
        return 16300


def compute_us_portfolio(positions_df: pd.DataFrame, prices: dict, usd_idr: float = 16300) -> pd.DataFrame:
    """Add P/L columns."""
    if positions_df.empty:
        return positions_df

    df = positions_df.copy()
    df["Current Price USD"] = df["Ticker"].map(prices)
    df["Market Value USD"] = df["Shares"] * df["Current Price USD"]
    df["P/L USD"] = df["Market Value USD"] - df["Cost Basis USD"]
    df["P/L %"] = (df["P/L USD"] / df["Cost Basis USD"]) * 100

    # IDR conversion
    df["Market Value IDR"] = df["Market Value USD"] * usd_idr
    df["Cost Basis IDR"] = df["Cost Basis USD"] * usd_idr
    df["P/L IDR"] = df["P/L USD"] * usd_idr

    return df


def summarize_us(df: pd.DataFrame, usd_idr: float = 16300) -> dict:
    """Aggregate US portfolio stats."""
    if df.empty or "Market Value USD" not in df.columns:
        return {"total_cost_usd": 0, "total_value_usd": 0, "total_pl_usd": 0,
                "total_cost_idr": 0, "total_value_idr": 0, "total_pl_idr": 0,
                "usd_idr_rate": usd_idr}

    return {
        "total_cost_usd": df["Cost Basis USD"].sum(),
        "total_value_usd": df["Market Value USD"].sum(),
        "total_pl_usd": df["P/L USD"].sum(),
        "total_pct": (df["P/L USD"].sum() / df["Cost Basis USD"].sum() * 100) if df["Cost Basis USD"].sum() else 0,
        "total_cost_idr": df["Cost Basis IDR"].sum(),
        "total_value_idr": df["Market Value IDR"].sum(),
        "total_pl_idr": df["P/L IDR"].sum(),
        "usd_idr_rate": usd_idr,
    }
