#!/usr/bin/env python3
"""
US Market Screener — momentum + quality scoring for NYSE/NASDAQ.
Reuses core scoring logic from screener.py but with US ticker handling.

Usage:
    ./us_screener.py                  # screen all US tickers
    ./us_screener.py --top 20
    ./us_screener.py --ticker AVGO
    ./us_screener.py --category Tech
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from tabulate import tabulate

from screener import (
    compute_metrics,
    score_momentum,
    score_overall,
    score_quality,
    verdict,
    volatility_label,
)

BASE_DIR = Path(__file__).parent
US_TICKERS_FILE = BASE_DIR / "us_tickers.txt"


def load_us_tickers(path: Path = None):
    path = path or US_TICKERS_FILE
    tickers = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("#", 1)
        ticker = parts[0].strip()
        category = parts[1].strip() if len(parts) > 1 else "Uncategorized"
        tickers.append((ticker, category))
    return tickers


def fetch_us_history(ticker: str, period="1y", interval="1d"):
    """Fetch US market history (no suffix)."""
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if data.empty or len(data) < 20:
            return None
        return data
    except Exception as e:
        print(f"[WARN] {ticker}: {e}", file=sys.stderr)
        return None


def fetch_us_fundamentals(ticker: str) -> dict:
    """Fetch US fundamentals via yfinance info."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToBook": info.get("priceToBook"),
            "returnOnEquity": info.get("returnOnEquity"),
            "dividendYield": info.get("dividendYield"),
            "debtToEquity": info.get("debtToEquity"),
            "earningsGrowth": info.get("earningsGrowth"),
            "profitMargins": info.get("profitMargins"),
            "marketCap": info.get("marketCap"),
        }
    except Exception:
        return {}


def analyze_us_ticker(ticker: str, category: str, verbose=False):
    daily = fetch_us_history(ticker, period="1y", interval="1d")
    if daily is None:
        return None

    weekly = fetch_us_history(ticker, period="2y", interval="1wk")
    fund = fetch_us_fundamentals(ticker)

    try:
        m = compute_metrics(daily, weekly_df=weekly, fundamentals=fund)
        mom = score_momentum(m)
        qual = score_quality(m)
        overall = score_overall(mom, qual)
        return {
            "ticker": ticker,
            "category": category,
            "metrics": m,
            "momentum": mom,
            "quality": qual,
            "overall": overall,
            "verdict": verdict(overall),
        }
    except Exception as e:
        if verbose:
            print(f"[WARN] {ticker} compute: {e}", file=sys.stderr)
        return None


def screen_us_all(tickers: list = None, progress_callback=None) -> pd.DataFrame:
    """Run full US screener."""
    if tickers is None:
        tickers = load_us_tickers()

    results = []
    total = len(tickers)

    for i, (ticker, category) in enumerate(tickers):
        if progress_callback:
            progress_callback(i + 1, total, ticker)

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
            "Divergence": m.get("divergence"),
            "RSI": m["rsi"],
            "ATR%": m["atr_pct"],
            "PER": m.get("pe"),
            "PBV": m.get("pbv"),
            "ROE": m.get("roe"),
            "DivY": m.get("div_yield"),
            "D/E": m.get("de_ratio"),
            "Momentum": r["momentum"],
            "Quality": r["quality"],
            "Score": r["overall"],
            "Verdict": r["verdict"],
            "Vol": volatility_label(m["atr_pct"]),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="US Market Momentum + Quality Screener")
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--min-score", type=int, default=0)
    parser.add_argument("--sort", default="overall")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    tickers = load_us_tickers()
    if args.ticker:
        tickers = [(t, c) for t, c in tickers if t.upper() == args.ticker.upper()]
    if args.category:
        tickers = [(t, c) for t, c in tickers if args.category.lower() in c.lower()]

    print(f"Screening {len(tickers)} US tickers...\n", file=sys.stderr)

    df = screen_us_all(tickers)
    if df.empty:
        print("No results.", file=sys.stderr)
        return

    df = df[df["Score"] >= args.min_score]
    df = df.sort_values("Score", ascending=False)
    if args.top:
        df = df.head(args.top)

    # Display
    display = df.copy()
    display["HL"] = display["HL"].map({True: "✓", False: "✗"})
    display["MTF"] = display["MTF"].map({True: "✓", False: "✗"})
    display["Price"] = display["Price"].apply(lambda x: f"${x:,.2f}")
    for col in ["1Y%", "3M%", "1M%", "1W%", "vsMA50%"]:
        display[col] = display[col].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "-")
    display["RSI"] = display["RSI"].apply(lambda x: f"{x:.0f}")
    display["ATR%"] = display["ATR%"].apply(lambda x: f"{x:.1f}")
    for col in ["PER", "PBV", "D/E"]:
        display[col] = display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    for col in ["ROE", "DivY"]:
        display[col] = display[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) and abs(x) < 1 else (f"{x:.1f}" if pd.notna(x) else "-"))

    keep = ["Ticker", "Category", "Price", "1Y%", "3M%", "1M%",
            "HL", "MTF", "RSI", "PER", "ROE", "DivY", "Momentum", "Quality", "Score", "Verdict"]
    display = display[keep]

    print(tabulate(display, headers="keys", tablefmt="simple", stralign="right"))
    print(f"\nTotal: {len(display)} US tickers")


if __name__ == "__main__":
    main()
