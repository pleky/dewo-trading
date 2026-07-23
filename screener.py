#!/usr/bin/env python3
"""
IDX Stock Momentum + Quality Screener v2

Improvements over v1:
1. Fundamental score (PER, PBV, ROE, Dividend Yield, D/E)
2. ATR-based volatility (position sizing signal)
3. Multi-timeframe (daily + weekly trend alignment)
4. Divergence detection (RSI vs Price)
5. Sector strength ranking
6. Two-score system: Momentum (technical) + Quality (fundamental)

Usage:
    ./screener.py                    # screen all tickers
    ./screener.py --top 10           # top 10 by overall score
    ./screener.py --min-score 60
    ./screener.py --sort momentum    # sort by momentum score
    ./screener.py --category Mining
    ./screener.py --ticker MAPI      # single ticker analysis
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

BASE_DIR = Path(__file__).parent
TICKERS_FILE = BASE_DIR / "tickers.txt"


def load_tickers(path: Path):
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


def fetch_history(ticker: str, period="1y", interval="1d"):
    yf_ticker = f"{ticker}.JK"
    try:
        data = yf.Ticker(yf_ticker).history(period=period, interval=interval, auto_adjust=False)
        if data.empty or len(data) < 20:
            return None
        return data
    except Exception as e:
        print(f"[WARN] {ticker}: {e}", file=sys.stderr)
        return None


def fetch_fundamentals(ticker: str) -> dict:
    """Fetch fundamentals from yfinance info. May return partial data."""
    yf_ticker = f"{ticker}.JK"
    try:
        info = yf.Ticker(yf_ticker).info
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


# ==============================================================
# Metric computation
# ==============================================================

def compute_atr(df: pd.DataFrame, period=14):
    """Average True Range — volatility measure."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    atr_pct = (atr / close.iloc[-1]) * 100  # ATR as % of price
    return atr, atr_pct


def compute_rsi(close: pd.Series, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_divergence(close: pd.Series, rsi: pd.Series, lookback=30):
    """Detect bearish (price ↑, RSI ↓) or bullish (price ↓, RSI ↑) divergence."""
    if len(close) < lookback:
        return None
    price_recent = close.iloc[-lookback:]
    rsi_recent = rsi.iloc[-lookback:]

    price_slope = (price_recent.iloc[-1] - price_recent.iloc[0]) / price_recent.iloc[0]
    rsi_slope = rsi_recent.iloc[-1] - rsi_recent.iloc[0]

    if price_slope > 0.05 and rsi_slope < -5:
        return "bearish"  # Price up, RSI down = bearish divergence (warning)
    if price_slope < -0.05 and rsi_slope > 5:
        return "bullish"  # Price down, RSI up = bullish divergence (opportunity)
    return None


def check_multi_timeframe(daily_df: pd.DataFrame, weekly_df: pd.DataFrame):
    """Check if daily + weekly both bullish (above MA)."""
    daily_bullish = False
    weekly_bullish = False

    if len(daily_df) >= 50:
        daily_bullish = daily_df["Close"].iloc[-1] > daily_df["Close"].rolling(50).mean().iloc[-1]

    if len(weekly_df) >= 20:
        weekly_bullish = weekly_df["Close"].iloc[-1] > weekly_df["Close"].rolling(20).mean().iloc[-1]

    return {
        "daily_bullish": daily_bullish,
        "weekly_bullish": weekly_bullish,
        "aligned": daily_bullish and weekly_bullish,
    }


def compute_metrics(df: pd.DataFrame, weekly_df: pd.DataFrame = None, fundamentals: dict = None):
    close = df["Close"]
    volume = df["Volume"]

    current = close.iloc[-1]
    start_1y = close.iloc[0]
    return_1y = (current / start_1y - 1) * 100

    return_3m = (current / close.iloc[-63] - 1) * 100 if len(close) >= 63 else None
    return_1m = (current / close.iloc[-22] - 1) * 100 if len(close) >= 22 else None
    return_1w = (current / close.iloc[-6] - 1) * 100 if len(close) >= 6 else None

    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

    price_vs_ma20 = (current / ma20 - 1) * 100
    price_vs_ma50 = (current / ma50 - 1) * 100
    price_vs_ma200 = (current / ma200 - 1) * 100 if ma200 else None

    # Higher lows detection
    lows = []
    chunk_size = max(len(close) // 6, 5)
    for i in range(3):
        start = -(i + 1) * chunk_size
        end = -i * chunk_size if i > 0 else None
        chunk = close.iloc[start:end] if end else close.iloc[start:]
        if len(chunk) > 0:
            lows.append(chunk.min())
    higher_lows = len(lows) == 3 and lows[2] < lows[1] < lows[0]

    # Volume trend
    vol_recent = volume.iloc[-30:].mean() if len(volume) >= 30 else volume.mean()
    vol_prior = volume.iloc[-90:-30].mean() if len(volume) >= 90 else volume.mean()
    volume_trend = (vol_recent / vol_prior - 1) * 100 if vol_prior > 0 else 0

    peak = close.max()
    peak_idx = close.idxmax()
    trough = close.min()
    trough_idx = close.idxmin()
    drawdown = (current / peak - 1) * 100  # % below ATH (from 1Y data)
    recovery_from_bottom = (current / trough - 1) * 100 if trough > 0 else 0  # % above 1Y low
    days_from_peak = (close.index[-1] - peak_idx).days if hasattr(peak_idx, 'to_pydatetime') else None
    days_from_trough = (close.index[-1] - trough_idx).days if hasattr(trough_idx, 'to_pydatetime') else None

    rsi_series = compute_rsi(close)
    rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50

    # NEW: ATR volatility
    atr, atr_pct = compute_atr(df)

    # NEW: Divergence
    divergence = detect_divergence(close, rsi_series)

    # NEW: Multi-timeframe
    if weekly_df is not None:
        mtf = check_multi_timeframe(df, weekly_df)
    else:
        mtf = {"daily_bullish": current > ma50, "weekly_bullish": None, "aligned": False}

    metrics = {
        "current": current,
        "return_1y": return_1y,
        "return_3m": return_3m,
        "return_1m": return_1m,
        "return_1w": return_1w,
        "price_vs_ma20": price_vs_ma20,
        "price_vs_ma50": price_vs_ma50,
        "price_vs_ma200": price_vs_ma200,
        "higher_lows": higher_lows,
        "volume_trend": volume_trend,
        "drawdown": drawdown,
        "recovery_from_bottom": recovery_from_bottom,
        "days_from_peak": days_from_peak,
        "days_from_trough": days_from_trough,
        "peak_1y": peak,
        "trough_1y": trough,
        "rsi": rsi,
        "atr_pct": atr_pct,
        "divergence": divergence,
        "mtf_aligned": mtf["aligned"],
        "daily_bullish": mtf["daily_bullish"],
        "weekly_bullish": mtf["weekly_bullish"],
    }

    if fundamentals:
        metrics.update({
            "pe": fundamentals.get("trailingPE"),
            "pbv": fundamentals.get("priceToBook"),
            "roe": fundamentals.get("returnOnEquity"),
            "div_yield": fundamentals.get("dividendYield"),
            "de_ratio": fundamentals.get("debtToEquity"),
            "earnings_growth": fundamentals.get("earningsGrowth"),
            "market_cap": fundamentals.get("marketCap"),
        })

    return metrics


# ==============================================================
# Scoring
# ==============================================================

def score_momentum(m: dict) -> int:
    """Technical momentum score 0-100."""
    score = 0

    # 1Y return (max 20)
    if m["return_1y"] > 50: score += 20
    elif m["return_1y"] > 20: score += 16
    elif m["return_1y"] > 10: score += 12
    elif m["return_1y"] > 0: score += 8
    elif m["return_1y"] > -10: score += 4

    # 3M return (max 20)
    if m["return_3m"] is not None:
        if m["return_3m"] > 20: score += 20
        elif m["return_3m"] > 10: score += 15
        elif m["return_3m"] > 5: score += 10
        elif m["return_3m"] > 0: score += 5

    # 1M return (max 15)
    if m["return_1m"] is not None:
        if m["return_1m"] > 10: score += 15
        elif m["return_1m"] > 5: score += 10
        elif m["return_1m"] > 0: score += 5

    # MA alignment (max 15)
    if m["price_vs_ma20"] > 0: score += 4
    if m["price_vs_ma50"] > 0: score += 5
    if m["price_vs_ma200"] is not None and m["price_vs_ma200"] > 0: score += 6

    # Higher lows (max 10)
    if m["higher_lows"]: score += 10

    # Volume trend (max 5)
    if m["volume_trend"] > 20: score += 5
    elif m["volume_trend"] > 0: score += 3

    # RSI health (max 5)
    if 40 <= m["rsi"] <= 70: score += 5
    elif 30 <= m["rsi"] < 40: score += 3

    # NEW: Multi-timeframe alignment (max 5)
    if m.get("mtf_aligned"): score += 5

    # NEW: Divergence penalty (max -5)
    if m.get("divergence") == "bearish": score -= 5
    elif m.get("divergence") == "bullish": score += 5

    return max(0, min(100, score))


def score_quality(m: dict) -> int:
    """Fundamental quality score 0-100. Returns None if no fundamental data."""
    score = 0
    valid_metrics = 0

    # Valuation — PE (max 20)
    pe = m.get("pe")
    if pe is not None and pe > 0:
        valid_metrics += 1
        if pe < 10: score += 20
        elif pe < 15: score += 15
        elif pe < 20: score += 10
        elif pe < 30: score += 5

    # Valuation — PBV (max 15)
    pbv = m.get("pbv")
    if pbv is not None and pbv > 0:
        valid_metrics += 1
        if pbv < 1: score += 15
        elif pbv < 2: score += 12
        elif pbv < 3: score += 8
        elif pbv < 5: score += 4

    # Profitability — ROE (max 20)
    roe = m.get("roe")
    if roe is not None:
        valid_metrics += 1
        roe_pct = roe * 100
        if roe_pct > 20: score += 20
        elif roe_pct > 15: score += 15
        elif roe_pct > 10: score += 10
        elif roe_pct > 5: score += 5

    # Growth — Earnings growth (max 15)
    eg = m.get("earnings_growth")
    if eg is not None:
        valid_metrics += 1
        eg_pct = eg * 100
        if eg_pct > 20: score += 15
        elif eg_pct > 10: score += 10
        elif eg_pct > 0: score += 5

    # Income — Dividend yield (max 15)
    dy = m.get("div_yield")
    if dy is not None:
        valid_metrics += 1
        dy_pct = dy * 100 if dy < 1 else dy  # Handle both fraction and pct
        if dy_pct > 7: score += 15
        elif dy_pct > 4: score += 10
        elif dy_pct > 2: score += 5

    # Leverage — D/E ratio (max 15)
    de = m.get("de_ratio")
    if de is not None:
        valid_metrics += 1
        if de < 30: score += 15
        elif de < 60: score += 10
        elif de < 100: score += 5

    # Normalize: if <3 metrics available, return None (unreliable)
    if valid_metrics < 3:
        return None

    return min(100, score)


def score_overall(momentum: int, quality) -> int:
    """Combined score: 60% momentum + 40% quality. If no quality data, use momentum only."""
    if quality is None:
        return momentum
    return int(0.6 * momentum + 0.4 * quality)


def verdict(overall: int) -> str:
    if overall >= 75: return "🟢 STRONG BUY"
    if overall >= 60: return "🟢 BUY"
    if overall >= 45: return "🟡 WATCH"
    if overall >= 30: return "🟠 WEAK"
    return "🔴 AVOID"


def classify_pattern(m: dict) -> dict:
    """Classify price pattern: Bullish Momentum / Recovery Play / Downtrend / Sideways."""
    ret_1y = m.get("return_1y", 0)
    ret_3m = m.get("return_3m", 0) or 0
    ret_1m = m.get("return_1m", 0) or 0
    drawdown = m.get("drawdown", 0)
    recovery = m.get("recovery_from_bottom", 0)
    rsi = m.get("rsi", 50)
    hl = m.get("higher_lows", False)
    mtf = m.get("mtf_aligned", False)

    # BULLISH MOMENTUM: 1Y strong up, dekat ATH, sehat (mtf OR higher_lows)
    if ret_1y > 20 and drawdown > -15 and ret_1m > -5 and rsi < 75 and (mtf or hl):
        pullback = " (small pullback dalam uptrend)" if ret_1m < 0 else ""
        return {
            "pattern": "🚀 BULLISH MOMENTUM",
            "desc": f"Actively rising. Dekat ATH ({drawdown:+.1f}%), 1M {ret_1m:+.1f}%{pullback}. RSI healthy {rsi:.0f}.",
            "action": "Prime setup. Good entry area (RSI 40-60 = ideal).",
            "color": "green",
        }

    # RECOVERY PLAY: Big drawdown but bouncing
    if drawdown < -25 and ret_1m > 10 and ret_3m > -20 and recovery > 30:
        return {
            "pattern": "🔄 RECOVERY PLAY",
            "desc": f"Bouncing dari bottom (+{recovery:.0f}%). Masih -{abs(drawdown):.0f}% dari ATH. Reversal in progress.",
            "action": "Potential 2nd wave. Wait higher lows confirmation. Position kecil, stop ketat.",
            "color": "yellow",
        }

    # PARABOLIC / OVERBOUGHT: Rally massive + RSI extreme
    if ret_1y > 100 and rsi > 80:
        return {
            "pattern": "⚠️ PARABOLIC / OVERBOUGHT",
            "desc": f"Rally massive 1Y {ret_1y:+.0f}%. RSI {rsi:.0f} extreme. Post-parabolic risk.",
            "action": "AVOID new entry. Kalau punya, take profit / trailing stop. Wait koreksi.",
            "color": "red",
        }

    # DEAD CAT BOUNCE: Bounce short-term tapi 3M masih bearish
    if ret_1m > 15 and ret_3m < -15 and not hl:
        return {
            "pattern": "🐈 DEAD CAT BOUNCE?",
            "desc": f"Bounce 1M {ret_1m:+.1f}% tapi 3M {ret_3m:+.1f}%. No higher lows confirmation.",
            "action": "Suspicious. Wait higher lows terbentuk. Skip untuk sekarang.",
            "color": "orange",
        }

    # DOWNTREND: Everything bearish
    if ret_1y < -10 and ret_3m < 0 and ret_1m < 0:
        return {
            "pattern": "📉 DOWNTREND",
            "desc": f"Bearish di semua timeframe. 1Y {ret_1y:+.1f}% / 3M {ret_3m:+.1f}% / 1M {ret_1m:+.1f}%.",
            "action": "AVOID. Wait reversal signal (higher lows + volume return).",
            "color": "red",
        }

    # SIDEWAYS / CONSOLIDATION
    if abs(ret_1m) < 5 and abs(ret_3m) < 10:
        return {
            "pattern": "➡️ SIDEWAYS / CONSOLIDATION",
            "desc": f"Range-bound. 1M {ret_1m:+.1f}%, 3M {ret_3m:+.1f}%. Belum ada breakout signal.",
            "action": "Wait breakout arah mana. Set alert di resistance/support.",
            "color": "gray",
        }

    # UPTREND EARLY: Ada momentum tapi belum kuat
    if ret_1m > 5 and ret_3m > 0 and ret_1y > 0:
        return {
            "pattern": "📈 UPTREND EARLY",
            "desc": f"Uptrend baru forming. 1M {ret_1m:+.1f}%, MTF {'aligned' if mtf else 'not aligned'}.",
            "action": "Monitor. Kalau MTF confirm + score naik ≥60, potential entry.",
            "color": "green",
        }

    return {
        "pattern": "❓ MIXED / UNCLEAR",
        "desc": f"Signal campuran. 1Y {ret_1y:+.1f}%, 3M {ret_3m:+.1f}%, 1M {ret_1m:+.1f}%.",
        "action": "Wait clarity. Cek indicator lain (volume, MA alignment).",
        "color": "gray",
    }


def volatility_label(atr_pct: float) -> str:
    if atr_pct > 5: return "🔥 HIGH"
    if atr_pct > 3: return "⚡ MED"
    return "🧊 LOW"


# ==============================================================
# Main
# ==============================================================

def analyze_ticker(ticker: str, category: str, verbose=False):
    daily = fetch_history(ticker, period="1y", interval="1d")
    if daily is None:
        return None

    weekly = fetch_history(ticker, period="2y", interval="1wk")
    fund = fetch_fundamentals(ticker)

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


def main():
    parser = argparse.ArgumentParser(description="IDX Momentum + Quality Screener v2")
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--min-score", type=int, default=0)
    parser.add_argument("--sort", default="overall", choices=["overall", "momentum", "quality", "return_1y", "return_3m", "return_1m"])
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--fundamental-only", action="store_true", help="Only show tickers with fundamental data")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    tickers = load_tickers(TICKERS_FILE)
    if args.ticker:
        tickers = [(t, c) for t, c in tickers if t.upper() == args.ticker.upper()]
    if args.category:
        tickers = [(t, c) for t, c in tickers if args.category.lower() in c.lower()]

    print(f"Screening {len(tickers)} tickers (this takes ~1-2 min for fundamentals)...\n", file=sys.stderr)

    results = []
    for i, (ticker, category) in enumerate(tickers):
        if args.verbose:
            print(f"  [{i+1}/{len(tickers)}] {ticker}...", file=sys.stderr)
        r = analyze_ticker(ticker, category, verbose=args.verbose)
        if r is None:
            continue
        m = r["metrics"]
        results.append({
            "Ticker": ticker,
            "Category": category,
            "Price": f"{m['current']:,.0f}",
            "1Y%": f"{m['return_1y']:+.1f}",
            "3M%": f"{m['return_3m']:+.1f}" if m['return_3m'] is not None else "-",
            "1M%": f"{m['return_1m']:+.1f}" if m['return_1m'] is not None else "-",
            "vsMA50%": f"{m['price_vs_ma50']:+.1f}",
            "HL": "✓" if m['higher_lows'] else "✗",
            "MTF": "✓" if m['mtf_aligned'] else "✗",
            "Div": {"bearish": "⬇", "bullish": "⬆", None: "-"}[m.get('divergence')],
            "RSI": f"{m['rsi']:.0f}",
            "ATR%": f"{m['atr_pct']:.1f}",
            "Vol": volatility_label(m['atr_pct']),
            "PER": f"{m.get('pe'):.1f}" if m.get('pe') else "-",
            "PBV": f"{m.get('pbv'):.1f}" if m.get('pbv') else "-",
            "ROE%": f"{(m.get('roe')*100 if abs(m.get('roe')) < 1 else m.get('roe')):.0f}" if m.get('roe') is not None else "-",
            "DivY%": f"{(m.get('div_yield')*100 if m.get('div_yield') < 1 else m.get('div_yield')):.1f}" if m.get('div_yield') is not None else "-",
            "Mom": r["momentum"],
            "Qual": r["quality"] if r["quality"] is not None else "N/A",
            "Score": r["overall"],
            "Verdict": r["verdict"],
            "_sort_overall": r["overall"],
            "_sort_momentum": r["momentum"],
            "_sort_quality": r["quality"] if r["quality"] is not None else -1,
            "_sort_1y": m['return_1y'],
            "_sort_3m": m['return_3m'] if m['return_3m'] is not None else -999,
            "_sort_1m": m['return_1m'] if m['return_1m'] is not None else -999,
            "_has_quality": r["quality"] is not None,
        })

    if args.fundamental_only:
        results = [r for r in results if r["_has_quality"]]

    results = [r for r in results if r["_sort_overall"] >= args.min_score]

    sort_key = {
        "overall": "_sort_overall",
        "momentum": "_sort_momentum",
        "quality": "_sort_quality",
        "return_1y": "_sort_1y",
        "return_3m": "_sort_3m",
        "return_1m": "_sort_1m",
    }[args.sort]
    results.sort(key=lambda r: r[sort_key], reverse=True)

    if args.top:
        results = results[:args.top]

    display = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    if not display:
        print("No results.", file=sys.stderr)
        sys.exit(0)

    print(tabulate(display, headers="keys", tablefmt="simple", stralign="right"))
    print(f"\nTotal: {len(display)} tickers | Legend: HL=Higher Lows, MTF=Multi-Timeframe Aligned, Div=RSI Divergence, ATR=Volatility %")


if __name__ == "__main__":
    main()
