#!/usr/bin/env python3
"""
Backtest Engine — validate momentum scoring system.

Approach:
- Fetch 2-year OHLCV per ticker
- Walk-forward: at each historical date, compute momentum score using trailing data
- If score >= threshold, simulate buy, record forward N-day return
- Aggregate: win rate, avg return, drawdown, benchmark comparison

Note: Uses momentum score only (fundamental data not available historically).
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from screener import compute_metrics, load_tickers, score_momentum, TICKERS_FILE

BASE_DIR = Path(__file__).parent


def fetch_long_history(ticker: str, period="2y"):
    """Fetch 2-year daily data."""
    yf_ticker = f"{ticker}.JK"
    try:
        data = yf.Ticker(yf_ticker).history(period=period, interval="1d", auto_adjust=False)
        if data.empty or len(data) < 260:  # ~1 year minimum
            return None
        return data
    except Exception:
        return None


def compute_score_at_date(df: pd.DataFrame, target_date_idx: int) -> int:
    """Compute momentum score using data up to target_date_idx (walk-forward)."""
    if target_date_idx < 220:  # Need enough history for MA200 etc
        return None

    slice_df = df.iloc[:target_date_idx + 1]
    try:
        m = compute_metrics(slice_df)
        return score_momentum(m)
    except Exception:
        return None


def backtest_ticker(ticker: str, df: pd.DataFrame, min_score: int = 60,
                     hold_days: int = 60, step_days: int = 30) -> list:
    """Walk-forward backtest for one ticker.

    Returns list of trade records: {entry_date, exit_date, entry_price, exit_price, score, return_pct}
    """
    trades = []
    close = df["Close"]
    n = len(df)

    # Walk forward monthly
    for entry_idx in range(220, n - hold_days, step_days):
        score = compute_score_at_date(df, entry_idx)
        if score is None or score < min_score:
            continue

        exit_idx = entry_idx + hold_days
        entry_price = close.iloc[entry_idx]
        exit_price = close.iloc[exit_idx]

        # Max drawdown during hold period (worst point vs entry)
        hold_slice = close.iloc[entry_idx:exit_idx + 1]
        max_dd = (hold_slice.min() - entry_price) / entry_price * 100

        # Max upside during hold
        max_up = (hold_slice.max() - entry_price) / entry_price * 100

        trades.append({
            "ticker": ticker,
            "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
            "exit_date": df.index[exit_idx].strftime("%Y-%m-%d"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "score": score,
            "return_pct": (exit_price / entry_price - 1) * 100,
            "max_drawdown_pct": max_dd,
            "max_upside_pct": max_up,
        })

    return trades


def run_backtest(tickers: list = None, min_score: int = 60, hold_days: int = 60,
                  step_days: int = 30, progress_callback=None) -> pd.DataFrame:
    """Run backtest across multiple tickers."""
    if tickers is None:
        tickers = [t for t, _ in load_tickers(TICKERS_FILE)]

    all_trades = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i + 1, total, ticker)

        df = fetch_long_history(ticker)
        if df is None:
            continue

        trades = backtest_ticker(ticker, df, min_score=min_score,
                                 hold_days=hold_days, step_days=step_days)
        all_trades.extend(trades)

    return pd.DataFrame(all_trades)


def summarize_backtest(trades_df: pd.DataFrame) -> dict:
    """Compute aggregate metrics."""
    if trades_df.empty:
        return {"total_trades": 0}

    returns = trades_df["return_pct"]

    win_trades = trades_df[returns > 0]
    loss_trades = trades_df[returns <= 0]

    total = len(trades_df)
    wins = len(win_trades)
    win_rate = wins / total * 100

    avg_return = returns.mean()
    median_return = returns.median()

    avg_win = win_trades["return_pct"].mean() if not win_trades.empty else 0
    avg_loss = loss_trades["return_pct"].mean() if not loss_trades.empty else 0

    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Compound return (if trades run sequentially, approximation)
    total_return_compound = (1 + returns / 100).prod() - 1
    total_return_compound *= 100

    # Drawdown metrics
    worst_trade = returns.min()
    best_trade = returns.max()

    # Volatility
    std_return = returns.std()

    # Sharpe (annualized approx, assuming 60-day hold, 6 trades/year)
    sharpe = (avg_return / std_return) * np.sqrt(6) if std_return > 0 else 0

    return {
        "total_trades": total,
        "winning_trades": wins,
        "losing_trades": len(loss_trades),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "median_return": median_return,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "risk_reward_ratio": risk_reward,
        "compound_return": total_return_compound,
        "worst_trade": worst_trade,
        "best_trade": best_trade,
        "std_return": std_return,
        "sharpe_approx": sharpe,
    }


def benchmark_ihsg(hold_days: int = 60, step_days: int = 30) -> pd.DataFrame:
    """Compare with buy-and-hold IHSG for same periods."""
    ihsg = yf.Ticker("^JKSE").history(period="2y", interval="1d")
    if ihsg.empty:
        return pd.DataFrame()

    close = ihsg["Close"]
    trades = []
    for entry_idx in range(0, len(ihsg) - hold_days, step_days):
        exit_idx = entry_idx + hold_days
        entry_price = close.iloc[entry_idx]
        exit_price = close.iloc[exit_idx]
        trades.append({
            "entry_date": ihsg.index[entry_idx].strftime("%Y-%m-%d"),
            "exit_date": ihsg.index[exit_idx].strftime("%Y-%m-%d"),
            "return_pct": (exit_price / entry_price - 1) * 100,
        })
    return pd.DataFrame(trades)
