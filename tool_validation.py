#!/usr/bin/env python3
"""
Tool Validation Framework
Measures confidence level of screener + financial + valuation as advisor.

Tests:
1. Screener predictive power (backtest)
2. Financial verdict correlation with actual returns
3. Combined signal (BUY NOW / VALUE TRAP / etc) accuracy
4. Fair value reversion test
5. Sanity check known stocks
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

from screener import analyze_ticker, load_tickers, TICKERS_FILE
from financials import full_financial_summary, interpret_financials
from valuation import full_valuation
from backtest import backtest_ticker, fetch_long_history


BASE_DIR = Path(__file__).parent


def test_1_screener_backtest(min_scores=[50, 60, 70, 75], hold_days=60):
    """Test 1: How does screener perform at different thresholds?"""
    print("\n" + "=" * 70)
    print("TEST 1: Screener Predictive Power (Backtest)")
    print("=" * 70)

    tickers = [t for t, _ in load_tickers(TICKERS_FILE)][:30]  # sample 30 tickers
    results_by_threshold = {}

    for min_score in min_scores:
        all_trades = []
        for ticker in tickers:
            df = fetch_long_history(ticker)
            if df is None:
                continue
            trades = backtest_ticker(ticker, df, min_score=min_score, hold_days=hold_days, step_days=30)
            all_trades.extend(trades)

        if not all_trades:
            results_by_threshold[min_score] = {"trades": 0}
            continue

        df_trades = pd.DataFrame(all_trades)
        wins = (df_trades["return_pct"] > 0).sum()
        total = len(df_trades)

        results_by_threshold[min_score] = {
            "trades": total,
            "win_rate": wins / total * 100,
            "avg_return": df_trades["return_pct"].mean(),
            "median": df_trades["return_pct"].median(),
            "best": df_trades["return_pct"].max(),
            "worst": df_trades["return_pct"].min(),
        }

    # Print results
    print(f"\n{'Threshold':<12}{'Trades':<10}{'Win Rate':<12}{'Avg Return':<14}{'Median':<12}")
    print("-" * 70)
    for score, r in results_by_threshold.items():
        if r.get("trades", 0) == 0:
            print(f"Score ≥{score:<8}0 trades")
            continue
        print(f"Score ≥{score:<8}{r['trades']:<10}{r['win_rate']:.1f}%{'':<7}{r['avg_return']:+.2f}%{'':<8}{r['median']:+.2f}%")

    # Confidence assessment
    print("\n--- Confidence Assessment ---")
    high_score = results_by_threshold.get(75, {})
    if high_score.get("win_rate", 0) > 55 and high_score.get("avg_return", 0) > 15:
        confidence_1 = "HIGH (system predicts winners at score ≥75)"
    elif high_score.get("win_rate", 0) > 50 and high_score.get("avg_return", 0) > 10:
        confidence_1 = "MEDIUM (edge exists but not extreme)"
    else:
        confidence_1 = "LOW (signal not consistent)"
    print(f"Score threshold ≥75: {confidence_1}")

    return results_by_threshold


def test_2_financial_verdict_correlation(sample_size=15):
    """Test 2: Does financial verdict correlate with 1Y return?"""
    print("\n" + "=" * 70)
    print("TEST 2: Financial Verdict vs Actual 1Y Return")
    print("=" * 70)

    tickers = [t for t, _ in load_tickers(TICKERS_FILE)][:sample_size]
    results = []

    for ticker in tickers:
        try:
            # Get financial verdict
            fs = full_financial_summary(ticker)
            interp = interpret_financials(fs)
            verdict = interp["verdict"]

            # Get 1Y return
            df = fetch_long_history(ticker, period="1y")
            if df is None or len(df) < 200:
                continue
            ret_1y = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

            results.append({
                "ticker": ticker,
                "verdict": verdict.split(" — ")[0],
                "return_1y": ret_1y,
            })
        except Exception:
            continue

    df_r = pd.DataFrame(results)

    if df_r.empty or "verdict" not in df_r.columns:
        print("⚠️ No financial data available for sample")
        return df_r

    # Group by verdict
    print(f"\n{'Verdict':<35}{'N':<5}{'Avg 1Y Return':<15}{'Median':<10}")
    print("-" * 70)
    for v, group in df_r.groupby("verdict"):
        print(f"{v:<35}{len(group):<5}{group['return_1y'].mean():+.1f}%{'':<7}{group['return_1y'].median():+.1f}%")

    # Correlation check
    strong = df_r[df_r["verdict"].str.contains("STRONG")]
    healthy = df_r[df_r["verdict"].str.contains("HEALTHY")]
    risky = df_r[df_r["verdict"].str.contains("RISKY|HIGH RISK")]

    print("\n--- Confidence Assessment ---")
    if not strong.empty and not risky.empty:
        strong_avg = strong["return_1y"].mean()
        risky_avg = risky["return_1y"].mean()
        gap = strong_avg - risky_avg
        print(f"STRONG BUY avg: {strong_avg:+.1f}% | RISKY avg: {risky_avg:+.1f}% | Gap: {gap:+.1f}%")
        if gap > 20:
            print("✅ HIGH confidence — verdict correlates with returns")
        elif gap > 10:
            print("🟡 MEDIUM confidence — some correlation")
        else:
            print("🔴 LOW confidence — verdict doesn't predict returns well")
    else:
        print("⚠️ Insufficient sample for STRONG vs RISKY comparison")

    return df_r


def test_3_sanity_check_known():
    """Test 3: Sanity check known stocks with expected outcomes."""
    print("\n" + "=" * 70)
    print("TEST 3: Sanity Check Known Stocks")
    print("=" * 70)

    expected = {
        "BBCA": {"expect_fin": "HEALTHY", "reason": "Bank premium blue chip"},
        "BBRI": {"expect_fin": "HEALTHY", "reason": "Bank BUMN dividen tinggi"},
        "TLKM": {"expect_fin": "HEALTHY", "reason": "Telco monopoli"},
        "UNVR": {"expect_fin": "HEALTHY_or_MIXED", "reason": "Consumer staples classic (mungkin margin turun)"},
        "ICBP": {"expect_fin": "HEALTHY", "reason": "Consumer staples (Indomie)"},
        "MAPI": {"expect_fin": "HEALTHY", "reason": "Retail brand growth"},
        "DSSA": {"expect_fin": "RISKY", "reason": "MSCI removal, laba turun"},
        "IKAN": {"expect_fin": "RISKY", "reason": "Small cap, no dividen, insider sell"},
    }

    correct = 0
    total = 0
    print(f"\n{'Ticker':<10}{'Expected':<25}{'Actual':<30}{'Match':<10}")
    print("-" * 80)

    for ticker, meta in expected.items():
        try:
            fs = full_financial_summary(ticker)
            interp = interpret_financials(fs)
            actual = interp["verdict"].split(" — ")[0]

            expected_v = meta["expect_fin"]
            if "or" in expected_v:
                options = expected_v.split("_or_")
                match = any(opt in actual for opt in options)
            else:
                match = expected_v in actual

            symbol = "✅" if match else "❌"
            if match:
                correct += 1
            total += 1
            print(f"{ticker:<10}{expected_v:<25}{actual[:28]:<30}{symbol:<10}")
        except Exception as e:
            print(f"{ticker:<10}(error: {e})")

    accuracy = correct / total * 100 if total else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.0f}%")

    if accuracy >= 75:
        print("✅ HIGH confidence — tool aligns with known expectations")
    elif accuracy >= 50:
        print("🟡 MEDIUM confidence — some misalignment")
    else:
        print("🔴 LOW confidence — significant misalignment")

    return accuracy


def test_4_fair_value_reversion(sample=10):
    """Test 4: Do UNDERVALUED stocks tend to rise, OVERVALUED to fall?"""
    print("\n" + "=" * 70)
    print("TEST 4: Fair Value Verdict vs Recent Price Movement")
    print("=" * 70)

    tickers = [t for t, _ in load_tickers(TICKERS_FILE)][:sample]
    results = []

    for ticker in tickers:
        try:
            v = full_valuation(ticker)
            s = v.get("summary")
            if not s:
                continue

            df = fetch_long_history(ticker, period="6mo")
            if df is None or len(df) < 30:
                continue
            ret_6m = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

            results.append({
                "ticker": ticker,
                "verdict": s["verdict"].split(" — ")[0],
                "upside_est": s["upside_pct"],
                "actual_6m": ret_6m,
            })
        except Exception:
            continue

    df_r = pd.DataFrame(results)
    if df_r.empty:
        print("No valuation data available")
        return

    print(f"\n{'Ticker':<10}{'Verdict':<30}{'Upside Est':<15}{'Actual 6M':<15}")
    print("-" * 70)
    for _, r in df_r.iterrows():
        print(f"{r['ticker']:<10}{r['verdict'][:28]:<30}{r['upside_est']:+.1f}%{'':<8}{r['actual_6m']:+.1f}%")

    # Correlation
    if len(df_r) > 3:
        corr = df_r["upside_est"].corr(df_r["actual_6m"])
        print(f"\nCorrelation (upside_est vs actual_6m): {corr:.2f}")
        if corr > 0.5:
            print("✅ HIGH — Fair value estimates align with actual moves")
        elif corr > 0.2:
            print("🟡 MEDIUM — Weak positive correlation")
        elif corr > -0.2:
            print("⚠️ NONE — Fair value not predictive")
        else:
            print("🔴 NEGATIVE — Fair value inverse to actual (mean reversion delayed)")

    return df_r


def main():
    print("\n" + "=" * 70)
    print("🧪 TOOL VALIDATION — Confidence Level Assessment")
    print("=" * 70)
    print("Testing screener + financial verdict + fair value against real data...\n")

    # Run all tests
    r1 = test_1_screener_backtest()
    r2 = test_2_financial_verdict_correlation()
    r3 = test_3_sanity_check_known()
    r4 = test_4_fair_value_reversion()

    # Overall summary
    print("\n" + "=" * 70)
    print("📊 OVERALL CONFIDENCE ASSESSMENT")
    print("=" * 70)

    print("""
Summary:
- Test 1 (Screener Backtest): Validates score threshold effectiveness
- Test 2 (Financial Verdict): Correlation with actual 1Y returns
- Test 3 (Sanity Check): Alignment with known stock quality
- Test 4 (Fair Value): Predictive power of valuation methods

Use combined signals (BUY NOW / VALUE TRAP / etc) for highest confidence.
Always cross-check with:
  1. Chart pattern validation
  2. Recent news (aksi korporasi)
  3. Sector rotation context
  4. Personal risk tolerance
""")


if __name__ == "__main__":
    main()
