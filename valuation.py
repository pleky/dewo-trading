#!/usr/bin/env python3
"""
Fair value / intrinsic value calculator.
Multiple valuation methods: PE, PBV, Graham, DDM, DCF simplified.
"""

import math
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent


def fetch_valuation_inputs(ticker: str, market: str = "IDX") -> dict:
    """Fetch inputs needed for valuation."""
    yf_ticker = f"{ticker}.JK" if market == "IDX" else ticker
    try:
        t = yf.Ticker(yf_ticker)
        info = t.info

        return {
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "eps_trailing": info.get("trailingEps"),
            "eps_forward": info.get("forwardEps"),
            "book_value": info.get("bookValue"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "pbv": info.get("priceToBook"),
            "dividend_rate": info.get("dividendRate"),
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            "roe": info.get("returnOnEquity"),
            "earnings_growth": info.get("earningsGrowth"),
            "free_cash_flow": info.get("freeCashflow"),
            "beta": info.get("beta"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
        }
    except Exception:
        return {}


def calc_pe_valuation(eps: float, target_pe: float) -> float:
    """Simple PE multiple valuation."""
    if eps is None or target_pe is None or eps <= 0:
        return None
    return eps * target_pe


def calc_pbv_valuation(book_value_per_share: float, target_pbv: float) -> float:
    """PBV multiple valuation."""
    if book_value_per_share is None or target_pbv is None:
        return None
    return book_value_per_share * target_pbv


def calc_graham_number(eps: float, book_value: float) -> float:
    """Graham Number = sqrt(22.5 × EPS × BVPS)
    Conservative fair value for defensive stocks."""
    if eps is None or book_value is None or eps <= 0 or book_value <= 0:
        return None
    return math.sqrt(22.5 * eps * book_value)


def calc_ddm(dividend: float, required_return: float, growth_rate: float) -> float:
    """Gordon Growth Model / Dividend Discount Model.
    Fair Price = Div_next_year / (r - g)"""
    if dividend is None or dividend <= 0:
        return None
    if required_return is None or growth_rate is None:
        return None
    if required_return <= growth_rate:
        return None  # Invalid - denominator zero or negative
    next_div = dividend * (1 + growth_rate)
    return next_div / (required_return - growth_rate)


def calc_dcf_simple(free_cash_flow: float, shares_outstanding: float,
                    growth_rate: float = 0.05, discount_rate: float = 0.10,
                    terminal_multiple: float = 15) -> float:
    """Simplified DCF using perpetuity growth.
    FV = FCF per share × Multiplier
    Multiplier = 1 / (discount_rate - growth_rate)"""
    if free_cash_flow is None or shares_outstanding is None or shares_outstanding <= 0:
        return None
    if discount_rate <= growth_rate:
        return None

    fcf_per_share = free_cash_flow / shares_outstanding
    if fcf_per_share <= 0:
        return None

    multiplier = 1 / (discount_rate - growth_rate)
    return fcf_per_share * multiplier


def calc_peg_valuation(eps: float, growth_rate_pct: float, target_peg: float = 1.0) -> float:
    """PEG-based fair value. Peter Lynch style.
    Fair PE = Growth Rate × Target PEG
    Fair Price = EPS × Fair PE"""
    if eps is None or eps <= 0 or growth_rate_pct is None or growth_rate_pct <= 0:
        return None
    fair_pe = growth_rate_pct * target_peg
    return eps * fair_pe


def full_valuation(ticker: str, market: str = "IDX",
                   custom_target_pe: float = None, custom_target_pbv: float = None,
                   custom_growth: float = None, custom_discount: float = 0.10) -> dict:
    """Compute all valuation methods for a ticker."""
    inputs = fetch_valuation_inputs(ticker, market)
    if not inputs:
        return {"error": "Failed to fetch data"}

    current = inputs.get("current_price")
    eps = inputs.get("eps_trailing") or inputs.get("eps_forward")
    bvps = inputs.get("book_value")
    dividend = inputs.get("dividend_rate")
    fcf = inputs.get("free_cash_flow")
    shares = inputs.get("shares_outstanding")

    # Growth rate estimate (default 5% if not available)
    growth = custom_growth
    if growth is None:
        eg = inputs.get("earnings_growth")
        growth = eg if (eg is not None and 0 <= eg <= 0.5) else 0.05

    # Target multipliers (industry benchmarks)
    # Default: PE 15 (S&P avg), PBV 2 (fair for most)
    target_pe = custom_target_pe or 15
    target_pbv = custom_target_pbv or 2

    valuations = {}

    # 1. PE Multiple
    pe_val = calc_pe_valuation(eps, target_pe)
    if pe_val:
        valuations["PE Multiple"] = {
            "fair_value": pe_val,
            "formula": f"EPS ({eps:.0f}) × Target PE ({target_pe})",
            "note": f"Assumes fair PE {target_pe}x. Conservative benchmark.",
        }

    # 2. PBV Multiple
    pbv_val = calc_pbv_valuation(bvps, target_pbv)
    if pbv_val:
        valuations["PBV Multiple"] = {
            "fair_value": pbv_val,
            "formula": f"BVPS ({bvps:.0f}) × Target PBV ({target_pbv})",
            "note": f"Book value based. Good for asset-heavy business.",
        }

    # 3. Graham Number
    graham_val = calc_graham_number(eps, bvps)
    if graham_val:
        valuations["Graham Number"] = {
            "fair_value": graham_val,
            "formula": f"√(22.5 × EPS × BVPS) = √(22.5 × {eps:.0f} × {bvps:.0f})",
            "note": "Benjamin Graham's conservative formula. For defensive value investors.",
        }

    # 4. Dividend Discount Model
    if dividend and dividend > 0:
        ddm_val = calc_ddm(dividend, custom_discount, growth)
        if ddm_val:
            valuations["Dividend Discount"] = {
                "fair_value": ddm_val,
                "formula": f"Div ({dividend:.0f}) × (1+{growth:.2%}) / ({custom_discount:.0%} - {growth:.2%})",
                "note": f"For dividend-paying stocks. Requires return >{custom_discount:.0%}.",
            }

    # 5. PEG Valuation
    if growth and growth > 0:
        growth_pct = growth * 100 if growth < 1 else growth
        peg_val = calc_peg_valuation(eps, growth_pct)
        if peg_val:
            valuations["PEG (Peter Lynch)"] = {
                "fair_value": peg_val,
                "formula": f"EPS × (Growth% × 1.0) = {eps:.0f} × {growth_pct:.1f}",
                "note": "Fair PE = growth rate. Peter Lynch style.",
            }

    # 6. DCF Simplified
    dcf_val = calc_dcf_simple(fcf, shares, growth, custom_discount)
    if dcf_val:
        valuations["DCF Simplified"] = {
            "fair_value": dcf_val,
            "formula": f"FCF per share × 1/({custom_discount:.0%} - {growth:.2%})",
            "note": f"Perpetuity growth model. WACC {custom_discount:.0%}, Growth {growth:.2%}.",
        }

    # Compute average fair value + upside/downside
    fair_values = [v["fair_value"] for v in valuations.values() if v["fair_value"] is not None and v["fair_value"] > 0]

    if fair_values and current:
        avg_fair = sum(fair_values) / len(fair_values)
        median_fair = sorted(fair_values)[len(fair_values) // 2]
        upside = (avg_fair - current) / current * 100
        median_upside = (median_fair - current) / current * 100

        # Interpretation
        if upside > 30:
            verdict = "🟢 UNDERVALUED — Significant upside potential."
        elif upside > 10:
            verdict = "🟢 SLIGHTLY UNDERVALUED — Moderate upside."
        elif upside > -10:
            verdict = "🟡 FAIRLY PRICED — Trading near fair value."
        elif upside > -30:
            verdict = "🟠 SLIGHTLY OVERVALUED — Modest downside risk."
        else:
            verdict = "🔴 OVERVALUED — High downside risk."

        summary = {
            "current_price": current,
            "avg_fair_value": avg_fair,
            "median_fair_value": median_fair,
            "upside_pct": upside,
            "median_upside_pct": median_upside,
            "verdict": verdict,
            "n_methods": len(fair_values),
        }
    else:
        summary = None

    return {
        "inputs": inputs,
        "valuations": valuations,
        "summary": summary,
    }
