#!/usr/bin/env python3
"""
Financial statement summary module.
Fetches income statement, balance sheet, cash flow via yfinance.
Computes YoY/QoQ growth, margins, ratios.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent


def fetch_financials(ticker: str, market: str = "IDX") -> dict:
    """Fetch financial statements. market='IDX' appends .JK, 'US' uses raw ticker."""
    yf_ticker = f"{ticker}.JK" if market == "IDX" else ticker
    try:
        t = yf.Ticker(yf_ticker)
        return {
            "income_annual": t.income_stmt if hasattr(t, "income_stmt") else pd.DataFrame(),
            "income_quarterly": t.quarterly_income_stmt if hasattr(t, "quarterly_income_stmt") else pd.DataFrame(),
            "balance_annual": t.balance_sheet if hasattr(t, "balance_sheet") else pd.DataFrame(),
            "cashflow_annual": t.cashflow if hasattr(t, "cashflow") else pd.DataFrame(),
        }
    except Exception:
        return {}


def get_metric(df, metric_name, period_idx=0):
    if df is None or df.empty:
        return None
    if metric_name not in df.index:
        return None
    try:
        val = df.iloc[df.index.get_loc(metric_name), period_idx]
        return float(val) if pd.notna(val) else None
    except (IndexError, ValueError, TypeError):
        return None


def summarize_income(income_df):
    if income_df is None or income_df.empty:
        return []
    periods = []
    for i, col_date in enumerate(income_df.columns):
        row = {
            "period": col_date.strftime("%Y-%m-%d") if hasattr(col_date, "strftime") else str(col_date),
            "revenue": get_metric(income_df, "Total Revenue", i),
            "gross_profit": get_metric(income_df, "Gross Profit", i),
            "operating_income": get_metric(income_df, "Operating Income", i),
            "net_income": get_metric(income_df, "Net Income", i),
            "ebitda": get_metric(income_df, "EBITDA", i),
            "eps": get_metric(income_df, "Diluted EPS", i) or get_metric(income_df, "Basic EPS", i),
        }
        if row["revenue"] and row["revenue"] != 0:
            row["gross_margin"] = (row["gross_profit"] / row["revenue"] * 100) if row["gross_profit"] else None
            row["net_margin"] = (row["net_income"] / row["revenue"] * 100) if row["net_income"] else None
        else:
            row["gross_margin"] = None
            row["net_margin"] = None
        periods.append(row)
    return periods


def compute_growth(current, previous):
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / abs(previous) * 100


def summarize_balance(balance_df):
    if balance_df is None or balance_df.empty:
        return {}
    return {
        "period": balance_df.columns[0].strftime("%Y-%m-%d") if hasattr(balance_df.columns[0], "strftime") else str(balance_df.columns[0]),
        "total_assets": get_metric(balance_df, "Total Assets", 0),
        "total_liab": get_metric(balance_df, "Total Liabilities Net Minority Interest", 0),
        "total_equity": get_metric(balance_df, "Stockholders Equity", 0) or get_metric(balance_df, "Total Equity Gross Minority Interest", 0),
        "cash": get_metric(balance_df, "Cash And Cash Equivalents", 0) or get_metric(balance_df, "Cash Cash Equivalents And Short Term Investments", 0),
        "total_debt": get_metric(balance_df, "Total Debt", 0),
        "long_term_debt": get_metric(balance_df, "Long Term Debt", 0),
    }


def summarize_cashflow(cashflow_df):
    if cashflow_df is None or cashflow_df.empty:
        return {}
    return {
        "period": cashflow_df.columns[0].strftime("%Y-%m-%d") if hasattr(cashflow_df.columns[0], "strftime") else str(cashflow_df.columns[0]),
        "operating_cash": get_metric(cashflow_df, "Operating Cash Flow", 0),
        "investing_cash": get_metric(cashflow_df, "Investing Cash Flow", 0),
        "financing_cash": get_metric(cashflow_df, "Financing Cash Flow", 0),
        "free_cash_flow": get_metric(cashflow_df, "Free Cash Flow", 0),
        "capex": get_metric(cashflow_df, "Capital Expenditure", 0),
    }


def format_large(val, currency="Rp"):
    if val is None:
        return "-"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}{currency} {abs_val/1e12:.2f} T"
    if abs_val >= 1e9:
        return f"{sign}{currency} {abs_val/1e9:.2f} B"
    if abs_val >= 1e6:
        return f"{sign}{currency} {abs_val/1e6:.2f} M"
    if abs_val >= 1e3:
        return f"{sign}{currency} {abs_val/1e3:.2f} K"
    return f"{sign}{currency} {abs_val:.0f}"


def interpret_financials(summary: dict) -> dict:
    """Rule-based interpretation of financial data.
    Returns dict with narrative sections + flags + verdict.
    """
    income_annual = summary.get("income_annual", [])
    income_q = summary.get("income_quarterly", [])
    balance = summary.get("balance", {})
    cashflow = summary.get("cashflow", {})

    green_flags = []
    red_flags = []
    yellow_flags = []
    narratives = {}

    # ==========================================
    # 1. REVENUE GROWTH STORY (Annual)
    # ==========================================
    if len(income_annual) >= 3:
        rev_growths = [p.get("revenue_yoy") for p in income_annual[:4] if p.get("revenue_yoy") is not None]
        if rev_growths:
            avg_growth = sum(rev_growths) / len(rev_growths)
            latest_growth = rev_growths[0] if rev_growths else 0

            if avg_growth > 20:
                narratives["revenue"] = f"🚀 Revenue growth **SANGAT KUAT** — rata-rata +{avg_growth:.1f}%/tahun. Perusahaan dalam fase ekspansi agresif."
                green_flags.append(f"Revenue growth avg +{avg_growth:.1f}%")
            elif avg_growth > 10:
                narratives["revenue"] = f"📈 Revenue growth **SOLID** — rata-rata +{avg_growth:.1f}%/tahun. Growth konsisten double-digit."
                green_flags.append(f"Revenue growth avg +{avg_growth:.1f}%")
            elif avg_growth > 5:
                narratives["revenue"] = f"➡️ Revenue growth **MODERAT** — rata-rata +{avg_growth:.1f}%/tahun. Growth di atas inflasi."
            elif avg_growth > 0:
                narratives["revenue"] = f"🐢 Revenue growth **LAMBAT** — rata-rata +{avg_growth:.1f}%/tahun. Mature company atau kompetisi kuat."
                yellow_flags.append(f"Revenue growth lambat +{avg_growth:.1f}%")
            else:
                narratives["revenue"] = f"⚠️ Revenue **MENURUN** — rata-rata {avg_growth:.1f}%/tahun. Waspada demand issue atau market share loss."
                red_flags.append(f"Revenue turun rata-rata {avg_growth:.1f}%")

            # Trend check: apakah accelerating atau decelerating?
            if len(rev_growths) >= 3:
                if rev_growths[0] > rev_growths[1] > rev_growths[2]:
                    narratives["revenue"] += " Growth **AKSELERASI** (latest > prev > earlier). 🎯"
                    green_flags.append("Growth accelerating")
                elif rev_growths[0] < rev_growths[1] < rev_growths[2]:
                    narratives["revenue"] += " Growth **DECELERATING** (latest < prev < earlier). ⚠️"
                    yellow_flags.append("Growth decelerating")

    # ==========================================
    # 2. PROFITABILITY STORY
    # ==========================================
    if len(income_annual) >= 2:
        latest = income_annual[0]
        prev = income_annual[1]

        margin = latest.get("net_margin")
        prev_margin = prev.get("net_margin")

        if margin is not None:
            if margin > 20:
                narratives["profitability"] = f"💎 Net margin **EXCELLENT** {margin:.1f}% — premium business (software/luxury/monopoli)."
                green_flags.append(f"Net margin tinggi {margin:.1f}%")
            elif margin > 10:
                narratives["profitability"] = f"✅ Net margin **BAGUS** {margin:.1f}% — profitable business."
                green_flags.append(f"Net margin sehat {margin:.1f}%")
            elif margin > 5:
                narratives["profitability"] = f"👍 Net margin **NORMAL** {margin:.1f}% — competitive industry (retail/manufacturing)."
            elif margin > 0:
                narratives["profitability"] = f"⚠️ Net margin **TIPIS** {margin:.1f}% — thin margin business."
                yellow_flags.append(f"Margin tipis {margin:.1f}%")
            else:
                narratives["profitability"] = f"🔴 Net margin **NEGATIF** {margin:.1f}% — RUGI."
                red_flags.append(f"Perusahaan RUGI, margin {margin:.1f}%")

            # Margin trend
            if prev_margin is not None:
                margin_delta = margin - prev_margin
                if margin_delta > 1:
                    narratives["profitability"] += f" Margin **NAIK** dari {prev_margin:.1f}% (Δ +{margin_delta:.1f}%). Operational efficiency improving."
                    green_flags.append("Margin expanding")
                elif margin_delta < -1:
                    narratives["profitability"] += f" Margin **TURUN** dari {prev_margin:.1f}% (Δ {margin_delta:.1f}%). Cost pressure atau harga jual turun."
                    yellow_flags.append("Margin compression")

        # Net income trend
        ni_growth = latest.get("net_income_yoy")
        if ni_growth is not None:
            if ni_growth > 30:
                narratives["net_income"] = f"🚀 Net income **MELONJAK** +{ni_growth:.1f}% YoY. Bottom line accelerating."
                green_flags.append(f"Net income +{ni_growth:.1f}%")
            elif ni_growth > 10:
                narratives["net_income"] = f"📈 Net income **NAIK** +{ni_growth:.1f}% YoY. Profitability improving."
                green_flags.append(f"Net income +{ni_growth:.1f}%")
            elif ni_growth > 0:
                narratives["net_income"] = f"➡️ Net income **NAIK TIPIS** +{ni_growth:.1f}% YoY."
            elif ni_growth > -20:
                narratives["net_income"] = f"⚠️ Net income **TURUN** {ni_growth:.1f}% YoY. Waspada penurunan profitabilitas."
                yellow_flags.append(f"Net income turun {ni_growth:.1f}%")
            else:
                narratives["net_income"] = f"🔴 Net income **ANJLOK** {ni_growth:.1f}% YoY. Major issue."
                red_flags.append(f"Net income drop {ni_growth:.1f}%")

    # ==========================================
    # 3. QUARTERLY MOMENTUM
    # ==========================================
    if len(income_q) >= 2:
        q_latest = income_q[0]
        q_prev = income_q[1]

        qoq = q_latest.get("revenue_qoq")
        yoy = q_latest.get("revenue_yoy")

        if qoq is not None:
            if qoq > 10:
                narratives["quarterly"] = f"🔥 Revenue QoQ **NAIK KUAT** +{qoq:.1f}%. Recent momentum strong."
                green_flags.append(f"QoQ revenue +{qoq:.1f}%")
            elif qoq < -10:
                narratives["quarterly"] = f"⚠️ Revenue QoQ **TURUN TAJAM** {qoq:.1f}%. Recent weakness."
                yellow_flags.append(f"QoQ revenue {qoq:.1f}%")
            else:
                narratives["quarterly"] = f"📊 Revenue QoQ {qoq:+.1f}%, YoY {yoy:+.1f}%." if yoy is not None else f"Revenue QoQ {qoq:+.1f}%."

        ni_qoq = q_latest.get("net_income_qoq")
        if ni_qoq is not None:
            if ni_qoq < -30:
                narratives["quarterly"] = (narratives.get("quarterly", "") + f" Net income Q terakhir **TURUN TAJAM** {ni_qoq:.1f}% QoQ.").strip()
                red_flags.append(f"Q terakhir NI drop {ni_qoq:.1f}%")

    # ==========================================
    # 4. BALANCE SHEET HEALTH
    # ==========================================
    if balance:
        de = balance.get("debt_to_equity")
        eq_ratio = balance.get("equity_ratio")
        cash = balance.get("cash")
        debt = balance.get("total_debt")

        bs_parts = []
        if de is not None:
            if de < 0.3:
                bs_parts.append(f"💰 D/E {de:.2f} **RENDAH** — hutang minimal, financial risk low.")
                green_flags.append(f"D/E rendah {de:.2f}")
            elif de < 1.0:
                bs_parts.append(f"✅ D/E {de:.2f} **SEHAT** — leverage manageable.")
            elif de < 2.0:
                bs_parts.append(f"⚠️ D/E {de:.2f} **TINGGI** — hutang cukup besar, hati-hati saat suku bunga naik.")
                yellow_flags.append(f"D/E tinggi {de:.2f}")
            else:
                bs_parts.append(f"🔴 D/E {de:.2f} **SANGAT TINGGI** — heavy leverage, financial risk BESAR.")
                red_flags.append(f"D/E ekstrim {de:.2f}")

        if cash and debt and cash > debt:
            bs_parts.append(f"💵 **NET CASH POSITION** — cash ({format_large(cash)}) > total debt ({format_large(debt)}).")
            green_flags.append("Net cash position")

        if eq_ratio is not None and eq_ratio > 60:
            bs_parts.append(f"🏦 Equity ratio {eq_ratio:.0f}% — modal sendiri dominan.")

        if bs_parts:
            narratives["balance_sheet"] = " ".join(bs_parts)

    # ==========================================
    # 5. CASH FLOW HEALTH
    # ==========================================
    if cashflow:
        ocf = cashflow.get("operating_cash")
        fcf = cashflow.get("free_cash_flow")
        capex = cashflow.get("capex")

        cf_parts = []
        if ocf is not None:
            if ocf > 0:
                cf_parts.append(f"💧 Operating Cash Flow **POSITIF** {format_large(ocf)} — bisnis inti generate cash.")
                green_flags.append("OCF positive")
            else:
                cf_parts.append(f"⚠️ Operating Cash Flow **NEGATIF** {format_large(ocf)} — cash burn dari operasi.")
                red_flags.append("OCF negative")

        if fcf is not None:
            if fcf > 0:
                cf_parts.append(f"💎 Free Cash Flow **POSITIF** {format_large(fcf)} — after capex, ada cash untuk dividen/buyback/hutang.")
                green_flags.append("FCF positive")
            else:
                cf_parts.append(f"⚠️ Free Cash Flow **NEGATIF** {format_large(fcf)} — capex melebihi operating cash.")
                yellow_flags.append("FCF negative")

        if cf_parts:
            narratives["cash_flow"] = " ".join(cf_parts)

    # ==========================================
    # 6. OVERALL VERDICT
    # ==========================================
    n_green = len(green_flags)
    n_yellow = len(yellow_flags)
    n_red = len(red_flags)

    if n_red >= 3:
        verdict = "🔴 HIGH RISK — Multiple red flags. Hindari investasi baru."
    elif n_red >= 1 and n_green < 3:
        verdict = "🟠 RISKY — Ada red flag serius. Riset lebih dalam sebelum decide."
    elif n_green >= 5 and n_red == 0:
        verdict = "🟢 STRONG BUY CANDIDATE — Fundamental sangat sehat."
    elif n_green >= 3 and n_red == 0:
        verdict = "🟢 HEALTHY — Fundamental sehat. Kandidat investasi."
    elif n_green >= 2 and n_yellow <= 2:
        verdict = "🟡 NEUTRAL — Fundamental OK dengan sedikit concern."
    elif n_yellow >= 3:
        verdict = "🟠 MIXED — Banyak warning, hati-hati."
    else:
        verdict = "⚪ INSUFFICIENT DATA — Perlu cross-check sumber lain."

    return {
        "narratives": narratives,
        "green_flags": green_flags,
        "yellow_flags": yellow_flags,
        "red_flags": red_flags,
        "verdict": verdict,
    }


def full_financial_summary(ticker, market="IDX"):
    data = fetch_financials(ticker, market)
    income_annual = summarize_income(data.get("income_annual", pd.DataFrame()))
    income_quarterly = summarize_income(data.get("income_quarterly", pd.DataFrame()))
    balance = summarize_balance(data.get("balance_annual", pd.DataFrame()))
    cashflow = summarize_cashflow(data.get("cashflow_annual", pd.DataFrame()))

    for i in range(len(income_annual) - 1):
        income_annual[i]["revenue_yoy"] = compute_growth(income_annual[i]["revenue"], income_annual[i+1]["revenue"])
        income_annual[i]["net_income_yoy"] = compute_growth(income_annual[i]["net_income"], income_annual[i+1]["net_income"])

    for i in range(len(income_quarterly) - 1):
        income_quarterly[i]["revenue_qoq"] = compute_growth(income_quarterly[i]["revenue"], income_quarterly[i+1]["revenue"])
        income_quarterly[i]["net_income_qoq"] = compute_growth(income_quarterly[i]["net_income"], income_quarterly[i+1]["net_income"])
    for i in range(len(income_quarterly) - 4):
        income_quarterly[i]["revenue_yoy"] = compute_growth(income_quarterly[i]["revenue"], income_quarterly[i+4]["revenue"])
        income_quarterly[i]["net_income_yoy"] = compute_growth(income_quarterly[i]["net_income"], income_quarterly[i+4]["net_income"])

    if balance and balance.get("total_equity") and balance.get("total_debt"):
        balance["debt_to_equity"] = balance["total_debt"] / balance["total_equity"]
    if balance and balance.get("total_assets") and balance.get("total_equity"):
        balance["equity_ratio"] = balance["total_equity"] / balance["total_assets"] * 100

    return {
        "income_annual": income_annual,
        "income_quarterly": income_quarterly,
        "balance": balance,
        "cashflow": cashflow,
    }
