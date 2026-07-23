#!/usr/bin/env python3
"""One-shot migration: local CSV/SQLite -> Neon Postgres.

Usage:
    export DATABASE_URL='postgresql://...'
    python3 migrate_to_postgres.py [--source-dir /path/to/data]

Reads:
    02-position-tracker.csv, pending_orders.csv, us_positions.csv,
    05-monthly-summary.csv, cash_state.txt, history.db
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from db import get_cursor


def migrate_positions(src_dir: Path, cur):
    path = src_dir / "02-position-tracker.csv"
    if not path.exists():
        print(f"SKIP positions: {path} not found")
        return
    df = pd.read_csv(path)
    n = 0
    for _, r in df.iterrows():
        cur.execute("""
            INSERT INTO positions
              (ticker, lot, avg_price, cost_basis, layer, stop_loss,
               take_profit_1, take_profit_2, entry_date, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
              lot = EXCLUDED.lot, avg_price = EXCLUDED.avg_price,
              cost_basis = EXCLUDED.cost_basis, layer = EXCLUDED.layer,
              stop_loss = EXCLUDED.stop_loss,
              take_profit_1 = EXCLUDED.take_profit_1,
              take_profit_2 = EXCLUDED.take_profit_2,
              entry_date = EXCLUDED.entry_date, notes = EXCLUDED.notes,
              updated_at = NOW()
        """, (
            r["Ticker"], r["Lot"], r["Avg Price"], r["Cost Basis"], r.get("Layer"),
            _num(r.get("Stop Loss")), _num(r.get("Take Profit 1")),
            _num(r.get("Take Profit 2")), _date(r.get("Entry Date")), r.get("Notes"),
        ))
        n += 1
    print(f"positions: {n} rows")


def migrate_pending(src_dir: Path, cur):
    path = src_dir / "pending_orders.csv"
    if not path.exists():
        print(f"SKIP pending_orders: {path} not found")
        return
    df = pd.read_csv(path)
    n = 0
    for _, r in df.iterrows():
        cur.execute("""
            INSERT INTO pending_orders
              (ticker, action, limit_price, lot, amount, status, expiry, layer,
               stop_loss_target, take_profit_1, take_profit_2, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
              action = EXCLUDED.action, limit_price = EXCLUDED.limit_price,
              lot = EXCLUDED.lot, amount = EXCLUDED.amount, status = EXCLUDED.status,
              expiry = EXCLUDED.expiry, layer = EXCLUDED.layer,
              stop_loss_target = EXCLUDED.stop_loss_target,
              take_profit_1 = EXCLUDED.take_profit_1,
              take_profit_2 = EXCLUDED.take_profit_2, notes = EXCLUDED.notes,
              updated_at = NOW()
        """, (
            r["Ticker"], r["Action"], r["Limit Price"], r["Lot"], r["Amount"],
            r["Status"], r.get("Expiry"), r.get("Layer"),
            _num(r.get("Stop Loss Target")), _num(r.get("Take Profit 1")),
            _num(r.get("Take Profit 2")), r.get("Notes"),
        ))
        n += 1
    print(f"pending_orders: {n} rows")


def migrate_cash(src_dir: Path, cur):
    path = src_dir / "cash_state.txt"
    if not path.exists():
        print(f"SKIP cash_state: {path} not found")
        return
    n = 0
    for line in path.read_text().strip().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        cur.execute("""
            INSERT INTO cash_state (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        """, (k.strip(), float(v.strip())))
        n += 1
    print(f"cash_state: {n} rows")


def migrate_us(src_dir: Path, cur):
    path = src_dir / "us_positions.csv"
    if not path.exists():
        print(f"SKIP us_positions: {path} not found")
        return
    df = pd.read_csv(path)
    n = 0
    for _, r in df.iterrows():
        cur.execute("""
            INSERT INTO us_positions (ticker, name, shares, avg_cost_usd, cost_basis_usd, notes)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
              name = EXCLUDED.name, shares = EXCLUDED.shares,
              avg_cost_usd = EXCLUDED.avg_cost_usd,
              cost_basis_usd = EXCLUDED.cost_basis_usd,
              notes = EXCLUDED.notes, updated_at = NOW()
        """, (
            r["Ticker"], r.get("Name"), r["Shares"], r["Avg Cost USD"],
            r["Cost Basis USD"], r.get("Notes"),
        ))
        n += 1
    print(f"us_positions: {n} rows")


def migrate_monthly(src_dir: Path, cur):
    path = src_dir / "05-monthly-summary.csv"
    if not path.exists():
        print(f"SKIP monthly_summary: {path} not found")
        return
    df = pd.read_csv(path)
    n = 0
    for _, r in df.iterrows():
        cur.execute("""
            INSERT INTO monthly_summary
              (bulan, total_trades, winning_trades, losing_trades, win_rate_pct,
               total_realized_pl, avg_win_rp, avg_loss_rp, risk_reward_ratio,
               portfolio_value_awal, portfolio_value_akhir, return_pct, catatan)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (bulan) DO UPDATE SET
              total_trades = EXCLUDED.total_trades,
              winning_trades = EXCLUDED.winning_trades,
              losing_trades = EXCLUDED.losing_trades,
              win_rate_pct = EXCLUDED.win_rate_pct,
              total_realized_pl = EXCLUDED.total_realized_pl,
              avg_win_rp = EXCLUDED.avg_win_rp,
              avg_loss_rp = EXCLUDED.avg_loss_rp,
              risk_reward_ratio = EXCLUDED.risk_reward_ratio,
              portfolio_value_awal = EXCLUDED.portfolio_value_awal,
              portfolio_value_akhir = EXCLUDED.portfolio_value_akhir,
              return_pct = EXCLUDED.return_pct,
              catatan = EXCLUDED.catatan, updated_at = NOW()
        """, (
            r["Bulan"], _int(r.get("Total Trades")), _int(r.get("Winning Trades")),
            _int(r.get("Losing Trades")), _num(r.get("Win Rate %")),
            _num(r.get("Total Realized P/L")), _num(r.get("Avg Win Rp")),
            _num(r.get("Avg Loss Rp")), _num(r.get("Risk Reward Ratio")),
            _num(r.get("Portfolio Value Awal")), _num(r.get("Portfolio Value Akhir")),
            _num(r.get("Return %")), r.get("Catatan"),
        ))
        n += 1
    print(f"monthly_summary: {n} rows")


def migrate_scores(src_dir: Path, cur):
    path = src_dir / "history.db"
    if not path.exists():
        print(f"SKIP score_snapshots: {path} not found")
        return
    sc = sqlite3.connect(path)
    sc.row_factory = sqlite3.Row
    rows = sc.execute("SELECT * FROM score_snapshots").fetchall()
    n = 0
    for r in rows:
        cur.execute("""
            INSERT INTO score_snapshots
              (timestamp, date, ticker, price, momentum, quality, score, verdict,
               return_1y, return_3m, return_1m, rsi, atr_pct)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, ticker) DO UPDATE SET
              timestamp = EXCLUDED.timestamp, price = EXCLUDED.price,
              momentum = EXCLUDED.momentum, quality = EXCLUDED.quality,
              score = EXCLUDED.score, verdict = EXCLUDED.verdict,
              return_1y = EXCLUDED.return_1y, return_3m = EXCLUDED.return_3m,
              return_1m = EXCLUDED.return_1m, rsi = EXCLUDED.rsi,
              atr_pct = EXCLUDED.atr_pct
        """, (
            r["timestamp"], r["date"], r["ticker"], r["price"],
            r["momentum"], r["quality"], r["score"], r["verdict"],
            r["return_1y"], r["return_3m"], r["return_1m"], r["rsi"], r["atr_pct"],
        ))
        n += 1
    sc.close()
    print(f"score_snapshots: {n} rows")


def _num(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _int(v):
    x = _num(v)
    return int(x) if x is not None else None


def _date(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default=".", help="Directory with source data files")
    args = ap.parse_args()

    src = Path(args.source_dir).resolve()
    print(f"Source dir: {src}")

    with get_cursor() as cur:
        migrate_positions(src, cur)
        migrate_pending(src, cur)
        migrate_cash(src, cur)
        migrate_us(src, cur)
        migrate_monthly(src, cur)
        migrate_scores(src, cur)

    print("Migration complete.")


if __name__ == "__main__":
    sys.exit(main())
