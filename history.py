#!/usr/bin/env python3
"""
Score history tracking + watchlist management.
Uses Neon Postgres for score snapshots (append-only).
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from db import get_cursor, get_engine

BASE_DIR = Path(__file__).parent
WATCHLIST_FILE = BASE_DIR / "watchlist.txt"


def init_db():
    """Idempotent — safe to call, schema.sql already creates the table."""
    with get_cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS score_snapshots (
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                price NUMERIC,
                momentum INTEGER,
                quality INTEGER,
                score INTEGER,
                verdict TEXT,
                return_1y NUMERIC,
                return_3m NUMERIC,
                return_1m NUMERIC,
                rsi NUMERIC,
                atr_pct NUMERIC,
                PRIMARY KEY (date, ticker)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON score_snapshots(ticker)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_date ON score_snapshots(date)")


def snapshot_scores(results: list):
    """Save today's scores. Upsert on (date, ticker)."""
    if not results:
        return 0

    now = datetime.now().isoformat()
    today = datetime.now().strftime("%Y-%m-%d")

    count = 0
    with get_cursor() as cur:
        for r in results:
            try:
                metrics = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
                cur.execute("""
                    INSERT INTO score_snapshots
                    (timestamp, date, ticker, price, momentum, quality, score, verdict,
                     return_1y, return_3m, return_1m, rsi, atr_pct)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, ticker) DO UPDATE SET
                        timestamp = EXCLUDED.timestamp,
                        price = EXCLUDED.price,
                        momentum = EXCLUDED.momentum,
                        quality = EXCLUDED.quality,
                        score = EXCLUDED.score,
                        verdict = EXCLUDED.verdict,
                        return_1y = EXCLUDED.return_1y,
                        return_3m = EXCLUDED.return_3m,
                        return_1m = EXCLUDED.return_1m,
                        rsi = EXCLUDED.rsi,
                        atr_pct = EXCLUDED.atr_pct
                """, (
                    now, today,
                    r.get("Ticker") or r.get("ticker"),
                    r.get("Price") or r.get("price") or metrics.get("current"),
                    r.get("Momentum") or r.get("momentum"),
                    r.get("Quality") or r.get("quality"),
                    r.get("Score") or r.get("score") or r.get("overall"),
                    r.get("Verdict") or r.get("verdict"),
                    r.get("1Y%") or metrics.get("return_1y"),
                    r.get("3M%") or metrics.get("return_3m"),
                    r.get("1M%") or metrics.get("return_1m"),
                    r.get("RSI") or metrics.get("rsi"),
                    r.get("ATR%") or metrics.get("atr_pct"),
                ))
                count += 1
            except Exception as e:
                print(f"[WARN] Snapshot {r.get('Ticker')}: {e}")

    return count


def get_history(ticker: str, days: int = 30) -> pd.DataFrame:
    """Get score history for a ticker."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT * FROM score_snapshots WHERE ticker = %(t)s AND date >= %(s)s ORDER BY date",
        get_engine(), params={"t": ticker, "s": since}
    )
    return df


def get_latest_and_previous(ticker: str) -> tuple:
    """Get today's + latest previous snapshot for delta comparison."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT date, score, momentum, quality, price, verdict
            FROM score_snapshots WHERE ticker = %s
            ORDER BY date DESC LIMIT 2
        """, (ticker,))
        rows = cur.fetchall()
    if not rows:
        return None, None
    today = rows[0] if len(rows) >= 1 else None
    prev = rows[1] if len(rows) >= 2 else None
    return today, prev


def get_score_changes(min_change: int = 5) -> pd.DataFrame:
    """Find tickers with score change ≥ min_change since previous snapshot."""
    df = pd.read_sql_query("""
        SELECT ticker,
               MAX(CASE WHEN rn = 1 THEN score END) AS score_now,
               MAX(CASE WHEN rn = 2 THEN score END) AS score_prev,
               MAX(CASE WHEN rn = 1 THEN date END) AS date_now,
               MAX(CASE WHEN rn = 2 THEN date END) AS date_prev,
               MAX(CASE WHEN rn = 1 THEN price END) AS price_now,
               MAX(CASE WHEN rn = 2 THEN price END) AS price_prev,
               MAX(CASE WHEN rn = 1 THEN verdict END) AS verdict_now,
               MAX(CASE WHEN rn = 2 THEN verdict END) AS verdict_prev
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM score_snapshots
        ) t
        WHERE rn <= 2
        GROUP BY ticker
        HAVING MAX(CASE WHEN rn = 2 THEN score END) IS NOT NULL
    """, get_engine())

    if df.empty:
        return df

    df["score_now"] = pd.to_numeric(df["score_now"], errors="coerce")
    df["score_prev"] = pd.to_numeric(df["score_prev"], errors="coerce")
    df["price_now"] = pd.to_numeric(df["price_now"], errors="coerce")
    df["price_prev"] = pd.to_numeric(df["price_prev"], errors="coerce")
    df["score_change"] = df["score_now"] - df["score_prev"]
    df["price_change_pct"] = (df["price_now"] - df["price_prev"]) / df["price_prev"] * 100
    df = df[df["score_change"].abs() >= min_change].sort_values("score_change", ascending=False)
    return df


# ==============================================================
# Watchlist (still file-based — small, config-like)
# ==============================================================

def load_watchlist() -> list:
    if not WATCHLIST_FILE.exists():
        return []
    lines = WATCHLIST_FILE.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def save_watchlist(tickers: list):
    header = "# Watchlist — one ticker per line\n# Edit to add/remove tickers\n\n"
    WATCHLIST_FILE.write_text(header + "\n".join(tickers) + "\n")


def add_to_watchlist(ticker: str):
    wl = load_watchlist()
    if ticker not in wl:
        wl.append(ticker)
        save_watchlist(wl)
        return True
    return False


def remove_from_watchlist(ticker: str):
    wl = load_watchlist()
    if ticker in wl:
        wl.remove(ticker)
        save_watchlist(wl)
        return True
    return False


def get_days_tracked() -> int:
    """How many days of history tracked."""
    with get_cursor() as cur:
        cur.execute("SELECT COUNT(DISTINCT date) FROM score_snapshots")
        return cur.fetchone()[0]
