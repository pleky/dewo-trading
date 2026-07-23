#!/usr/bin/env python3
"""
Score history tracking + watchlist management.
Uses SQLite for score snapshots (append-only).
"""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR)))
DB_PATH = DATA_DIR / "history.db"
WATCHLIST_FILE = BASE_DIR / "watchlist.txt"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS score_snapshots (
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            price REAL,
            momentum INTEGER,
            quality INTEGER,
            score INTEGER,
            verdict TEXT,
            return_1y REAL,
            return_3m REAL,
            return_1m REAL,
            rsi REAL,
            atr_pct REAL,
            PRIMARY KEY (date, ticker)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON score_snapshots(ticker)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_date ON score_snapshots(date)")
    conn.commit()
    conn.close()


def snapshot_scores(results: list):
    """Save today's scores. Skip if already recorded today."""
    if not results:
        return 0

    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = datetime.now().isoformat()
    today = datetime.now().strftime("%Y-%m-%d")

    count = 0
    for r in results:
        try:
            c.execute("""
                INSERT OR REPLACE INTO score_snapshots
                (timestamp, date, ticker, price, momentum, quality, score, verdict,
                 return_1y, return_3m, return_1m, rsi, atr_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, today,
                r.get("Ticker") or r.get("ticker"),
                r.get("Price") or r.get("price") or (r.get("metrics", {}).get("current") if isinstance(r.get("metrics"), dict) else None),
                r.get("Momentum") or r.get("momentum"),
                r.get("Quality") or r.get("quality"),
                r.get("Score") or r.get("score") or r.get("overall"),
                r.get("Verdict") or r.get("verdict"),
                r.get("1Y%") or (r.get("metrics", {}).get("return_1y") if isinstance(r.get("metrics"), dict) else None),
                r.get("3M%") or (r.get("metrics", {}).get("return_3m") if isinstance(r.get("metrics"), dict) else None),
                r.get("1M%") or (r.get("metrics", {}).get("return_1m") if isinstance(r.get("metrics"), dict) else None),
                r.get("RSI") or (r.get("metrics", {}).get("rsi") if isinstance(r.get("metrics"), dict) else None),
                r.get("ATR%") or (r.get("metrics", {}).get("atr_pct") if isinstance(r.get("metrics"), dict) else None),
            ))
            count += 1
        except Exception as e:
            print(f"[WARN] Snapshot {r.get('Ticker')}: {e}")

    conn.commit()
    conn.close()
    return count


def get_history(ticker: str, days: int = 30) -> pd.DataFrame:
    """Get score history for a ticker."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT * FROM score_snapshots WHERE ticker = ? AND date >= ? ORDER BY date",
        conn, params=(ticker, since)
    )
    conn.close()
    return df


def get_latest_and_previous(ticker: str) -> tuple:
    """Get today's + latest previous snapshot for delta comparison."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT date, score, momentum, quality, price, verdict
        FROM score_snapshots WHERE ticker = ?
        ORDER BY date DESC LIMIT 2
    """, (ticker,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return None, None
    today = rows[0] if len(rows) >= 1 else None
    prev = rows[1] if len(rows) >= 2 else None
    return today, prev


def get_score_changes(min_change: int = 5) -> pd.DataFrame:
    """Find tickers with score change ≥ min_change since previous snapshot."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
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
        )
        WHERE rn <= 2
        GROUP BY ticker
        HAVING score_prev IS NOT NULL
    """, conn)
    conn.close()

    if df.empty:
        return df

    df["score_change"] = df["score_now"] - df["score_prev"]
    df["price_change_pct"] = (df["price_now"] - df["price_prev"]) / df["price_prev"] * 100
    df = df[df["score_change"].abs() >= min_change].sort_values("score_change", ascending=False)
    return df


# ==============================================================
# Watchlist
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
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT date) FROM score_snapshots")
    n = c.fetchone()[0]
    conn.close()
    return n
