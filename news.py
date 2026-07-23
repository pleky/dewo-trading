#!/usr/bin/env python3
"""
News scraper module — fetch Indonesian stock news per ticker.
Uses Google News RSS + Yahoo Finance news as fallback.
"""

from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import yfinance as yf

BASE_DIR = Path(__file__).parent

# Keywords for news classification
BULLISH_KEYWORDS = [
    "naik", "menguat", "meroket", "melejit", "rebound", "recovery", "cuan",
    "profit", "laba", "dividen", "buyback", "rights issue", "akuisisi",
    "ekspansi", "kinerja", "positif", "melonjak", "reli",
]
BEARISH_KEYWORDS = [
    "turun", "melemah", "anjlok", "jatuh", "crash", "rugi", "kerugian",
    "PKPU", "pailit", "delisting", "suspensi", "koreksi", "tekanan",
    "MSCI", "removal", "keluar", "penurunan", "negatif",
]
CORPORATE_ACTION_KEYWORDS = [
    "dividen", "buyback", "rights issue", "stock split", "reverse split",
    "akuisisi", "merger", "IPO", "RUPS", "MSCI", "delisting", "MTO",
    "tender offer", "pengendali baru", "suspensi", "PKPU", "gugatan",
    "obligasi", "right issue", "private placement", "spin-off",
]


def fetch_google_news(ticker: str, count: int = 10) -> list:
    """Fetch Indonesian news via Google News RSS."""
    query = f"saham {ticker} IDX"
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=id&gl=ID&ceid=ID:id"

    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:count]:
            items.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "source": entry.get("source", {}).get("title", "Unknown") if hasattr(entry, "source") else "Google News",
            })
        return items
    except Exception as e:
        print(f"[WARN] Google News fetch {ticker}: {e}")
        return []


def fetch_yahoo_news(ticker: str) -> list:
    """Fetch news via yfinance (mostly English but sometimes has IDX news)."""
    try:
        yf_ticker = f"{ticker}.JK"
        news = yf.Ticker(yf_ticker).news
        items = []
        for n in news[:5]:
            content = n.get("content", {}) if isinstance(n.get("content"), dict) else {}
            items.append({
                "title": content.get("title") or n.get("title", ""),
                "link": content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else n.get("link", ""),
                "published": content.get("pubDate") or n.get("providerPublishTime", ""),
                "source": (content.get("provider", {}) or {}).get("displayName") or n.get("publisher", "Yahoo Finance"),
            })
        return items
    except Exception:
        return []


def fetch_news(ticker: str, source: str = "combined", count: int = 10) -> list:
    """Fetch news from Google News (default) or combined sources."""
    items = []
    if source in ("google", "combined"):
        items.extend(fetch_google_news(ticker, count))
    if source in ("yahoo", "combined"):
        items.extend(fetch_yahoo_news(ticker))

    # Dedup by title
    seen_titles = set()
    unique = []
    for item in items:
        title = item.get("title", "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique.append(item)

    return unique[:count]


def classify_sentiment(text: str) -> str:
    """Simple keyword-based sentiment: bullish / bearish / neutral."""
    if not text:
        return "neutral"
    text_lower = text.lower()

    bull_score = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bear_score = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)

    if bull_score > bear_score:
        return "bullish"
    if bear_score > bull_score:
        return "bearish"
    return "neutral"


def detect_corporate_action(text: str) -> list:
    """Detect corporate action keywords in headline."""
    if not text:
        return []
    text_lower = text.lower()
    return [kw for kw in CORPORATE_ACTION_KEYWORDS if kw.lower() in text_lower]


def enrich_news(items: list) -> list:
    """Add sentiment + corporate action detection to news items."""
    for item in items:
        title = item.get("title", "")
        item["sentiment"] = classify_sentiment(title)
        item["corporate_actions"] = detect_corporate_action(title)
        item["is_high_impact"] = len(item["corporate_actions"]) > 0
    return items


def has_high_impact_news(items: list) -> bool:
    return any(item.get("is_high_impact") for item in items)


def format_published(pub_str: str) -> str:
    """Try to format publish date to human-friendly string."""
    if not pub_str:
        return ""
    # Try RFC 822 (Google News format)
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(pub_str, fmt)
            return dt.strftime("%d %b %Y %H:%M")
        except ValueError:
            continue
    return pub_str[:30]  # fallback: truncate
