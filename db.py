#!/usr/bin/env python3
"""Neon Postgres connection helper.

DATABASE_URL priority:
1. env var DATABASE_URL
2. st.secrets["DATABASE_URL"] (when running under Streamlit)
"""

import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine

_engine = None


def _get_database_url():
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    try:
        import streamlit as st
        return st.secrets["DATABASE_URL"]
    except Exception:
        raise RuntimeError(
            "DATABASE_URL not set. Set env var or Streamlit secret."
        )


def get_engine():
    """Cached SQLAlchemy engine for pandas.read_sql."""
    global _engine
    if _engine is None:
        _engine = create_engine(_get_database_url(), pool_pre_ping=True)
    return _engine


@contextmanager
def get_conn():
    """Raw psycopg2 connection (for pandas.read_sql, use get_engine())."""
    conn = psycopg2.connect(_get_database_url())
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_cursor(dict_cursor: bool = False):
    with get_conn() as conn:
        factory = psycopg2.extras.RealDictCursor if dict_cursor else None
        cur = conn.cursor(cursor_factory=factory)
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
