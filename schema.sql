-- Trading journal schema for Neon Postgres
-- Idempotent: safe to re-run

CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT PRIMARY KEY,
    lot NUMERIC NOT NULL,
    avg_price NUMERIC NOT NULL,
    cost_basis NUMERIC NOT NULL,
    layer TEXT,
    stop_loss NUMERIC,
    take_profit_1 NUMERIC,
    take_profit_2 NUMERIC,
    entry_date DATE,
    notes TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pending_orders (
    ticker TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    limit_price NUMERIC NOT NULL,
    lot NUMERIC NOT NULL,
    amount NUMERIC NOT NULL,
    status TEXT NOT NULL,
    expiry TEXT,
    layer TEXT,
    stop_loss_target NUMERIC,
    take_profit_1 NUMERIC,
    take_profit_2 NUMERIC,
    notes TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cash_state (
    key TEXT PRIMARY KEY,
    value NUMERIC NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS us_positions (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    shares NUMERIC NOT NULL,
    avg_cost_usd NUMERIC NOT NULL,
    cost_basis_usd NUMERIC NOT NULL,
    notes TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monthly_summary (
    bulan TEXT PRIMARY KEY,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate_pct NUMERIC,
    total_realized_pl NUMERIC,
    avg_win_rp NUMERIC,
    avg_loss_rp NUMERIC,
    risk_reward_ratio NUMERIC,
    portfolio_value_awal NUMERIC,
    portfolio_value_akhir NUMERIC,
    return_pct NUMERIC,
    catatan TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

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
);

CREATE INDEX IF NOT EXISTS idx_score_ticker ON score_snapshots(ticker);
CREATE INDEX IF NOT EXISTS idx_score_date ON score_snapshots(date);
