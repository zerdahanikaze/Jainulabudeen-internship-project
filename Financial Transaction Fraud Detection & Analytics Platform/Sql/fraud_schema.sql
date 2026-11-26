-- PostgreSQL operational reporting schema.
-- Do not load synthetic ground-truth labels into operational score tables.

CREATE TABLE IF NOT EXISTS dim_accounts (
    account_id            VARCHAR(32) PRIMARY KEY,
    account_age_days      INTEGER NOT NULL CHECK (account_age_days >= 0),
    kyc_tier              VARCHAR(32) NOT NULL,
    avg_monthly_spend     NUMERIC(18, 2) NOT NULL CHECK (avg_monthly_spend >= 0),
    home_country          CHAR(2) NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_transaction_scores (
    transaction_id        VARCHAR(32) PRIMARY KEY,
    account_id            VARCHAR(32) NOT NULL REFERENCES dim_accounts(account_id),
    transaction_timestamp TIMESTAMPTZ NOT NULL,
    amount                NUMERIC(18, 2) NOT NULL CHECK (amount >= 0),
    risk_score            DOUBLE PRECISION NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    alert_flag            BOOLEAN NOT NULL,
    alert_rank_daily      INTEGER NOT NULL CHECK (alert_rank_daily >= 1),
    daily_capacity        INTEGER NOT NULL CHECK (daily_capacity >= 0),
    model_version         VARCHAR(128) NOT NULL,
    scored_at             TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_scores_timestamp
    ON fact_transaction_scores (transaction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_scores_alert_queue
    ON fact_transaction_scores (alert_flag, alert_rank_daily, transaction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_scores_account_timestamp
    ON fact_transaction_scores (account_id, transaction_timestamp DESC);

CREATE TABLE IF NOT EXISTS fact_investigation_outcomes (
    transaction_id        VARCHAR(32) PRIMARY KEY
                          REFERENCES fact_transaction_scores(transaction_id),
    account_id            VARCHAR(32) NOT NULL REFERENCES dim_accounts(account_id),
    reviewed_at           TIMESTAMPTZ NOT NULL,
    outcome               VARCHAR(32) NOT NULL
                          CHECK (outcome IN ('confirmed_fraud', 'false_positive', 'inconclusive', 'escalated')),
    prevented_loss        NUMERIC(18, 2) NOT NULL DEFAULT 0 CHECK (prevented_loss >= 0),
    reviewer_id           VARCHAR(128) NOT NULL,
    outcome_source        VARCHAR(64) NOT NULL DEFAULT 'human_review',
    notes                 TEXT,
    CHECK (outcome <> 'confirmed_fraud' OR prevented_loss >= 0)
);

CREATE INDEX IF NOT EXISTS idx_outcomes_reviewed_at
    ON fact_investigation_outcomes (reviewed_at DESC);
CREATE INDEX IF NOT EXISTS idx_outcomes_result
    ON fact_investigation_outcomes (outcome, reviewed_at DESC);

-- One row per transaction day. "Confirmed fraud rate" is precision among reviewed alerts.
CREATE OR REPLACE VIEW vw_daily_fraud_operations AS
SELECT
    s.transaction_timestamp::date AS transaction_date,
    COUNT(*) AS transaction_volume,
    COALESCE(SUM(s.amount), 0)::NUMERIC(18, 2) AS total_amount,
    COUNT(*) FILTER (WHERE s.alert_flag) AS flagged_volume,
    COUNT(*) FILTER (WHERE o.outcome IN ('confirmed_fraud', 'false_positive')) AS reviewed_alerts,
    COUNT(*) FILTER (WHERE o.outcome = 'confirmed_fraud') AS confirmed_fraud_count,
    COUNT(*) FILTER (WHERE o.outcome = 'false_positive') AS false_positive_count,
    COUNT(*) FILTER (WHERE o.outcome = 'inconclusive') AS inconclusive_count,
    COUNT(*) FILTER (WHERE o.outcome = 'confirmed_fraud')::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE o.outcome IN ('confirmed_fraud', 'false_positive')), 0)
        AS confirmed_fraud_rate_precision,
    COUNT(*) FILTER (WHERE o.outcome = 'false_positive')::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE o.outcome IN ('confirmed_fraud', 'false_positive')), 0)
        AS false_positive_share_of_reviewed
FROM fact_transaction_scores AS s
LEFT JOIN fact_investigation_outcomes AS o
    ON o.transaction_id = s.transaction_id
GROUP BY s.transaction_timestamp::date;

CREATE OR REPLACE VIEW vw_daily_flagged_volume AS
SELECT
    transaction_timestamp::date AS transaction_date,
    COUNT(*) AS flagged_volume,
    COALESCE(SUM(amount), 0)::NUMERIC(18, 2) AS flagged_amount
FROM fact_transaction_scores
WHERE alert_flag
GROUP BY transaction_timestamp::date;

CREATE OR REPLACE VIEW vw_daily_confirmed_fraud_rate AS
SELECT
    s.transaction_timestamp::date AS transaction_date,
    COUNT(*) FILTER (WHERE o.outcome = 'confirmed_fraud') AS confirmed_fraud_alerts,
    COUNT(*) FILTER (WHERE o.outcome IN ('confirmed_fraud', 'false_positive')) AS resolved_reviews,
    COUNT(*) FILTER (WHERE o.outcome = 'confirmed_fraud')::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE o.outcome IN ('confirmed_fraud', 'false_positive')), 0)
        AS confirmed_fraud_rate_precision
FROM fact_transaction_scores AS s
JOIN fact_investigation_outcomes AS o ON o.transaction_id = s.transaction_id
WHERE s.alert_flag
GROUP BY s.transaction_timestamp::date;

-- This is the false-positive share among resolved reviews, not population FPR (FP / all negatives).
CREATE OR REPLACE VIEW vw_daily_false_positive_rate AS
SELECT
    s.transaction_timestamp::date AS transaction_date,
    COUNT(*) FILTER (WHERE o.outcome = 'false_positive') AS false_positive_alerts,
    COUNT(*) FILTER (WHERE o.outcome IN ('confirmed_fraud', 'false_positive')) AS resolved_reviews,
    COUNT(*) FILTER (WHERE o.outcome = 'false_positive')::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE o.outcome IN ('confirmed_fraud', 'false_positive')), 0)
        AS false_positive_share_of_reviewed
FROM fact_transaction_scores AS s
JOIN fact_investigation_outcomes AS o ON o.transaction_id = s.transaction_id
WHERE s.alert_flag
GROUP BY s.transaction_timestamp::date;

-- Pending, highest-risk alert first. Risk score is a prioritization signal for investigator review.
CREATE OR REPLACE VIEW vw_top_priority_alert_queue AS
SELECT
    s.transaction_id,
    s.account_id,
    s.transaction_timestamp,
    s.amount,
    s.risk_score,
    s.alert_rank_daily,
    s.model_version
FROM fact_transaction_scores AS s
WHERE s.alert_flag
  AND NOT EXISTS (
      SELECT 1
      FROM fact_investigation_outcomes AS o
      WHERE o.transaction_id = s.transaction_id
  )
ORDER BY s.risk_score DESC, s.transaction_timestamp ASC;
