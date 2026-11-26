-- Privacy-safe reporting layer for the synthetic pipeline exports.
-- Load only data/processed/aggregate CSVs into these tables; never load encounter rows.

CREATE TABLE IF NOT EXISTS agg_cohort_risk (
    cohort_dimension VARCHAR(40) NOT NULL,
    cohort_value VARCHAR(80) NOT NULL,
    cohort_size INTEGER,
    adverse_outcomes INTEGER,
    adverse_outcome_rate DECIMAL(8, 6),
    high_risk_count INTEGER,
    high_risk_share DECIMAL(8, 6),
    mean_length_of_stay_days DECIMAL(10, 3),
    is_suppressed BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (cohort_dimension, cohort_value),
    CHECK (is_suppressed = TRUE OR cohort_size IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS agg_subgroup_fairness (
    dimension VARCHAR(40) NOT NULL,
    subgroup VARCHAR(80) NOT NULL,
    sample_size INTEGER,
    positive_count INTEGER,
    negative_count INTEGER,
    suppressed BOOLEAN NOT NULL DEFAULT FALSE,
    recall DECIMAL(8, 6),
    precision DECIMAL(8, 6),
    f1 DECIMAL(8, 6),
    roc_auc DECIMAL(8, 6),
    predicted_positive_rate DECIMAL(8, 6),
    observed_rate DECIMAL(8, 6),
    calibration_gap DECIMAL(8, 6),
    parity_ratio DECIMAL(8, 6),
    PRIMARY KEY (dimension, subgroup),
    CHECK (suppressed = TRUE OR sample_size IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS model_evaluation_metrics (
    model VARCHAR(100) PRIMARY KEY,
    sample_size INTEGER NOT NULL,
    positive_count INTEGER NOT NULL,
    recall DECIMAL(8, 6) NOT NULL,
    precision DECIMAL(8, 6) NOT NULL,
    f1 DECIMAL(8, 6) NOT NULL,
    roc_auc DECIMAL(8, 6) NOT NULL,
    predicted_positive_rate DECIMAL(8, 6) NOT NULL,
    observed_rate DECIMAL(8, 6) NOT NULL,
    calibration_gap DECIMAL(8, 6) NOT NULL,
    cv_roc_auc_mean DECIMAL(8, 6) NOT NULL,
    cv_roc_auc_std DECIMAL(8, 6) NOT NULL,
    threshold DECIMAL(5, 4) NOT NULL
);

-- Each row describes exactly one population-level cohort partition.
CREATE VIEW IF NOT EXISTS vw_population_risk_distribution AS
SELECT
    cohort_dimension,
    cohort_value,
    cohort_size,
    adverse_outcomes,
    adverse_outcome_rate,
    high_risk_count,
    high_risk_share,
    mean_length_of_stay_days
FROM agg_cohort_risk
WHERE is_suppressed = FALSE;

CREATE VIEW IF NOT EXISTS vw_department_risk_trends AS
SELECT
    cohort_value AS department_id,
    cohort_size,
    adverse_outcomes,
    adverse_outcome_rate,
    high_risk_count,
    high_risk_share,
    mean_length_of_stay_days
FROM agg_cohort_risk
WHERE cohort_dimension = 'department_id'
  AND is_suppressed = FALSE;
