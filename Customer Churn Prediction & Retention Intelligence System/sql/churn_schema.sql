-- Project 01: Customer Churn Prediction & Retention Intelligence
-- Power BI/SQL Server-style schemas

CREATE TABLE DimCustomer (
    customer_id BIGINT PRIMARY KEY,
    signup_date DATE,
    age INT,
    region VARCHAR(50),
    tenure_months INT,
    subscription_plan VARCHAR(50),
    contract_type VARCHAR(50),
    payment_method VARCHAR(50),
    monthly_fee DECIMAL(10,2),
    avg_monthly_usage_gb DECIMAL(10,2),
    missed_payments_12m INT,
    support_tickets_12m INT,
    satisfaction_score DECIMAL(4,1),
    churn_flag INT NOT NULL
);

CREATE TABLE FactChurnScore (
    customer_id BIGINT PRIMARY KEY,
    churn_probability DECIMAL(9,6) NOT NULL,
    predicted_churn INT NOT NULL,
    retention_risk_segment VARCHAR(20),
    score_run_timestamp DATETIME2 DEFAULT SYSUTCDATETIME(),
    FOREIGN KEY (customer_id) REFERENCES DimCustomer(customer_id)
);

CREATE TABLE FactRetentionAction (
    action_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    action_type VARCHAR(100),
    action_status VARCHAR(30),
    action_month DATE,
    offer_cost DECIMAL(10,2),
    expected_retention_rate DECIMAL(6,4),
    FOREIGN KEY (customer_id) REFERENCES DimCustomer(customer_id)
);

-- Clean customer profile load view
GO
CREATE VIEW vw_CustomerChurn AS
SELECT
    c.customer_id,
    c.signup_date,
    c.age,
    c.region,
    c.tenure_months,
    c.subscription_plan,
    c.contract_type,
    c.payment_method,
    c.monthly_fee,
    c.avg_monthly_usage_gb,
    c.missed_payments_12m,
    c.support_tickets_12m,
    c.satisfaction_score,
    c.churn_flag,
    s.churn_probability,
    s.predicted_churn,
    s.retention_risk_segment
FROM DimCustomer c
LEFT JOIN FactChurnScore s ON c.customer_id = s.customer_id;
GO

-- Store the monthly risk segment
MERGE INTO FactChurnScore AS target
USING (
    SELECT customer_id, churn_probability, predicted_churn, retention_risk_segment
    FROM staging_churn_scores
) AS source
ON target.customer_id = source.customer_id
WHEN MATCHED THEN UPDATE SET
    target.churn_probability = source.churn_probability,
    target.predicted_churn = source.predicted_churn,
    target.retention_risk_segment = source.retention_risk_segment,
    target.score_run_timestamp = SYSUTCDATETIME()
WHEN NOT MATCHED THEN INSERT
    (customer_id, churn_probability, predicted_churn, retention_risk_segment)
VALUES
    (source.customer_id, source.churn_probability,
     source.predicted_churn, source.retention_risk_segment);

-- High-risk monthly work queue
SELECT
    customer_id,
    churn_probability,
    retention_risk_segment,
    subscription_plan,
    contract_type,
    missed_payments_12m,
    support_tickets_12m
FROM vw_CustomerChurn
WHERE retention_risk_segment IN ('High', 'Critical')
ORDER BY churn_probability DESC;

-- Segment-level performance and opportunity
SELECT
    retention_risk_segment,
    COUNT(*) AS customer_count,
    ROUND(AVG(churn_probability), 4) AS avg_risk,
    SUM(churn_flag) AS observed_churners
FROM vw_CustomerChurn
GROUP BY retention_risk_segment
ORDER BY AVG(churn_probability) DESC;
