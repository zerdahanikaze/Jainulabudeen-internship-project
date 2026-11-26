-- PostgreSQL reporting schema for data/processed/*.csv exports.
-- Load dimensions first, then facts. Use the source CSV headers as column mappings.

CREATE TABLE IF NOT EXISTS dim_products (
    product_id       VARCHAR(16) PRIMARY KEY,
    category         VARCHAR(80) NOT NULL,
    subcategory      VARCHAR(80) NOT NULL,
    base_price       NUMERIC(12, 2) NOT NULL,
    unit_cost        NUMERIC(12, 2) NOT NULL,
    lead_time_days   INTEGER NOT NULL CHECK (lead_time_days > 0)
);

CREATE TABLE IF NOT EXISTS dim_stores (
    store_id         VARCHAR(16) PRIMARY KEY,
    city             VARCHAR(100) NOT NULL,
    store_size       VARCHAR(20) NOT NULL CHECK (store_size IN ('Small', 'Medium', 'Large')),
    region           VARCHAR(40) NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_sales_forecasts (
    forecast_date    DATE NOT NULL,
    store_id         VARCHAR(16) NOT NULL REFERENCES dim_stores(store_id),
    product_id       VARCHAR(16) NOT NULL REFERENCES dim_products(product_id),
    model            VARCHAR(80) NOT NULL,
    actual_units     NUMERIC(14, 4) NOT NULL CHECK (actual_units >= 0),
    forecast_units   NUMERIC(14, 4) NOT NULL CHECK (forecast_units >= 0),
    unit_price       NUMERIC(12, 2) NOT NULL,
    promo_flag       SMALLINT NOT NULL CHECK (promo_flag IN (0, 1)),
    PRIMARY KEY (forecast_date, store_id, product_id, model)
);

CREATE TABLE IF NOT EXISTS fact_inventory (
    store_id                   VARCHAR(16) NOT NULL REFERENCES dim_stores(store_id),
    product_id                 VARCHAR(16) NOT NULL REFERENCES dim_products(product_id),
    current_stock_on_hand      INTEGER NOT NULL CHECK (current_stock_on_hand >= 0),
    reorder_point              INTEGER NOT NULL CHECK (reorder_point >= 0),
    lead_time_days             INTEGER NOT NULL CHECK (lead_time_days > 0),
    avg_daily_forecast_units   NUMERIC(14, 4) NOT NULL CHECK (avg_daily_forecast_units >= 0),
    lead_time_demand_units     NUMERIC(14, 4) NOT NULL CHECK (lead_time_demand_units >= 0),
    expected_stockout_units    NUMERIC(14, 4) NOT NULL CHECK (expected_stockout_units >= 0),
    stockout_exposure_flag     SMALLINT NOT NULL CHECK (stockout_exposure_flag IN (0, 1)),
    inventory_coverage_days    NUMERIC(14, 4),
    reorder_priority_score     NUMERIC(6, 2) NOT NULL CHECK (reorder_priority_score BETWEEN 0 AND 100),
    PRIMARY KEY (store_id, product_id)
);

CREATE INDEX IF NOT EXISTS ix_forecasts_date_model
    ON fact_sales_forecasts (forecast_date, model);
CREATE INDEX IF NOT EXISTS ix_forecasts_product_date
    ON fact_sales_forecasts (product_id, forecast_date);
CREATE INDEX IF NOT EXISTS ix_inventory_risk
    ON fact_inventory (stockout_exposure_flag, reorder_priority_score DESC);

-- Model-level forecast quality; MAPE is omitted here because zero actuals make it undefined.
CREATE OR REPLACE VIEW vw_forecast_accuracy AS
SELECT
    f.forecast_date,
    f.model,
    COUNT(*) AS sku_store_observations,
    AVG(ABS(f.actual_units - f.forecast_units)) AS mae,
    SQRT(AVG(POWER(f.actual_units - f.forecast_units, 2))) AS rmse,
    SUM(ABS(f.actual_units - f.forecast_units))
        / NULLIF(SUM(ABS(f.actual_units)), 0) AS weighted_absolute_percentage_error,
    100 * (
        1 - SUM(ABS(f.actual_units - f.forecast_units))
            / NULLIF(SUM(ABS(f.actual_units)), 0)
    ) AS forecast_accuracy_pct
FROM fact_sales_forecasts AS f
GROUP BY f.forecast_date, f.model;

-- Current lead-time risk by store and SKU. Expected stockout units are a modeled exposure.
CREATE OR REPLACE VIEW vw_stockout_exposure AS
SELECT
    s.region,
    s.city,
    s.store_id,
    p.category,
    p.subcategory,
    p.product_id,
    i.current_stock_on_hand,
    i.reorder_point,
    i.lead_time_days,
    i.lead_time_demand_units,
    i.expected_stockout_units,
    i.inventory_coverage_days,
    i.stockout_exposure_flag,
    i.reorder_priority_score
FROM fact_inventory AS i
JOIN dim_stores AS s ON s.store_id = i.store_id
JOIN dim_products AS p ON p.product_id = i.product_id;

-- Category performance for forecasted and actual units, with error measures.
CREATE OR REPLACE VIEW vw_category_performance AS
SELECT
    p.category,
    f.model,
    MIN(f.forecast_date) AS period_start,
    MAX(f.forecast_date) AS period_end,
    SUM(f.actual_units) AS actual_units,
    SUM(f.forecast_units) AS forecast_units,
    AVG(ABS(f.actual_units - f.forecast_units)) AS mae,
    SQRT(AVG(POWER(f.actual_units - f.forecast_units, 2))) AS rmse,
    SUM(f.actual_units * f.unit_price) AS actual_sales_value
FROM fact_sales_forecasts AS f
JOIN dim_products AS p ON p.product_id = f.product_id
GROUP BY p.category, f.model;

-- Example drill-down query: category -> store -> product.
SELECT
    p.category,
    s.store_id,
    f.product_id,
    SUM(f.actual_units) AS actual_units,
    SUM(f.forecast_units) AS forecast_units,
    AVG(ABS(f.actual_units - f.forecast_units)) AS mae
FROM fact_sales_forecasts AS f
JOIN dim_products AS p ON p.product_id = f.product_id
JOIN dim_stores AS s ON s.store_id = f.store_id
WHERE f.model = 'Random Forest'
GROUP BY p.category, s.store_id, f.product_id
ORDER BY p.category, s.store_id, f.product_id;
