# Processed Data Dictionary

Forecasts are rolling one-day-ahead backtests: prior-day actual sales are available when each daily forecast is issued.

- **forecast_date** (`actuals_vs_forecasts.csv`): Date being forecast (daily, rolling one-day-ahead backtest).
- **store_id / product_id** (`actuals_vs_forecasts.csv`): Retail store and SKU keys.
- **actual_units** (`actuals_vs_forecasts.csv`): Observed units sold on forecast_date.
- **forecast_units** (`actuals_vs_forecasts.csv`): Non-negative model prediction for units sold.
- **unit_price / promo_flag** (`actuals_vs_forecasts.csv`): Observed/planned price and promotion input on that date.
- **model** (`actuals_vs_forecasts.csv`): Model that generated the forecast.
- **mae / rmse / r2** (`forecast_metrics.csv`): Test-set error metrics; lower MAE/RMSE and higher R-squared are preferred.
- **validation_mae** (`forecast_metrics.csv`): Validation MAE used for model parameter selection.
- **current_stock_on_hand / reorder_point** (`inventory_risk.csv`): Latest synthetic stock and policy threshold.
- **avg_daily_forecast_units** (`inventory_risk.csv`): Mean forecast over the latest 28 test dates from the best-MAE model.
- **lead_time_demand_units** (`inventory_risk.csv`): Average daily forecast multiplied by product lead time.
- **expected_stockout_units** (`inventory_risk.csv`): Positive lead-time demand less on-hand stock; a planning exposure, not a lost-sales observation.
- **inventory_coverage_days** (`inventory_risk.csv`): On-hand units divided by average forecast demand per day.
- **reorder_priority_score** (`inventory_risk.csv`): 0-100 unmet share of lead-time demand; 100 denotes zero on-hand stock.
- **mean / median / std / q1 / q3 / iqr** (`demand_statistics.csv`): Descriptive statistics over the completed daily panel.
- **coefficient_of_variation** (`product_demand_variability.csv`): Product standard deviation divided by mean daily units.
- **mae / rmse / r2** (`error_analysis_by_product.csv and error_analysis_by_store.csv`): Test error slices for the globally best-MAE model.
