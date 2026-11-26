"""Reusable ingestion, analysis, modeling, and export functions for the notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SALES_COLUMNS = {"date", "store_id", "product_id", "units_sold", "unit_price", "promo_flag"}
PRODUCT_COLUMNS = {"product_id", "category", "subcategory", "base_price", "unit_cost", "lead_time_days"}
STORE_COLUMNS = {"store_id", "city", "store_size", "region"}
INVENTORY_COLUMNS = {"store_id", "product_id", "current_stock_on_hand", "reorder_point"}
KEY_COLUMNS = ["date", "store_id", "product_id"]
FEATURE_COLUMNS = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_std_7",
    "day_of_week",
    "month",
    "is_weekend",
    "holiday_flag",
    "promo_flag",
    "price_change_pct",
    "unit_price",
]


def _parse_mixed_dates(values: pd.Series) -> pd.Series:
    """Parse the ISO, US numeric, and day-month-name date encodings in the raw feed."""
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for date_format in ("%Y-%m-%d", "%m/%d/%Y", "%d %b %Y"):
        missing = parsed.isna()
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format=date_format, errors="coerce")
    if parsed.isna().any():
        examples = text.loc[parsed.isna()].head(5).tolist()
        raise ValueError(f"Unparseable sales dates found, examples: {examples}")
    return parsed


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path.name} contains no rows")
    return frame


def load_and_prepare(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate all inputs, deduplicate observations, and complete the daily panel."""
    sales = _read_csv(raw_dir / "sales_history.csv", SALES_COLUMNS)
    products = _read_csv(raw_dir / "products.csv", PRODUCT_COLUMNS)
    stores = _read_csv(raw_dir / "stores.csv", STORE_COLUMNS)
    inventory = _read_csv(raw_dir / "current_inventory.csv", INVENTORY_COLUMNS)
    sales["date"] = _parse_mixed_dates(sales["date"])
    sales["units_sold"] = pd.to_numeric(sales["units_sold"], errors="raise").clip(lower=0)
    sales["unit_price"] = pd.to_numeric(sales["unit_price"], errors="raise")
    sales["promo_flag"] = pd.to_numeric(sales["promo_flag"], errors="raise").astype("int8")

    duplicates = int(sales.duplicated(KEY_COLUMNS, keep="first").sum())
    sales = sales.drop_duplicates(KEY_COLUMNS, keep="first")
    unknown_products = set(sales["product_id"].unique()).difference(products["product_id"])
    unknown_stores = set(sales["store_id"].unique()).difference(stores["store_id"])
    if unknown_products or unknown_stores:
        raise ValueError(
            f"Sales rows contain unknown dimension keys: products={sorted(unknown_products)}, "
            f"stores={sorted(unknown_stores)}"
        )

    dates = pd.date_range(sales["date"].min(), sales["date"].max(), freq="D")
    combos = pd.MultiIndex.from_product(
        [dates, stores["store_id"].drop_duplicates(), products["product_id"].drop_duplicates()],
        names=KEY_COLUMNS,
    )
    panel = sales.set_index(KEY_COLUMNS).reindex(combos).reset_index()
    panel = panel.merge(products, on="product_id", how="left", validate="many_to_one")
    panel = panel.merge(stores, on="store_id", how="left", validate="many_to_one")
    panel["units_sold"] = panel["units_sold"].fillna(0).astype("int32")
    panel["promo_flag"] = panel["promo_flag"].fillna(0).astype("int8")
    panel = panel.sort_values(KEY_COLUMNS).reset_index(drop=True)

    # Prices carry forward within each SKU/store when a day is missing; leading gaps use product base price.
    panel["unit_price"] = panel.groupby(["store_id", "product_id"], sort=False)["unit_price"].ffill()
    panel["unit_price"] = panel["unit_price"].fillna(panel["base_price"])
    panel["unit_price"] = panel["unit_price"].clip(lower=0.01)
    panel["sales_amount"] = panel["units_sold"] * panel["unit_price"]
    panel.attrs["duplicate_rows_removed"] = duplicates
    return panel, inventory


def run_eda(panel: pd.DataFrame, output_dir: Path = PROCESSED_DIR) -> None:
    """Save daily, weekly, monthly, category, promotion, and store-level diagnostic plots."""
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    daily = panel.groupby("date", as_index=False)["units_sold"].sum()
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    axes[0].plot(daily["date"], daily["units_sold"], linewidth=0.8)
    axes[0].set_title("Daily units sold")
    weekly = panel.set_index("date").groupby(pd.Grouper(freq="W"))["units_sold"].sum()
    axes[1].plot(weekly.index, weekly.values)
    axes[1].set_title("Weekly units sold")
    monthly = panel.set_index("date").groupby(pd.Grouper(freq="MS"))["units_sold"].sum()
    axes[2].plot(monthly.index, monthly.values, marker="o")
    axes[2].set_title("Monthly units sold")
    fig.tight_layout()
    fig.savefig(figures / "sales_trends.png", dpi=150)
    plt.close(fig)

    category = panel.groupby("category", as_index=False)["units_sold"].sum().sort_values("units_sold")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=category, x="units_sold", y="category", ax=ax)
    ax.set_title("Units sold by category")
    fig.tight_layout()
    fig.savefig(figures / "category_performance.png", dpi=150)
    plt.close(fig)

    promo = panel.groupby("promo_flag", as_index=False)["units_sold"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=promo, x="promo_flag", y="units_sold", ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No promotion", "Promotion"])
    ax.set_title("Average daily units by promotion status")
    fig.tight_layout()
    fig.savefig(figures / "promotion_lift.png", dpi=150)
    plt.close(fig)

    store = panel.groupby("store_id", as_index=False)["units_sold"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=store, x="store_id", y="units_sold", ax=ax)
    ax.set_title("Average daily units by store")
    fig.tight_layout()
    fig.savefig(figures / "store_patterns.png", dpi=150)
    plt.close(fig)


def calculate_statistics(panel: pd.DataFrame, output_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export descriptive statistics and product-level coefficient-of-variation flags."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = panel["units_sold"].agg(["mean", "median", "std"]).to_frame().T
    summary["q1"] = panel["units_sold"].quantile(0.25)
    summary["q3"] = panel["units_sold"].quantile(0.75)
    summary["iqr"] = summary["q3"] - summary["q1"]
    upper = summary.iloc[0]["q3"] + 3 * summary.iloc[0]["iqr"]
    summary["spike_threshold_units"] = upper
    summary["spike_observation_count"] = int((panel["units_sold"] > upper).sum())
    summary.to_csv(output_dir / "demand_statistics.csv", index=False)

    variability = (
        panel.groupby(["product_id", "category"])["units_sold"]
        .agg(mean_units="mean", std_units="std", median_units="median")
        .reset_index()
    )
    variability["coefficient_of_variation"] = variability["std_units"] / variability["mean_units"].replace(0, np.nan)
    variability["coefficient_of_variation"] = variability["coefficient_of_variation"].fillna(0)
    variability["variability_rank"] = variability["coefficient_of_variation"].rank(method="dense", ascending=False).astype(int)
    variability.to_csv(output_dir / "product_demand_variability.csv", index=False)
    return summary, variability


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Create strictly backward-looking demand features for rolling one-day-ahead forecasts."""
    frame = panel.sort_values(["store_id", "product_id", "date"]).copy()
    groups = frame.groupby(["store_id", "product_id"], sort=False)
    for lag in (7, 14, 28):
        frame[f"lag_{lag}"] = groups["units_sold"].shift(lag)
    frame["rolling_mean_7"] = groups["units_sold"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=7).mean()
    )
    frame["rolling_mean_30"] = groups["units_sold"].transform(
        lambda values: values.shift(1).rolling(30, min_periods=30).mean()
    )
    frame["rolling_std_7"] = groups["units_sold"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=7).std()
    )
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype("int8")

    # Recurring retail peak windows plus fixed holidays; no third-party calendar dependency.
    date = frame["date"]
    holiday = (
        ((date.dt.month == 12) & (date.dt.day >= 15))
        | ((date.dt.month == 1) & (date.dt.day <= 2))
        | ((date.dt.month == 11) & (date.dt.day >= 20) & (date.dt.day <= 30))
        | ((date.dt.month == 7) & (date.dt.day <= 5))
        | ((date.dt.month == 5) & (date.dt.day >= 8) & (date.dt.day <= 14))
    )
    frame["holiday_flag"] = holiday.astype("int8")
    frame["price_change_pct"] = (
        frame.groupby(["store_id", "product_id"], sort=False)["unit_price"].pct_change().replace([np.inf, -np.inf], np.nan)
    )
    frame["price_change_pct"] = frame["price_change_pct"].fillna(0).clip(-1, 1)
    return frame.dropna(subset=FEATURE_COLUMNS + ["units_sold"]).reset_index(drop=True)


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def _chronological_ranges(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = np.array(sorted(frame["date"].unique()))
    if len(dates) < 30:
        raise ValueError("At least 30 feature dates are required for chronological train/validation/test splits")
    train_end = pd.Timestamp(dates[int(len(dates) * 0.70) - 1])
    validation_end = pd.Timestamp(dates[int(len(dates) * 0.85) - 1])
    return train_end, validation_end


def fit_and_evaluate(frame: pd.DataFrame, output_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune on a chronological validation window and report untouched chronological test metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_end, validation_end = _chronological_ranges(frame)
    train = frame[frame["date"] <= train_end]
    validation = frame[(frame["date"] > train_end) & (frame["date"] <= validation_end)]
    test = frame[frame["date"] > validation_end]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition")

    x_train, y_train = train[FEATURE_COLUMNS], train["units_sold"]
    x_validation, y_validation = validation[FEATURE_COLUMNS], validation["units_sold"]
    candidates: dict[str, list[dict[str, Any]]] = {
        "Linear Regression": [{}],
        "Decision Tree": [
            {"max_depth": 10, "min_samples_leaf": 10, "random_state": 42},
            {"max_depth": 20, "min_samples_leaf": 5, "random_state": 42},
        ],
        "Random Forest": [
            {"n_estimators": 60, "max_depth": 14, "min_samples_leaf": 5, "max_features": 0.8, "n_jobs": -1, "random_state": 42},
            {"n_estimators": 80, "max_depth": 20, "min_samples_leaf": 3, "max_features": 0.8, "n_jobs": -1, "random_state": 42},
        ],
    }
    constructors = {
        "Linear Regression": LinearRegression,
        "Decision Tree": DecisionTreeRegressor,
        "Random Forest": RandomForestRegressor,
    }
    selected: dict[str, tuple[dict[str, Any], float]] = {}
    for name, parameter_sets in candidates.items():
        best_params: dict[str, Any] | None = None
        best_mae = float("inf")
        for parameters in parameter_sets:
            model = constructors[name](**parameters)
            model.fit(x_train, y_train)
            score = mean_absolute_error(y_validation, np.maximum(model.predict(x_validation), 0))
            if score < best_mae:
                best_params, best_mae = parameters, float(score)
        if best_params is None:
            raise RuntimeError(f"No hyperparameter candidate was evaluated for {name}")
        selected[name] = (best_params, best_mae)

    development = pd.concat([train, validation], ignore_index=True)
    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for name, (parameters, validation_mae) in selected.items():
        model = constructors[name](**parameters)
        model.fit(development[FEATURE_COLUMNS], development["units_sold"])
        predicted = np.maximum(model.predict(test[FEATURE_COLUMNS]), 0)
        result = test[KEY_COLUMNS + ["units_sold", "unit_price", "promo_flag"]].copy()
        result["forecast_units"] = predicted
        result["model"] = name
        predictions.append(result.rename(columns={"date": "forecast_date", "units_sold": "actual_units"}))
        metric_rows.append(
            {
                "model": name,
                "validation_mae": validation_mae,
                "train_end": train_end.date().isoformat(),
                "validation_end": validation_end.date().isoformat(),
                "test_start": pd.Timestamp(test["date"].min()).date().isoformat(),
                "test_end": pd.Timestamp(test["date"].max()).date().isoformat(),
                **_metrics(test["units_sold"], predicted),
            }
        )

    predictions_frame = pd.concat(predictions, ignore_index=True)
    metrics_frame = pd.DataFrame(metric_rows).sort_values("mae")
    predictions_frame.to_csv(output_dir / "actuals_vs_forecasts.csv", index=False)
    metrics_frame.to_csv(output_dir / "forecast_metrics.csv", index=False)

    best_model_name = str(metrics_frame.iloc[0]["model"])
    best_predictions = predictions_frame[predictions_frame["model"] == best_model_name]
    product_error = pd.DataFrame(
        [{"product_id": key, **_metrics(group["actual_units"], group["forecast_units"])}
         for key, group in best_predictions.groupby("product_id")]
    )
    store_error = pd.DataFrame(
        [{"store_id": key, **_metrics(group["actual_units"], group["forecast_units"])}
         for key, group in best_predictions.groupby("store_id")]
    )
    product_error.to_csv(output_dir / "error_analysis_by_product.csv", index=False)
    store_error.to_csv(output_dir / "error_analysis_by_store.csv", index=False)
    return predictions_frame, metrics_frame


def export_inventory_risk(
    predictions: pd.DataFrame,
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Estimate lead-time stock exposure from the best model's recent daily forecasts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(output_dir / "forecast_metrics.csv").sort_values("mae")
    best_model = metrics.iloc[0]["model"]
    recent = predictions[predictions["model"] == best_model].copy()
    recent = recent[recent["forecast_date"] >= recent["forecast_date"].max() - pd.Timedelta(days=27)]
    demand = (
        recent.groupby(["store_id", "product_id"], as_index=False)["forecast_units"]
        .mean()
        .rename(columns={"forecast_units": "avg_daily_forecast_units"})
    )
    risk = inventory.merge(products[["product_id", "lead_time_days"]], on="product_id", how="left", validate="many_to_one")
    risk = risk.merge(demand, on=["store_id", "product_id"], how="left", validate="one_to_one")
    risk["avg_daily_forecast_units"] = risk["avg_daily_forecast_units"].fillna(0)
    risk["lead_time_demand_units"] = risk["avg_daily_forecast_units"] * risk["lead_time_days"]
    risk["expected_stockout_units"] = np.maximum(
        risk["lead_time_demand_units"] - risk["current_stock_on_hand"], 0
    )
    risk["stockout_exposure_flag"] = (
        risk["current_stock_on_hand"] < risk["lead_time_demand_units"]
    ).astype("int8")
    risk["inventory_coverage_days"] = np.where(
        risk["avg_daily_forecast_units"] > 0,
        risk["current_stock_on_hand"] / risk["avg_daily_forecast_units"],
        np.nan,
    )
    risk["reorder_priority_score"] = np.where(
        risk["lead_time_demand_units"] > 0,
        100
        * np.clip(
            (risk["lead_time_demand_units"] - risk["current_stock_on_hand"])
            / risk["lead_time_demand_units"],
            0,
            1,
        ),
        0,
    )
    risk.to_csv(output_dir / "inventory_risk.csv", index=False)
    return risk


def export_data_dictionary(output_dir: Path = PROCESSED_DIR) -> None:
    """Write a field-level data dictionary for the generated processed outputs."""
    definitions = [
        ("actuals_vs_forecasts.csv", "forecast_date", "Date being forecast (daily, rolling one-day-ahead backtest)."),
        ("actuals_vs_forecasts.csv", "store_id / product_id", "Retail store and SKU keys."),
        ("actuals_vs_forecasts.csv", "actual_units", "Observed units sold on forecast_date."),
        ("actuals_vs_forecasts.csv", "forecast_units", "Non-negative model prediction for units sold."),
        ("actuals_vs_forecasts.csv", "unit_price / promo_flag", "Observed/planned price and promotion input on that date."),
        ("actuals_vs_forecasts.csv", "model", "Model that generated the forecast."),
        ("forecast_metrics.csv", "mae / rmse / r2", "Test-set error metrics; lower MAE/RMSE and higher R-squared are preferred."),
        ("forecast_metrics.csv", "validation_mae", "Validation MAE used for model parameter selection."),
        ("inventory_risk.csv", "current_stock_on_hand / reorder_point", "Latest synthetic stock and policy threshold."),
        ("inventory_risk.csv", "avg_daily_forecast_units", "Mean forecast over the latest 28 test dates from the best-MAE model."),
        ("inventory_risk.csv", "lead_time_demand_units", "Average daily forecast multiplied by product lead time."),
        ("inventory_risk.csv", "expected_stockout_units", "Positive lead-time demand less on-hand stock; a planning exposure, not a lost-sales observation."),
        ("inventory_risk.csv", "inventory_coverage_days", "On-hand units divided by average forecast demand per day."),
        ("inventory_risk.csv", "reorder_priority_score", "0-100 unmet share of lead-time demand; 100 denotes zero on-hand stock."),
        ("demand_statistics.csv", "mean / median / std / q1 / q3 / iqr", "Descriptive statistics over the completed daily panel."),
        ("product_demand_variability.csv", "coefficient_of_variation", "Product standard deviation divided by mean daily units."),
        ("error_analysis_by_product.csv and error_analysis_by_store.csv", "mae / rmse / r2", "Test error slices for the globally best-MAE model."),
    ]
    pd.DataFrame(definitions, columns=["file", "field", "definition"]).to_csv(
        output_dir / "data_dictionary.csv", index=False
    )
    lines = ["# Processed Data Dictionary", "", "Forecasts are rolling one-day-ahead backtests: prior-day actual sales are available when each daily forecast is issued.", ""]
    lines.extend(f"- **{field}** (`{filename}`): {definition}" for filename, field, definition in definitions)
    (output_dir / "data_dictionary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline() -> None:
    """Execute the complete pipeline from generated raw files to processed artifacts."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel, inventory = load_and_prepare()
    run_eda(panel)
    calculate_statistics(panel)
    featured = add_features(panel)
    predictions, _ = fit_and_evaluate(featured)
    products = _read_csv(RAW_DIR / "products.csv", PRODUCT_COLUMNS)
    export_inventory_risk(predictions, inventory, products)
    export_data_dictionary()
    print(f"Pipeline complete. Processed outputs: {PROCESSED_DIR}")
