"""Cleaning, EDA, statistics, modeling, evaluation, and export pipeline."""
import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 2026
SNAPSHOT = pd.Timestamp("2026-11-22")


def inspect_dtypes(frame, name):
    print(f"\n{name}: {frame.shape[0]:,} rows")
    print(frame.dtypes.to_string())


def iqrr_limits(series):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(lower), float(upper)


def clean_and_merge(data_dir):
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"), parse_dates=["signup_date"])
    subscriptions = pd.read_csv(os.path.join(data_dir, "subscriptions.csv"))
    billing = pd.read_csv(os.path.join(data_dir, "billing.csv"))
    support = pd.read_csv(os.path.join(data_dir, "support_tickets.csv"))

    for frame, name in [(customers, "customers"), (subscriptions, "subscriptions"),
                        (billing, "billing"), (support, "support_tickets")]:
        inspect_dtypes(frame, name)

    # Drop duplicate customer keys; keep the first, auditable record.
    subscriptions = subscriptions.drop_duplicates("customer_id", keep="first")
    billing = billing.drop_duplicates("customer_id", keep="first")
    support = support.drop_duplicates("customer_id", keep="first")
    customers = customers.drop_duplicates("customer_id", keep="first")

    merged = customers.merge(subscriptions, on="customer_id", how="left", validate="one_to_one")
    merged = merged.merge(billing, on="customer_id", how="left", validate="one_to_one")
    merged = merged.merge(support, on="customer_id", how="left", validate="one_to_one")

    numeric_columns = [
        "age", "monthly_fee", "tenure_months", "avg_monthly_usage_gb",
        "usage_trend_6m", "add_on_count", "last_payment_delay_days",
        "missed_payments_12m", "total_late_fees", "billing_issue_count",
        "support_tickets_12m", "open_tickets", "avg_resolution_hours",
        "satisfaction_score",
    ]
    categorical_columns = ["region", "subscription_plan", "contract_type", "payment_method"]

    # Numeric missing values are median-imputed because right-skew and outliers are common.
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        median = merged[column].median()
        merged[column] = merged[column].fillna(median)

    for column in categorical_columns:
        merged[column] = merged[column].fillna("Unknown").astype("object")

    # IQR limits are calculated on cleaned merged data and only used for model-safe winsorization.
    outlier_columns = ["monthly_fee", "avg_monthly_usage_gb", "last_payment_delay_days",
                       "missed_payments_12m", "support_tickets_12m", "avg_resolution_hours"]
    outlier_flags = {}
    for column in outlier_columns:
        lower, upper = iqrr_limits(merged[column])
        outlier_flags[column] = (lower, upper)
        flag = f"{column}_outlier"
        merged[flag] = ((merged[column] < lower) | (merged[column] > upper)).astype(int)
        merged[column] = merged[column].clip(lower, upper)

    return merged, outlier_flags


def save_eda(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    churn_tenure = df.groupby("tenure_band", observed=True)["churn"].mean().sort_index()
    sns.barplot(x=churn_tenure.index, y=churn_tenure.values, ax=axes[0, 0], palette="Reds")
    axes[0, 0].set_title("Churn Rate by Tenure Band")
    axes[0, 0].set_ylabel("Churn Rate")

    plan = df.groupby("subscription_plan")["churn"].agg(["count", "mean"]).reset_index()
    sns.barplot(data=plan, x="subscription_plan", y="mean", ax=axes[0, 1], palette="Blues")
    axes[0, 1].set_title("Churn Rate by Plan")
    axes[0, 1].set_ylabel("Churn Rate")

    payment = df.groupby("payment_method")["churn"].agg(["count", "mean"]).reset_index()
    sns.barplot(data=payment, x="payment_method", y="mean", ax=axes[1, 0], palette="Purples")
    axes[1, 0].set_title("Churn Rate by Payment Method")
    axes[1, 0].set_ylabel("Churn Rate")
    axes[1, 0].tick_params(axis="x", rotation=25)

    df["usage_band"] = pd.cut(df["avg_monthly_usage_gb"], bins=[-np.inf, 25, 50, 75, np.inf],
                              labels=["0-25 GB", "25-50 GB", "50-75 GB", "75+ GB"])
    usage = df.groupby("usage_band", observed=True)["churn"].agg(["count", "mean"]).reset_index()
    sns.barplot(data=usage, x="usage_band", y="mean", ax=axes[1, 1], palette="Greens")
    axes[1, 1].set_title("Churn Rate by Service Usage")
    axes[1, 1].set_ylabel("Churn Rate")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "eda_churn_drivers.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    corr_columns = ["churn", "tenure_months", "monthly_fee", "avg_monthly_usage_gb",
                    "missed_payments_12m", "last_payment_delay_days",
                    "support_tickets_12m", "open_tickets", "satisfaction_score"]
    sns.heatmap(df[corr_columns].corr(numeric_only=True), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=180)
    plt.close(fig)


def statistical_investigation(df, output_dir):
    results = []

    for column in ["contract_type", "subscription_plan", "payment_method"]:
        table = pd.crosstab(df[column], df["churn"])
        chi2, pvalue, dof, expected = chi2_contingency(table)
        results.append({"test": f"Chi-square: churn vs {column}", "statistic": chi2,
                        "p_value": pvalue, "detail": f"dof={dof}"})

    for column in ["tenure_months", "avg_monthly_usage_gb", "missed_payments_12m", "support_tickets_12m"]:
        retained = df.loc[df["churn"] == 0, column].dropna()
        churned = df.loc[df["churn"] == 1, column].dropna()
        statistic, pvalue = mannwhitneyu(retained, churned, alternative="two-sided")
        results.append({"test": f"Mann-Whitney U: retained vs churned {column}",
                        "statistic": statistic, "p_value": pvalue,
                        "detail": f"retained median={retained.median():.2f}; churned median={churned.median():.2f}"})

    pd.DataFrame(results).to_csv(os.path.join(output_dir, "statistical_results.csv"), index=False)

    numeric_stats = df[["tenure_months", "monthly_fee", "avg_monthly_usage_gb",
                        "last_payment_delay_days", "support_tickets_12m",
                        "satisfaction_score"]].describe(percentiles=[0.25, 0.5, 0.75]).T
    numeric_stats.to_csv(os.path.join(output_dir, "descriptive_statistics.csv"))


def engineer_features(df):
    result = df.copy()
    result["tenure_band"] = pd.cut(
        result["tenure_months"],
        bins=[-np.inf, 6, 12, 24, 48, np.inf],
        labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    )
    result["high_payment_delay"] = (result["last_payment_delay_days"] >= 15).astype(int)
    result["payment_delay_count_12m"] = result["missed_payments_12m"]
    result["negative_usage_trend"] = (result["usage_trend_6m"] < 0).astype(int)
    result["usage_decline_rate"] = result["usage_trend_6m"].clip(-1, 1)
    result["support_contact_frequency"] = result["support_tickets_12m"]
    result["high_support_contact"] = (result["support_tickets_12m"] >= 3).astype(int)
    result["month_to_month"] = (result["contract_type"] == "Month-to-Month").astype(int)
    result["is_two_year"] = (result["contract_type"] == "Two Year").astype(int)
    result["tenure_years"] = result["tenure_months"] / 12
    result["low_usage_flag"] = (result["avg_monthly_usage_gb"] < 25).astype(int)

    # Explicitly exclude churn and every raw target-derived field.
    feature_columns = [
        "age", "region", "subscription_plan", "contract_type", "monthly_fee",
        "tenure_months", "tenure_band", "avg_monthly_usage_gb", "usage_trend_6m",
        "payment_method", "add_on_count", "last_payment_delay_days",
        "missed_payments_12m", "total_late_fees", "billing_issue_count",
        "support_tickets_12m", "open_tickets", "avg_resolution_hours",
        "satisfaction_score", "high_payment_delay", "payment_delay_count_12m",
        "negative_usage_trend", "usage_decline_rate", "support_contact_frequency",
        "high_support_contact", "month_to_month", "is_two_year", "tenure_years",
        "low_usage_flag",
    ]
    return result, feature_columns


def build_preprocessor(numeric_features, categorical_features):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_features),
        ("cat", categorical, categorical_features),
    ])


def evaluate_model(name, model, X_test, y_test, threshold, output_dir):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "model": name,
        "threshold": threshold,
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "accuracy": (predictions == y_test).mean(),
    }
    cm = confusion_matrix(y_test, predictions)
    return metrics, cm, probabilities


def run_pipeline(data_dir="data", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    df, outlier_limits = clean_and_merge(data_dir)
    df["tenure_band"] = pd.cut(df["tenure_months"], bins=[-np.inf, 6, 12, 24, 48, np.inf],
                               labels=["0-6", "7-12", "13-24", "25-48", "49+"])
    save_eda(df, output_dir)
    statistical_investigation(df, output_dir)
    df, feature_columns = engineer_features(df)

    X = df[feature_columns]
    y = df["churn"]

    # 60% train / 20% validation / 20% test; test data remains untouched for threshold and metric selection.
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.50, random_state=RANDOM_STATE, stratify=y_remaining
    )

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    candidates = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
         {"model__C": [0.1, 1.0, 10.0]}),
        ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE),
         {"model__max_depth": [4, 6, 10], "model__min_samples_leaf": [10, 25]}),
        ("Random Forest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
         {"model__n_estimators": [200], "model__max_depth": [8, 12],
          "model__min_samples_leaf": [10, 20]}),
    ]

    from sklearn.model_selection import ParameterGrid
    all_metrics = []
    fitted_models = {}
    confusion_records = []
    feature_importance_records = []
    for name, estimator, grid in candidates:
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        best_model = None
        best_f1 = -1
        best_params = None
        for params in ParameterGrid(grid):
            model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            model.set_params(**params)
            model.fit(X_train, y_train)
            val_probability = model.predict_proba(X_validation)[:, 1]
            val_prediction = (val_probability >= 0.5).astype(int)
            val_f1 = f1_score(y_validation, val_prediction)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model = model
                best_params = params
        fitted_models[name] = best_model
        print(f"{name}: validation F1={best_f1:.4f}, params={best_params}")

    # Cost assumptions: false negative misses ₹500 revenue; false positive costs ₹50 offer.
    best_cost = np.inf
    best_threshold = 0.5
    threshold_grid = np.round(np.arange(0.05, 0.91, 0.01), 2)
    chosen_model_name = max(fitted_models, key=lambda n:
                            roc_auc_score(y_validation, fitted_models[n].predict_proba(X_validation)[:, 1]))
    chosen_model = fitted_models[chosen_model_name]
    for threshold in threshold_grid:
        probabilities = chosen_model.predict_proba(X_validation)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_validation, predictions).ravel()
        cost = fn * 500 + fp * 50
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)

    for name, model in fitted_models.items():
        metrics, cm, probabilities = evaluate_model(
            name, model, X_test, y_test, 0.5, output_dir
        )
        all_metrics.append(metrics)
        print(f"{name} test confusion matrix:\n{cm}")
        confusion_records.append({
            "model": name,
            "threshold": 0.5,
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        })

        if name == chosen_model_name:
            tuned_metrics, tuned_cm, tuned_probabilities = evaluate_model(
                name, model, X_test, y_test, best_threshold, output_dir
            )
            tuned_metrics["selection"] = "Selected model - validation-tuned threshold"
            all_metrics[-1]["selection"] = "Selected model - default threshold"
            all_metrics.append(tuned_metrics)
            print(f"Tuned threshold={best_threshold:.2f}, test confusion matrix:\n{tuned_cm}")
            confusion_records.append({
                "model": name,
                "threshold": best_threshold,
                "true_negative": int(tuned_cm[0, 0]),
                "false_positive": int(tuned_cm[0, 1]),
                "false_negative": int(tuned_cm[1, 0]),
                "true_positive": int(tuned_cm[1, 1]),
            })

    pd.DataFrame(all_metrics).to_csv(os.path.join(output_dir, "model_metrics.csv"), index=False)
    pd.DataFrame(confusion_records).to_csv(os.path.join(output_dir, "confusion_matrices.csv"), index=False)

    for name, model in fitted_models.items():
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        names = preprocessor.get_feature_names_out()
        if hasattr(estimator, "feature_importances_"):
            importance = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importance = np.abs(estimator.coef_[0])
        else:
            continue
        feature_importance_records.extend([
            {"model": name, "feature": str(feature), "importance": float(value)}
            for feature, value in sorted(zip(names, importance), key=lambda item: item[1], reverse=True)
        ])
    pd.DataFrame(feature_importance_records).to_csv(
        os.path.join(output_dir, "feature_importance.csv"), index=False
    )

    # Final scored predictions use the entire test/customer-safe data model and the tuned threshold.
    final_probabilities = chosen_model.predict_proba(X)[:, 1]
    final_predictions = (final_probabilities >= best_threshold).astype(int)
    score = pd.DataFrame({
        "customer_id": df["customer_id"],
        "churn_probability": final_probabilities,
        "predicted_churn": final_predictions,
    })
    score["retention_risk_segment"] = pd.cut(
        score["churn_probability"],
        bins=[-np.inf, 0.20, 0.50, 0.75, np.inf],
        labels=["Low", "Medium", "High", "Critical"],
    )
    score = score.merge(df[[ "customer_id", "tenure_months", "subscription_plan", "contract_type",
                             "payment_method", "monthly_fee", "avg_monthly_usage_gb",
                             "missed_payments_12m", "support_tickets_12m",
                             "satisfaction_score", "churn"]], on="customer_id", how="left")
    score.to_csv(os.path.join(output_dir, "final_scored_predictions.csv"), index=False)

    # Export customer profiles and a SQLite database for a direct SQL/Power BI workflow.
    df.to_csv(os.path.join(output_dir, "clean_customer_profiles.csv"), index=False)
    database_path = os.path.join(output_dir, "customer_churn.db")
    with sqlite3.connect(database_path) as connection:
        df.drop(columns=["usage_band"], errors="ignore").to_sql("clean_customer_profiles", connection,
                                                                 if_exists="replace", index=False)
        score.to_sql("churn_scores", connection, if_exists="replace", index=False)

    # Outlier and threshold documentation
    outlier_report = pd.DataFrame([
        {"column": column, "iqr_lower_limit": limits[0], "iqr_upper_limit": limits[1]}
        for column, limits in outlier_limits.items()
    ])
    outlier_report.to_csv(os.path.join(output_dir, "outlier_limits.csv"), index=False)
    threshold_report = pd.DataFrame([{
        "selected_model": chosen_model_name,
        "optimal_threshold": best_threshold,
        "validation_min_cost": best_cost,
        "missed_churn_cost": 500,
        "retention_offer_cost": 50,
        "justification": "Minimizes estimated ₹500 missed-churn cost plus ₹50 unnecessary-offer cost on validation data."
    }])
    threshold_report.to_csv(os.path.join(output_dir, "threshold_recommendation.csv"), index=False)

    print(f"\nOutput directory: {output_dir}")
    print(f"Selected model: {chosen_model_name}")
    print(f"Optimal threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    run_pipeline()
