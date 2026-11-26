"""Responsible synthetic healthcare risk analysis and evaluation pipeline.

Run generate_healthcare_data.py first, then run this file or the companion notebook.
Only de-identified synthetic source data are accepted by this project workflow.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TARGET = "adverse_outcome_30d"
RANDOM_STATE = 2025
PRIVACY_MIN_CELL = 30

RANGE_LIMITS = {
    "length_of_stay_days": (0, 90),
    "systolic_bp": (70, 250),
    "heart_rate": (30, 220),
    "glucose_mg_dl": (40, 600),
    "bmi": (10, 80),
    "comorbidity_count": (0, 20),
    "prior_admissions_12m": (0, 30),
}
MEASUREMENT_UNITS = {
    "length_of_stay_days": "days",
    "systolic_bp": "mmHg",
    "heart_rate": "beats/minute",
    "glucose_mg_dl": "mg/dL",
    "bmi": "kg/m^2",
    "comorbidity_count": "count",
    "prior_admissions_12m": "count in prior 12 months",
}
CATEGORICAL_FEATURES = [
    "age_group",
    "department_id",
    "admission_type",
    "lab_test_ordered",
]


def load_and_clean(path: Path = RAW_DIR / "patient_records.csv") -> tuple[pd.DataFrame, dict]:
    """Validate schema, remove duplicate IDs, and flag/clean implausible measurements."""
    data = pd.read_csv(path)
    required = {
        "encounter_id", "age_group", "sex", "department_id", "admission_type",
        *RANGE_LIMITS, "lab_test_ordered", TARGET,
    }
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")
    if data["encounter_id"].isna().any():
        raise ValueError("encounter_id contains null values; refusing ambiguous deduplication.")

    before = len(data)
    data = data.drop_duplicates(subset="encounter_id", keep="first").copy()
    duplicate_count = before - len(data)

    for column, (lower, upper) in RANGE_LIMITS.items():
        values = pd.to_numeric(data[column], errors="coerce")
        invalid = values.notna() & ~values.between(lower, upper)
        data[f"{column}_invalid"] = invalid.astype("int8")
        values = values.mask(invalid)
        data[f"{column}_missing"] = values.isna().astype("int8")
        data[column] = values
    for column in CATEGORICAL_FEATURES:
        data[f"{column}_missing"] = data[column].isna().astype("int8")

    if not set(data[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only binary 0/1 values.")
    if data[TARGET].isna().any():
        raise ValueError(f"{TARGET} has missing labels; resolve them before supervised modeling.")
    data[TARGET] = data[TARGET].astype(int)
    if data.empty or data[TARGET].nunique() != 2:
        raise ValueError("Cleaned data must be non-empty and contain both outcome classes.")

    report = {
        "input_rows": before,
        "duplicate_encounter_ids_removed": duplicate_count,
        "analysis_rows": len(data),
        "validated_measurement_units": MEASUREMENT_UNITS,
        "invalid_measurements_by_column": {
            column: int(data[f"{column}_invalid"].sum()) for column in RANGE_LIMITS
        },
        "missing_values_by_column": {
            column: int(data[f"{column}_missing"].sum()) for column in RANGE_LIMITS
        },
        "outcome_rate": float(data[TARGET].mean()),
    }
    return data, report


def build_pipeline(estimator) -> Pipeline:
    """Build leakage-safe imputation, scaling, and one-hot encoding plus estimator."""
    numeric_features = list(RANGE_LIMITS) + [
        f"{column}_invalid" for column in RANGE_LIMITS
    ] + [f"{column}_missing" for column in list(RANGE_LIMITS) + CATEGORICAL_FEATURES]
    numeric_transform = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transform = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transform, numeric_features),
            ("categorical", categorical_transform, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessing), ("classifier", estimator)])


def _show_or_save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _safe_metric_row(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "sample_size": int(len(y_true)),
        "positive_count": int(np.sum(y_true)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_true.nunique() == 2 else np.nan,
        "predicted_positive_rate": float(np.mean(y_pred)),
        "observed_rate": float(np.mean(y_true)),
        "calibration_gap": float(abs(np.mean(y_prob) - np.mean(y_true))),
    }


def _subgroup_metrics(
    test_frame: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    audit = test_frame[["age_group", "sex"]].reset_index(drop=True).copy()
    audit["actual"] = np.asarray(y_true)
    audit["predicted"] = y_pred
    audit["probability"] = y_prob
    rows = []
    for dimension in ("age_group", "sex"):
        for subgroup, group in audit.groupby(dimension, observed=True):
            positives = int(group["actual"].sum())
            negatives = len(group) - positives
            row = {
                "dimension": dimension,
                "subgroup": str(subgroup),
                "sample_size": int(len(group)),
                "positive_count": positives,
                "negative_count": negatives,
                "suppressed": bool(
                    len(group) < PRIVACY_MIN_CELL or positives < 10 or negatives < 10
                ),
            }
            if row["suppressed"]:
                row.update(
                    {
                        "recall": np.nan,
                        "precision": np.nan,
                        "f1": np.nan,
                        "roc_auc": np.nan,
                        "predicted_positive_rate": np.nan,
                        "observed_rate": np.nan,
                        "calibration_gap": np.nan,
                        "parity_ratio": np.nan,
                    }
                )
            else:
                metrics = _safe_metric_row(
                    group["actual"], group["predicted"].to_numpy(), group["probability"].to_numpy()
                )
                row.update({name: metrics[name] for name in (
                    "recall", "precision", "f1", "roc_auc",
                    "predicted_positive_rate", "observed_rate", "calibration_gap",
                )})
                row["parity_ratio"] = np.nan
            rows.append(row)
    result = pd.DataFrame(rows)
    for dimension in result["dimension"].unique():
        selected = result["dimension"].eq(dimension) & ~result["suppressed"]
        rates = result.loc[selected, "predicted_positive_rate"]
        if len(rates) and rates.max() > 0:
            result.loc[selected, "parity_ratio"] = rates / rates.max()
    return result


def run_pipeline() -> dict:
    """Execute cleaning, EDA, model validation, fairness audit, and aggregate exports."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    data, quality = load_and_clean()
    with (PROCESSED_DIR / "data_quality_report.json").open("w", encoding="utf-8") as file:
        json.dump(quality, file, indent=2)
    print("Data quality report:", json.dumps(quality, indent=2))

    # Descriptive statistics and outcome comparisons.
    numeric_summary = data[list(RANGE_LIMITS)].describe().T
    numeric_summary["median"] = data[list(RANGE_LIMITS)].median()
    numeric_summary["iqr"] = (
        data[list(RANGE_LIMITS)].quantile(0.75) - data[list(RANGE_LIMITS)].quantile(0.25)
    )
    numeric_summary.to_csv(PROCESSED_DIR / "descriptive_statistics.csv")
    print("\nNumeric summary (mean, std, median, IQR):\n", numeric_summary[
        ["mean", "std", "median", "iqr"]
    ].round(2))
    print("\nMeasurement comparison by outcome:\n", data.groupby(TARGET)[
        list(RANGE_LIMITS)
    ].agg(["mean", "median", "std"]).round(2))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.histplot(data=data, x="age_group", hue=TARGET, multiple="dodge",
                 discrete=True, ax=axes[0, 0])
    sns.histplot(data=data, x="systolic_bp", hue=TARGET, element="step",
                 stat="density", common_norm=False, ax=axes[0, 1])
    sns.boxplot(data=data, x=TARGET, y="glucose_mg_dl", ax=axes[1, 0])
    sns.countplot(data=data, x="department_id", hue=TARGET, ax=axes[1, 1])
    axes[0, 0].set_title("Encounter counts by age group and outcome")
    axes[0, 1].set_title("Validated systolic blood pressure")
    axes[1, 0].set_title("Glucose distribution by outcome")
    axes[1, 1].set_title("Department sample sizes and outcomes")
    _show_or_save(fig, PROCESSED_DIR / "exploratory_distributions.png")

    department_age = data.groupby(["age_group", "department_id"], observed=True)[TARGET].agg(
        sample_size="size", adverse_outcome_rate="mean"
    ).reset_index()
    department_age.to_csv(PROCESSED_DIR / "department_age_outcome_rates.csv", index=False)
    age_weights = data["age_group"].value_counts(normalize=True)
    age_rate_matrix = department_age.pivot(
        index="department_id", columns="age_group", values="adverse_outcome_rate"
    ).reindex(columns=age_weights.index)
    age_standardized_rates = age_rate_matrix.mul(age_weights, axis=1).sum(axis=1)
    crude_department_rates = data.groupby("department_id", observed=True)[TARGET].agg(
        sample_size="size", crude_rate="mean"
    )
    department_comparison = crude_department_rates.join(
        age_standardized_rates.rename("age_standardized_rate")
    )
    department_comparison["standardization_change"] = (
        department_comparison["age_standardized_rate"] - department_comparison["crude_rate"]
    )
    department_comparison.reset_index().to_csv(
        PROCESSED_DIR / "department_age_adjusted_comparison.csv", index=False
    )
    department_table = pd.crosstab(data["department_id"], data[TARGET])
    age_table = pd.crosstab(data["age_group"], data[TARGET])
    dept_chi2, dept_p, _, _ = chi2_contingency(department_table)
    age_chi2, age_p, _, _ = chi2_contingency(age_table)
    print(
        "\nUnadjusted association checks (chi-square; descriptive, not causal): "
        f"department p={dept_p:.4g}, age-group p={age_p:.4g}"
    )
    print(
        "Compare crude and age-standardized department rates in "
        "department_age_adjusted_comparison.csv; age-specific rates and support are in "
        "department_age_outcome_rates.csv."
    )
    if (department_age["sample_size"] < PRIVACY_MIN_CELL).any():
        warnings.warn("Some age-by-department cells are small; interpret rates cautiously.")

    # Keep sex out of predictors; retain it solely for subgroup fairness auditing.
    feature_columns = (
        CATEGORICAL_FEATURES
        + list(RANGE_LIMITS)
        + [f"{column}_invalid" for column in RANGE_LIMITS]
        + [f"{column}_missing" for column in list(RANGE_LIMITS) + CATEGORICAL_FEATURES]
    )
    X = data[feature_columns].copy()
    y = data[TARGET].copy()
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, data.index, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=40, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=15, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Feed-Forward Neural Network": MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=0.01, early_stopping=True,
            max_iter=250, random_state=RANDOM_STATE
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted_models = {}
    evaluation_rows = []
    predictions = {}
    for name, estimator in models.items():
        pipeline = build_pipeline(estimator)
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_test)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        metrics = _safe_metric_row(y_test, predicted, probability)
        metrics.update(
            {
                "model": name,
                "cv_roc_auc_mean": float(cv_auc.mean()),
                "cv_roc_auc_std": float(cv_auc.std(ddof=1)),
                "threshold": 0.5,
            }
        )
        evaluation_rows.append(metrics)
        fitted_models[name] = pipeline
        predictions[name] = (predicted, probability)
        print(f"\n{name}: {metrics}")
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, predicted, labels=[0, 1]),
            display_labels=["No adverse outcome", "Adverse outcome"],
        ).plot(cmap="Blues", values_format="d")
        plt.title(f"{name} - held-out confusion matrix")
        plt.tight_layout()
        plt.savefig(PROCESSED_DIR / f"confusion_matrix_{name.lower().replace(' ', '_').replace('-', '_')}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    evaluation = pd.DataFrame(evaluation_rows).sort_values(
        "cv_roc_auc_mean", ascending=False
    ).reset_index(drop=True)
    evaluation.to_csv(PROCESSED_DIR / "model_evaluation_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (_, probability) in predictions.items():
        observed_fraction, mean_probability = calibration_curve(
            y_test, probability, n_bins=10, strategy="quantile"
        )
        ax.plot(mean_probability, observed_fraction, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed event fraction",
           title="Held-out calibration curves")
    ax.legend()
    _show_or_save(fig, PROCESSED_DIR / "calibration_curves.png")

    best_name = str(evaluation.iloc[0]["model"])
    best_predicted, best_probability = predictions[best_name]
    test_frame = data.loc[idx_test].reset_index(drop=True)
    fairness = _subgroup_metrics(test_frame, y_test.reset_index(drop=True),
                                 best_predicted, best_probability)
    fairness.to_csv(PROCESSED_DIR / "agg_subgroup_fairness.csv", index=False)
    print(f"\nSubgroup audit for CV-selected model ({best_name}):\n", fairness.to_string(index=False))

    # Privacy-safe, one-dimension-at-a-time cohort aggregates from held-out encounters.
    cohort_source = test_frame.copy()
    cohort_source["actual"] = y_test.to_numpy()
    cohort_source["predicted"] = best_predicted
    cohort_source["probability"] = best_probability
    cohort_rows = []
    for dimension in ("age_group", "sex", "department_id", "admission_type"):
        for value, group in cohort_source.groupby(dimension, observed=True):
            n = len(group)
            suppressed = n < PRIVACY_MIN_CELL
            cohort_rows.append(
                {
                    "cohort_dimension": dimension,
                    "cohort_value": str(value),
                    "cohort_size": np.nan if suppressed else n,
                    "adverse_outcomes": np.nan if suppressed else int(group["actual"].sum()),
                    "adverse_outcome_rate": np.nan if suppressed else float(group["actual"].mean()),
                    "high_risk_count": np.nan if suppressed else int(group["predicted"].sum()),
                    "high_risk_share": np.nan if suppressed else float(group["predicted"].mean()),
                    "mean_length_of_stay_days": np.nan if suppressed else float(
                        group["length_of_stay_days"].mean()
                    ),
                    "is_suppressed": suppressed,
                }
            )
    cohorts = pd.DataFrame(cohort_rows)
    cohorts.to_csv(PROCESSED_DIR / "agg_cohort_risk.csv", index=False)

    # Export model interpretation; coefficients are associations, not causal effects.
    lr = fitted_models["Logistic Regression"]
    names = lr.named_steps["preprocess"].get_feature_names_out()
    coefficients = lr.named_steps["classifier"].coef_[0]
    pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients,
            "odds_ratio_per_encoded_unit": np.exp(np.clip(coefficients, -20, 20)),
        }
    ).sort_values("coefficient", key=lambda values: values.abs(), ascending=False).to_csv(
        PROCESSED_DIR / "logistic_regression_coefficients.csv", index=False
    )
    rf = fitted_models["Random Forest"]
    pd.DataFrame(
        {
            "feature": rf.named_steps["preprocess"].get_feature_names_out(),
            "importance": rf.named_steps["classifier"].feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        PROCESSED_DIR / "random_forest_feature_importances.csv", index=False
    )

    best_metrics = evaluation.loc[evaluation["model"].eq(best_name)].iloc[0].to_dict()
    model_card = f"""# Model Card: Synthetic 30-Day Adverse Outcome Risk

## Summary
- **Model selected for reporting:** {best_name}, selected by mean 5-fold stratified training ROC-AUC.
- **Evaluation:** held-out stratified test set; threshold = 0.50.
- **Synthetic outcome prevalence:** {quality['outcome_rate']:.1%} before deduplication.
- **Test ROC-AUC / recall / precision / F1:** {best_metrics['roc_auc']:.3f} / {best_metrics['recall']:.3f} / {best_metrics['precision']:.3f} / {best_metrics['f1']:.3f}.
- **Test calibration gap** (absolute difference in mean predicted and observed rate): {best_metrics['calibration_gap']:.3f}.

## Intended use
Educational demonstration and aggregate analytics using wholly synthetic data. This model is a research prototype and decision-support illustration only; it is **not for diagnosis, triage, treatment selection, or clinical judgment**.

## Data and methods
Rows with duplicate de-identified encounter IDs are reduced to the first occurrence. Measurements outside documented plausible ranges are set to missing and retained with invalid-value flags. Numeric missing values are median-imputed and categorical values are most-frequent-imputed inside each training fold; missingness flags are retained. The model uses age band, department, admission type, laboratory-order status, clinical measurements, and quality flags. Sex is excluded from predictors and retained for auditing. Test performance is reported once after 5-fold stratified cross-validation on training data.

## Performance and calibration
See `model_evaluation_metrics.csv`, `calibration_curves.png`, and the confusion-matrix plots for the complete comparison. The selected model is chosen using training cross-validation only, not the held-out test results. A 0.50 threshold is illustrative and has not been optimized for a clinical utility function.

## Fairness checks
Held-out recall, precision, F1, ROC-AUC, predicted-positive rate, calibration gap, and a min-to-max predicted-positive-rate parity ratio are reported by age group and sex in `agg_subgroup_fairness.csv`. Small or low-event cells are suppressed (fewer than {PRIVACY_MIN_CELL} records, or fewer than 10 positives or negatives). These are diagnostic summaries, not evidence of fairness; uncertainty intervals, clinical context, intersectional analysis, and stakeholder review are still required.

## Limitations and responsible-use constraints
- Data are synthetic and do not establish clinical validity, transportability, or real-world utility.
- Associations and model explanations are correlational, not causal; odds ratios and feature importances can be distorted by encoding, collinearity, missingness, and the synthetic data-generating process.
- The target definition, event ascertainment, measurement processes, and subgroup composition are simplified. Fairness metrics depend on outcome quality and threshold choice.
- This prototype has no external or temporal validation, prospective monitoring, clinician workflow assessment, or benefit-harm analysis. It must not be deployed on real patients.
- No patient-level predictions are exported. Only aggregated metrics and model summaries are written to `data/processed/`.
"""
    (PROCESSED_DIR / "model_card.md").write_text(model_card, encoding="utf-8")
    print(f"\nCompleted. Aggregated results and model card saved to {PROCESSED_DIR}.")
    return {
        "data_quality": quality,
        "evaluation": evaluation,
        "fairness": fairness,
        "cohorts": cohorts,
        "selected_model": best_name,
    }


if __name__ == "__main__":
    run_pipeline()
