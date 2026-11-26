# Project 01 - Customer Churn Prediction & Retention Intelligence

This project creates a synthetic telecom-style subscription dataset, cleans and merges it, performs EDA and statistical testing, engineers leakage-safe features, trains and tunes three classifiers, chooses a business-cost threshold, and exports Power BI-ready artifacts.

## Reproduce

```powershell
python -m pip install -r requirements.txt
python run_project.py
```

The run will regenerate all four source CSV files and write these deliverables:

- `data/customers.csv`
- `data/subscriptions.csv`
- `data/billing.csv`
- `data/support_tickets.csv`
- `outputs/clean_customer_profiles.csv`
- `outputs/final_scored_predictions.csv`
- `outputs/customer_churn.db`
- `outputs/model_metrics.csv`
- `outputs/confusion_matrices.csv`
- `outputs/feature_importance.csv`
- `outputs/threshold_recommendation.csv`
- `outputs/statistical_results.csv`
- `outputs/descriptive_statistics.csv`
- `outputs/outlier_limits.csv`
- `outputs/eda_churn_drivers.png`
- `outputs/correlation_heatmap.png`

## Cleaning decisions

1. Customer IDs are treated as keys. Duplicate profile, subscription, and billing rows are removed using `drop_duplicates(..., keep="first")`.
2. Numeric columns are median-imputed because skewed/outlier-heavy distributions make means unstable.
3. Region, plan, contract, and payment method missing values are labeled `Unknown`.
4. Outliers are detected using Tukey IQR limits. Outlier flags are retained for auditability and outlier values are winsorized for model features only.
5. `churn` is retained as a label only and excluded from every feature set.
6. Signup date is used to derive tenure at the snapshot date during generation; date fields and direct churn probability fields are not model features.

## Feature engineering

The pipeline creates tenure bands, month-to-month/two-year indicators, high delay and payment-delay counts, negative usage trend, support frequency and high-contact flags, tenure in years, and a low-usage flag. These features are derived only from pre-churn customer attributes and intentionally exclude `churn`, observed churn probability, and signup-date-derived target behavior.

## Validation and threshold

Data is split 60/40 into a training set and remaining data. The remaining data is split equally into validation and test sets using stratified sampling. Hyperparameters are selected with validation F1. The test set is used once for final reporting. The threshold is selected on validation predictions by minimizing:

`₹500 * false_negatives + ₹50 * false_positives`

All cost assumptions and exports use Indian rupees (₹). The default values are ₹500 for a missed churner and ₹50 for an unnecessary offer. They are replaceable: update the missed-churn cost and offer cost in `churn_pipeline.py` if the business values differ. The final score uses the whole customer feature set and the validation-derived threshold.

## Statistics

The pipeline reports Chi-square tests for churn versus contract, plan, and payment method, and Mann-Whitney U tests for churned versus retained customer distributions of tenure, usage, missed payments, and support tickets.

## SQL and Power BI

- Schema: `sql/churn_schema.sql`
- DAX: `powerbi/dax_measures.dax`
- Dashboard architecture: `powerbi/dashboard_spec.md`
