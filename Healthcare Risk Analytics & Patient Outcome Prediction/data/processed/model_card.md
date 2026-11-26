# Model Card: Synthetic 30-Day Adverse Outcome Risk

## Summary
- **Model selected for reporting:** Logistic Regression, selected by mean 5-fold stratified training ROC-AUC.
- **Evaluation:** held-out stratified test set; threshold = 0.50.
- **Synthetic outcome prevalence:** 17.9% before deduplication.
- **Test ROC-AUC / recall / precision / F1:** 0.786 / 0.716 / 0.356 / 0.476.
- **Test calibration gap** (absolute difference in mean predicted and observed rate): 0.234.

## Intended use
Educational demonstration and aggregate analytics using wholly synthetic data. This model is a research prototype and decision-support illustration only; it is **not for diagnosis, triage, treatment selection, or clinical judgment**.

## Data and methods
Rows with duplicate de-identified encounter IDs are reduced to the first occurrence. Measurements outside documented plausible ranges are set to missing and retained with invalid-value flags. Numeric missing values are median-imputed and categorical values are most-frequent-imputed inside each training fold; missingness flags are retained. The model uses age band, department, admission type, laboratory-order status, clinical measurements, and quality flags. Sex is excluded from predictors and retained for auditing. Test performance is reported once after 5-fold stratified cross-validation on training data.

## Performance and calibration
See `model_evaluation_metrics.csv`, `calibration_curves.png`, and the confusion-matrix plots for the complete comparison. The selected model is chosen using training cross-validation only, not the held-out test results. A 0.50 threshold is illustrative and has not been optimized for a clinical utility function.

## Fairness checks
Held-out recall, precision, F1, ROC-AUC, predicted-positive rate, calibration gap, and a min-to-max predicted-positive-rate parity ratio are reported by age group and sex in `agg_subgroup_fairness.csv`. Small or low-event cells are suppressed (fewer than 30 records, or fewer than 10 positives or negatives). These are diagnostic summaries, not evidence of fairness; uncertainty intervals, clinical context, intersectional analysis, and stakeholder review are still required.

## Limitations and responsible-use constraints
- Data are synthetic and do not establish clinical validity, transportability, or real-world utility.
- Associations and model explanations are correlational, not causal; odds ratios and feature importances can be distorted by encoding, collinearity, missingness, and the synthetic data-generating process.
- The target definition, event ascertainment, measurement processes, and subgroup composition are simplified. Fairness metrics depend on outcome quality and threshold choice.
- This prototype has no external or temporal validation, prospective monitoring, clinician workflow assessment, or benefit-harm analysis. It must not be deployed on real patients.
- No patient-level predictions are exported. Only aggregated metrics and model summaries are written to `data/processed/`.
