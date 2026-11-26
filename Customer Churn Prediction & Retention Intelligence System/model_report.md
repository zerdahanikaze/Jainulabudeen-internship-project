# Customer Churn Model Comparison & Evaluation Report

## Executive summary

The project evaluated Logistic Regression, Decision Tree, and Random Forest classifiers on held-out test data. The Decision Tree was the weakest default-threshold model, while Random Forest produced the strongest F1 score and recall. The pipeline selected **Logistic Regression** as the model for business-cost threshold tuning because it had the strongest validation ROC-AUC among the candidates.

The chosen probability threshold is **0.18**. The threshold was optimized on validation data using:

```text
Estimated cost = 500 * false negatives + 50 * false positives
```

The reported validation minimum cost is **₹22,300**, with missed churners valued at **₹500** and unnecessary retention offers valued at **₹50**.

## Model comparison

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.500 | 0.6661 | 0.7215 | 0.6927 | 0.6860 | 0.6460 |
| Logistic Regression, tuned | 0.180 | 0.5530 | 1.0000 | 0.7122 | 0.6860 | 0.5530 |
| Decision Tree | 0.500 | 0.6150 | 0.6817 | 0.6467 | 0.6108 | 0.5880 |
| Random Forest | 0.500 | 0.6474 | 0.7505 | 0.6951 | 0.6849 | 0.6360 |

Source: `outputs/model_metrics.csv`.

### Default-threshold interpretation

- Random Forest has the highest default-threshold F1 (0.6951), the highest recall (0.7505), and nearly the same ROC-AUC as Logistic Regression.
- Logistic Regression has the highest precision (0.6661) at 0.50 and the strongest recorded ROC-AUC (0.6860).
- Decision Tree has the lowest F1 and ROC-AUC in this comparison.

## Tuned Logistic Regression threshold

The validation-derived threshold is 0.18:

| Metric | Test result |
|---|---:|
| Precision | 0.5530 |
| Recall | 1.0000 |
| F1 | 0.7122 |
| ROC-AUC | 0.6860 |
| Accuracy | 0.5530 |

At the tuned threshold, the test confusion matrix contains **447 false positives and no false negatives**, so recall reaches 1.0 at the expense of low precision and lower overall accuracy. This behavior matches a conservative retention strategy, but it should be operationalized carefully because it offers retention actions to every customer scored above the threshold.

## Top churn drivers

The Logistic Regression feature importance exported in `outputs/feature_importance.csv` ranks the following as the strongest top-level drivers:

| Rank | Driver | Relative importance |
|---:|---|---:|
| 1 | Premium subscription plan | 0.3241 |
| 2 | Last payment delay | 0.3045 |
| 3 | Electronic Check payment method | 0.2855 |
| 4 | 0-6 month tenure band | 0.2835 |
| 5 | Month-to-month contract | 0.2329 |
| 6 | 7-12 month tenure band | 0.2314 |
| 7 | High support contact | 0.2208 |
| 8 | Month-to-month contract indicator | 0.2049 |
| 9 | Low usage flag | 0.1804 |
| 10 | Total late fees | 0.1635 |

Related Random Forest and Decision Tree importance records identify payment delays, missed payments, usage, resolution time, support contact frequency, and month-to-month contracts as important across model types.

## Business cost justification

- Missed churner cost: **₹500**
- Unnecessary offer cost: **₹50**
- Validation minimum cost: **₹22,300**
- Why the threshold was selected: ₹500 missed revenue is ten times larger than a ₹50 unnecessary offer, so the optimization strongly favors false negatives avoidance over false-positive avoidance.

The 0.18 threshold is therefore appropriate when avoiding missed churners is the priority. Before production deployment, the business should validate whether ₹500 and ₹50 are realistic and adjust the threshold if retention capacity or offer economics change.

## Operational recommendation

1. Use Logistic Regression probabilities for the retention action queue and apply capacity-based segmentation.
2. Prioritize Critical and High segments, with special attention to short-tenure month-to-month customers, low-usage customers, late payers, electronic-check users, and high-support-contact customers.
3. Use the 0.18 threshold for broad catchment only if the ₹500 missed-churn / ₹50 offer cost ratio remains valid.
4. For a more precise intervention policy, rank customers by probability and apply a monthly capacity limit rather than treating every probability above the threshold identically.
