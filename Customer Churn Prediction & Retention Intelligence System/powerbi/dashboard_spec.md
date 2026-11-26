# Power BI Dashboard Specification

## Model

- Fact table: `churn_scores` (customer_id, churn_probability, predicted_churn, retention_risk_segment)
- Dimension table: `clean_customer_profiles` (customer_id and customer attributes)
- Create a one-to-many relationship from `clean_customer_profiles[customer_id]` to `churn_scores[customer_id]`.
- Import `customer_churn.db` by first exporting it to an Excel/CSV source, or import the cleaned CSV and scored CSV through the Power BI folder connector.
- Mark `signup_date` as Date and sort any monthly/tenure labels explicitly.

## Page 1 - Executive Retention Overview

KPI cards:
- Total Customers
- Churn Rate %
- High-Risk Customer Count
- Average Churn Probability
- Retained Customers Estimate
- Monthly Retention Cost

Visuals:
- Clustered column chart: churn rate by tenure band
- Stacked column chart: churn by contract type and subscription plan
- Donut chart: customer count by retention risk segment
- Horizontal bar chart: top risk drivers based on model feature importance
- Line chart: monthly or tenure trend for churn probability
- Slicer: region, subscription plan, contract type, and score segment

## Page 2 - Customer Risk & Action Queue

KPI cards:
- Critical segment count
- High segment count
- High-risk revenue exposure
- Average support contacts

Visuals:
- Decomposition tree for churn probability and risk segments
- Table: customer_id, probability, segment, contract, payment method, monthly_fee, missed payments, support tickets
- Conditional formatting: red at probability >= 0.75, amber >= 0.50
- Scatter chart: usage vs. tenure, size=probability, legend=risk segment
- Slicer: risk segment and service usage band

## Page 3 - Drivers and Root-Cause Analysis

Visuals:
- Tornado/bar chart: support tickets, payment delays, month-to-month, low usage, and tenure feature effects
- 100% stacked column: churn by payment method
- 100% stacked column: churn by subscription plan
- Box plot: usage, delays, and satisfaction for retained vs. churned customers
- Chi-square/Mann-Whitney result table from `statistical_results.csv`

## Page 4 - Intervention Capacity What-If

Controls:
- Monthly Retention Capacity
- Monthly Retention Success Rate %
- Retention Offer Cost

KPI cards:
- Retained Customers Estimate
- Monthly Retention Cost
- Retention ROI Estimate
- Capacity-constrained high-risk count

Visuals:
- Line chart: retained customers against capacity
- Clustered bar: cost vs. retained customers by contract/segment
- Scenario table: 0, 100, 200, 300, 500 customers/month
- Tooltip: expected churn avoided and monthly cost
