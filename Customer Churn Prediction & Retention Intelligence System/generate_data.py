"""Generate the synthetic Project 01 churn datasets."""
import os
import numpy as np
import pandas as pd

SEED = 2026
N_CUSTOMERS = 5_000


def generate_data(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    customer_ids = np.arange(10001, 10001 + N_CUSTOMERS)
    signup_offsets = rng.integers(30, 1826, N_CUSTOMERS)
    snapshot = pd.Timestamp("2026-11-22")
    signup_dates = snapshot - pd.to_timedelta(signup_offsets, unit="D")
    age = np.clip(rng.normal(43, 15, N_CUSTOMERS).round().astype(int), 18, 85)
    region = rng.choice(["North", "South", "East", "West", "Central"], N_CUSTOMERS, p=[0.22, 0.21, 0.20, 0.22, 0.15])

    plans = rng.choice(["Basic", "Standard", "Premium"], N_CUSTOMERS, p=[0.38, 0.39, 0.23])
    contracts = rng.choice(["Month-to-Month", "One Year", "Two Year"], N_CUSTOMERS, p=[0.55, 0.30, 0.15])
    payment_methods = rng.choice(["Credit Card", "Debit Card", "Bank Transfer", "Electronic Check"], N_CUSTOMERS,
                                 p=[0.36, 0.24, 0.22, 0.18])

    tenure = np.clip((snapshot - signup_dates).days // 30, 1, 60)
    plan_fee = {"Basic": 29, "Standard": 59, "Premium": 99}
    monthly_fee = np.array([plan_fee[p] for p in plans], dtype=float)
    monthly_fee = monthly_fee + rng.normal(0, 3.5, N_CUSTOMERS)
    avg_usage = rng.gamma(2.1, 22, N_CUSTOMERS)
    usage_trend = rng.normal(0, 0.12, N_CUSTOMERS)

    missed_payments = np.where(rng.random(N_CUSTOMERS) < 0.21, rng.integers(1, 5, N_CUSTOMERS), 0)
    missed_payments = np.where(rng.random(N_CUSTOMERS) < 0.08, missed_payments + rng.integers(4, 9, N_CUSTOMERS), missed_payments)
    delay = np.clip(rng.gamma(1.5, 4.5, N_CUSTOMERS).astype(int) + missed_payments * 3, 0, 65)
    billing_issue = np.where(rng.random(N_CUSTOMERS) < 0.14, rng.integers(1, 4, N_CUSTOMERS), 0)

    support_tickets = np.where(rng.random(N_CUSTOMERS) < 0.28, rng.integers(1, 5, N_CUSTOMERS), 0)
    support_tickets = np.where(rng.random(N_CUSTOMERS) < 0.06, support_tickets + rng.integers(5, 11, N_CUSTOMERS), support_tickets)
    open_tickets = np.minimum(support_tickets, rng.integers(0, 4, N_CUSTOMERS))
    resolution_hours = np.clip(rng.gamma(2.2, 10, N_CUSTOMERS) + open_tickets * 8, 1, 240)
    satisfaction = np.clip(rng.normal(3.7, 0.9, N_CUSTOMERS) - open_tickets * 0.25, 1, 5).round(1)

    # Realistic churn score: contract, payment delay, support, usage, tenure, and plan effects.
    z = (
        -1.25
        + (contracts == "Month-to-Month") * 0.95
        + (contracts == "One Year") * 0.25
        + (payment_methods == "Electronic Check") * 0.38
        + np.maximum(delay - 5, 0) * 0.035
        + missed_payments * 0.18
        + support_tickets * 0.12
        + open_tickets * 0.16
        + (avg_usage < 25) * 0.55
        + (tenure < 12) * 0.48
        + (plans == "Premium") * 0.10
        + usage_trend * -1.0
        + rng.normal(0, 0.45, N_CUSTOMERS)
    )
    churn_probability = 1 / (1 + np.exp(-z))
    churn = rng.binomial(1, churn_probability)

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": signup_dates.strftime("%Y-%m-%d"),
        "age": age,
        "region": region,
        "churn": churn,
    })

    subscriptions = pd.DataFrame({
        "customer_id": customer_ids,
        "subscription_plan": plans,
        "contract_type": contracts,
        "monthly_fee": monthly_fee.round(2),
        "tenure_months": tenure,
        "avg_monthly_usage_gb": avg_usage.round(2),
        "usage_trend_6m": usage_trend.round(3),
        "payment_method": payment_methods,
        "add_on_count": rng.integers(0, 5, N_CUSTOMERS),
    })

    billing = pd.DataFrame({
        "customer_id": customer_ids,
        "last_payment_delay_days": delay,
        "missed_payments_12m": missed_payments,
        "total_late_fees": (missed_payments * rng.uniform(8, 32, N_CUSTOMERS)).round(2),
        "billing_issue_count": billing_issue,
    })

    support = pd.DataFrame({
        "customer_id": customer_ids,
        "support_tickets_12m": support_tickets,
        "open_tickets": open_tickets,
        "avg_resolution_hours": resolution_hours.round(1),
        "satisfaction_score": satisfaction,
    })

    # Intentional quality issues
    for frame, columns, max_missing in [
        (customers, ["age", "region"], 0.018),
        (subscriptions, ["monthly_fee", "avg_monthly_usage_gb", "payment_method"], 0.025),
        (billing, ["last_payment_delay_days", "missed_payments_12m", "billing_issue_count"], 0.02),
        (support, ["satisfaction_score", "avg_resolution_hours"], 0.03),
    ]:
        for column in columns:
            idx = rng.choice(frame.index, size=int(len(frame) * max_missing), replace=False)
            frame.loc[idx, column] = np.nan

    # Duplicate customer rows and unrealistic outliers
    subscriptions = pd.concat([subscriptions, subscriptions.iloc[rng.choice(len(subscriptions), 45, replace=False)].copy()],
                              ignore_index=True)
    billing = pd.concat([billing, billing.iloc[rng.choice(len(billing), 35, replace=False)].copy()], ignore_index=True)
    support.loc[rng.choice(len(support), 25, replace=False), "support_tickets_12m"] = rng.integers(35, 61, 25)
    subscriptions.loc[rng.choice(len(subscriptions), 20, replace=False), "avg_monthly_usage_gb"] = rng.integers(700, 1201, 20)
    billing.loc[rng.choice(len(billing), 20, replace=False), "last_payment_delay_days"] = rng.integers(120, 181, 20)

    customers.to_csv(os.path.join(output_dir, "customers.csv"), index=False)
    subscriptions.to_csv(os.path.join(output_dir, "subscriptions.csv"), index=False)
    billing.to_csv(os.path.join(output_dir, "billing.csv"), index=False)
    support.to_csv(os.path.join(output_dir, "support_tickets.csv"), index=False)


if __name__ == "__main__":
    generate_data()
