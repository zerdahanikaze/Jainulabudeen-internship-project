"""Generate reproducible synthetic payment transactions and account data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_data(output_dir: Path, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create 50,000 transactions across 3,000 accounts with labeled fraud signals."""
    rng = np.random.default_rng(seed)
    account_count = 3_000
    transaction_count = 50_000

    account_ids = np.array([f"A{i:05d}" for i in range(account_count)])
    home_countries = rng.choice(
        ["US", "GB", "CA", "AU", "DE", "FR", "IN", "SG"],
        size=account_count,
        p=[0.48, 0.12, 0.10, 0.07, 0.07, 0.05, 0.08, 0.03],
    )
    accounts = pd.DataFrame(
        {
            "account_id": account_ids,
            "account_age_days": np.maximum(
                1, rng.lognormal(mean=5.1, sigma=1.0, size=account_count).astype(int)
            ),
            "kyc_tier": rng.choice(
                ["basic", "standard", "enhanced"], account_count, p=[0.25, 0.60, 0.15]
            ),
            "avg_monthly_spend": np.round(
                rng.lognormal(mean=6.9, sigma=0.85, size=account_count), 2
            ),
            "home_country": home_countries,
        }
    )
    account_lookup = accounts.set_index("account_id")

    tx_accounts = rng.choice(account_ids, size=transaction_count)
    timestamps = (
        pd.Timestamp("2025-01-01", tz="UTC")
        + pd.to_timedelta(rng.integers(0, 90 * 24 * 60, size=transaction_count), unit="m")
    ).to_numpy()
    amount = np.clip(rng.lognormal(mean=3.45, sigma=1.15, size=transaction_count), 1, 25_000)
    fraud = np.zeros(transaction_count, dtype=np.int8)

    # Assign a stable 1.5% base rate, then give those transactions observable risk signals.
    fraud_indices = rng.choice(transaction_count, size=round(transaction_count * 0.015), replace=False)
    fraud[fraud_indices] = 1
    late_hours = rng.choice([1, 2, 3], size=len(fraud_indices), p=[0.4, 0.35, 0.25])
    fraud_times = pd.DatetimeIndex(timestamps[fraud_indices])
    fraud_times = fraud_times.normalize() + pd.to_timedelta(late_hours, unit="h")
    timestamps[fraud_indices] = fraud_times.to_numpy()

    fraud_accounts = tx_accounts[fraud_indices]
    account_spend = account_lookup.loc[fraud_accounts, "avg_monthly_spend"].to_numpy()
    amount[fraud_indices] = np.maximum(
        amount[fraud_indices],
        account_spend * rng.uniform(1.5, 5.0, size=len(fraud_indices)),
    )

    categories = np.array(
        ["grocery", "restaurants", "retail", "travel", "digital_goods", "fuel", "utilities"]
    )
    category = rng.choice(categories, size=transaction_count, p=[0.20, 0.18, 0.22, 0.08, 0.12, 0.08, 0.12])
    category[fraud_indices] = rng.choice(
        ["digital_goods", "travel", "retail"], size=len(fraud_indices), p=[0.55, 0.25, 0.20]
    )

    device = rng.choice(["mobile", "web", "pos"], transaction_count, p=[0.57, 0.25, 0.18]).astype(object)
    device[fraud_indices] = rng.choice(
        ["mobile", "web"], size=len(fraud_indices), p=[0.6, 0.4]
    )
    # Roughly one in five frauds is foreign and uses a newly observed device proxy.
    foreign_fraud_indices = fraud_indices[: max(1, len(fraud_indices) // 5)]
    device[foreign_fraud_indices] = "new_device"

    home_country_by_tx = account_lookup.loc[tx_accounts, "home_country"].to_numpy()
    country = home_country_by_tx.copy()
    foreign = rng.random(transaction_count) < 0.035
    foreign[fraud_indices] = rng.random(len(fraud_indices)) < 0.72
    country_codes = np.array(["US", "GB", "CA", "AU", "DE", "FR", "IN", "SG"])
    for index in np.flatnonzero(foreign):
        alternatives = country_codes[country_codes != home_country_by_tx[index]]
        country[index] = rng.choice(alternatives)

    frame = pd.DataFrame(
        {
            "transaction_id": [f"T{i:07d}" for i in range(transaction_count)],
            "account_id": tx_accounts,
            "timestamp": timestamps,
            "amount": np.round(amount, 2),
            "merchant_category": category,
            "device_type": device,
            "country_code": country,
            "is_foreign_tx": foreign.astype(np.int8),
            "is_fraud": fraud,
        }
    )

    # Create short transaction bursts around some fraud events without changing labels.
    burst_count = min(len(fraud_indices), 350)
    burst_fraud_indices = fraud_indices[:burst_count]
    burst_rows: list[dict[str, object]] = []
    for idx in burst_fraud_indices:
        for offset in (2, 5):
            burst_rows.append(
                {
                    "transaction_id": f"T{transaction_count + len(burst_rows):07d}",
                    "account_id": frame.at[idx, "account_id"],
                    "timestamp": frame.at[idx, "timestamp"] + pd.Timedelta(minutes=offset),
                    "amount": round(float(rng.uniform(8, 75)), 2),
                    "merchant_category": "digital_goods",
                    "device_type": frame.at[idx, "device_type"],
                    "country_code": frame.at[idx, "country_code"],
                    "is_foreign_tx": frame.at[idx, "is_foreign_tx"],
                    "is_fraud": 0,
                }
            )
    if burst_rows:
        # Replace an equal number of non-fraud rows so the requested row count and prevalence hold.
        replaceable = np.flatnonzero(frame["is_fraud"].to_numpy() == 0)
        replace_indices = rng.choice(replaceable, size=len(burst_rows), replace=False)
        frame = frame.drop(index=replace_indices).reset_index(drop=True)
        frame = pd.concat([frame, pd.DataFrame(burst_rows)], ignore_index=True)

    # Duplicate 0.5% of transaction IDs while retaining the same total row count.
    duplicate_count = round(transaction_count * 0.005)
    duplicate_rows = rng.choice(frame.index.to_numpy(), size=duplicate_count, replace=False)
    duplicate_sources = rng.choice(frame.index.to_numpy(), size=duplicate_count, replace=False)
    frame.loc[duplicate_rows, "transaction_id"] = frame.loc[
        duplicate_sources, "transaction_id"
    ].to_numpy()

    missing_count = round(transaction_count * 0.02)
    missing_rows = rng.choice(frame.index.to_numpy(), size=missing_count, replace=False)
    frame.loc[missing_rows, "device_type"] = np.nan

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["timestamp", "account_id"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "transactions.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    accounts.to_csv(output_dir / "accounts.csv", index=False)
    return frame, accounts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "raw",
        help="Directory for transactions.csv and accounts.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    transactions, accounts = generate_data(args.output_dir, seed=args.seed)
    print(f"Wrote {len(transactions):,} transactions and {len(accounts):,} accounts to {args.output_dir}")
    print(f"Fraud rate: {transactions['is_fraud'].mean():.2%}")
    print(f"Duplicate transaction IDs: {transactions['transaction_id'].duplicated().sum():,}")
    print(f"Missing device types: {transactions['device_type'].isna().sum():,}")


if __name__ == "__main__":
    main()
