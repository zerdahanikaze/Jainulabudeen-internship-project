"""Generate a reproducible, de-identified synthetic healthcare dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20250308
N_ENCOUNTERS = 12_000
AGE_GROUPS = ["18-34", "35-49", "50-64", "65-79", "80+"]
DEPARTMENTS = ["CARD", "EMER", "MED", "NEURO", "ORTHO", "SURG"]


def _calibrate_intercept(linear_predictor: np.ndarray, target_rate: float) -> float:
    """Find a logistic intercept whose mean predicted probability matches target_rate."""
    low, high = -20.0, 20.0
    for _ in range(100):
        middle = (low + high) / 2
        probabilities = 1 / (1 + np.exp(-(linear_predictor + middle)))
        if probabilities.mean() < target_rate:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def generate_data(output_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create patient encounters and a non-identifying department/facility lookup."""
    rng = np.random.default_rng(SEED)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    age_group = rng.choice(AGE_GROUPS, size=N_ENCOUNTERS)
    age_midpoint = pd.Series(age_group).map(
        {"18-34": 26, "35-49": 42, "50-64": 57, "65-79": 72, "80+": 85}
    ).to_numpy()
    sex = rng.choice(["Female", "Male"], size=N_ENCOUNTERS)
    department_id = rng.choice(DEPARTMENTS, size=N_ENCOUNTERS)
    admission_type = rng.choice(
        ["Emergency", "Elective", "Urgent"],
        size=N_ENCOUNTERS,
        p=[0.48, 0.28, 0.24],
    )

    age_years = np.clip(rng.normal(age_midpoint, 4.0), 18, 95)
    comorbidity_count = np.clip(
        rng.poisson(np.maximum((age_years - 25) / 23, 0.25)), 0, 10
    )
    prior_admissions = np.clip(
        rng.poisson(0.25 + 0.09 * comorbidity_count + (admission_type == "Emergency") * 0.2),
        0,
        8,
    )
    systolic_bp = rng.normal(121 + 0.16 * (age_years - 50), 15)
    heart_rate = rng.normal(78 + 3.0 * (admission_type == "Emergency"), 12)
    glucose = rng.lognormal(np.log(104 + 5.0 * comorbidity_count), 0.24)
    bmi = np.clip(rng.normal(27.5, 5.2, N_ENCOUNTERS), 16, 48)
    length_of_stay = np.clip(
        rng.gamma(1.7 + 0.12 * comorbidity_count + 0.25 * (admission_type != "Elective"), 1.4),
        0.1,
        35,
    )
    lab_test_ordered = rng.choice(["Yes", "No"], size=N_ENCOUNTERS, p=[0.72, 0.28])

    # Outcome risk is generated from clinically plausible associations, not causal effects.
    linear_predictor = (
        0.035 * (age_years - 55)
        + 0.29 * comorbidity_count
        + 0.34 * prior_admissions
        + 0.48 * (admission_type == "Emergency")
        + 0.22 * (admission_type == "Urgent")
        + 0.62 * (systolic_bp < 90)
        + 0.28 * (systolic_bp > 180)
        + 0.33 * (heart_rate < 50)
        + 0.30 * (heart_rate > 110)
        + 0.55 * (glucose < 60)
        + 0.38 * (glucose > 200)
        + 0.012 * (length_of_stay - 2)
    )
    intercept = _calibrate_intercept(linear_predictor, 0.18)
    outcome_probability = 1 / (1 + np.exp(-(linear_predictor + intercept)))
    adverse_outcome = rng.binomial(1, outcome_probability)

    # IDs are salted hashes of synthetic row counters; 1% of rows intentionally collide.
    identifiers = [
        hashlib.sha256(f"synthetic-{SEED}-{index}".encode("utf-8")).hexdigest()[:20]
        for index in range(N_ENCOUNTERS)
    ]
    duplicate_count = round(N_ENCOUNTERS * 0.01)
    duplicate_rows = rng.choice(np.arange(1, N_ENCOUNTERS), duplicate_count, replace=False)
    duplicate_sources = rng.integers(0, N_ENCOUNTERS, size=duplicate_count)
    for row, source in zip(duplicate_rows, duplicate_sources):
        identifiers[row] = identifiers[source]

    records = pd.DataFrame(
        {
            "encounter_id": identifiers,
            "age_group": age_group,
            "sex": sex,
            "department_id": department_id,
            "admission_type": admission_type,
            "length_of_stay_days": np.round(length_of_stay, 2),
            "systolic_bp": np.round(systolic_bp, 1),
            "heart_rate": np.round(heart_rate, 1),
            "glucose_mg_dl": np.round(glucose, 1),
            "bmi": np.round(bmi, 1),
            "comorbidity_count": comorbidity_count.astype(int),
            "prior_admissions_12m": prior_admissions.astype(int),
            "lab_test_ordered": lab_test_ordered,
            "adverse_outcome_30d": adverse_outcome.astype(int),
        }
    )

    # Missingness and impossible-value injections are explicit and reproducible.
    missing_rows = rng.choice(N_ENCOUNTERS, round(N_ENCOUNTERS * 0.04), replace=False)
    records.loc[missing_rows, "glucose_mg_dl"] = np.nan
    for column, count in {
        "systolic_bp": 24,
        "heart_rate": 24,
        "glucose_mg_dl": 24,
        "bmi": 24,
    }.items():
        rows = rng.choice(N_ENCOUNTERS, count, replace=False)
        invalid_value = {"systolic_bp": 999, "heart_rate": -5, "glucose_mg_dl": 2500, "bmi": -4}[column]
        records.loc[rows, column] = invalid_value

    lookup = pd.DataFrame(
        {
            "department_id": DEPARTMENTS,
            "department_name": [
                "Cardiology",
                "Emergency Medicine",
                "General Medicine",
                "Neurology",
                "Orthopedics",
                "Surgery",
            ],
            "facility_id": ["FAC-01", "FAC-01", "FAC-02", "FAC-02", "FAC-03", "FAC-03"],
            "facility_name": ["North Medical Center", "North Medical Center", "Central Hospital",
                              "Central Hospital", "West Clinic", "West Clinic"],
        }
    )
    records.to_csv(output_dir / "patient_records.csv", index=False)
    lookup.to_csv(output_dir / "facility_lookup.csv", index=False)
    return records, lookup


if __name__ == "__main__":
    generated, _ = generate_data()
    print(
        f"Wrote {len(generated):,} synthetic encounters to "
        f"{Path(__file__).resolve().parent / 'data' / 'raw'} "
        f"(observed outcome rate {generated['adverse_outcome_30d'].mean():.1%})."
    )
