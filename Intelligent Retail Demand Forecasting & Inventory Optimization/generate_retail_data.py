"""Generate reproducible synthetic retail demand and inventory datasets."""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
END_DATE = pd.Timestamp("2026-11-22")
START_DATE = END_DATE - pd.Timedelta(days=729)
N_STORES = 10
N_PRODUCTS = 50


def generate_retail_data(output_dir: str | Path = "data/raw") -> None:
    """Create sales, product, store, and inventory CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    products = pd.DataFrame({
        "product_id": [f"P{i:03d}" for i in range(1, N_PRODUCTS + 1)],
        "category": np.resize(["Grocery", "Beverages", "Household", "Personal Care", "Electronics"], N_PRODUCTS),
    })
    subcategories = {
        "Grocery": ["Pantry", "Snacks", "Dairy", "Frozen"],
        "Beverages": ["Water", "Soft Drinks", "Coffee", "Juice"],
        "Household": ["Cleaning", "Paper Goods", "Laundry"],
        "Personal Care": ["Hair Care", "Skin Care", "Oral Care"],
        "Electronics": ["Accessories", "Small Appliances", "Batteries"],
    }
    products["subcategory"] = [
        rng.choice(subcategories[category]) for category in products["category"]
    ]
    products["base_price"] = np.round(rng.uniform(3.0, 120.0, N_PRODUCTS), 2)
    products["unit_cost"] = np.round(products["base_price"] * rng.uniform(0.42, 0.78, N_PRODUCTS), 2)
    products["lead_time_days"] = rng.integers(2, 15, N_PRODUCTS)

    stores = pd.DataFrame({
        "store_id": [f"S{i:02d}" for i in range(1, N_STORES + 1)],
        "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                 "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"],
        "store_size": ["Large", "Large", "Large", "Medium", "Medium",
                       "Medium", "Small", "Small", "Medium", "Small"],
        "region": ["Northeast", "West", "Midwest", "South", "West",
                   "Northeast", "South", "West", "South", "West"],
    })

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    product_demand = rng.uniform(2.2, 13.0, N_PRODUCTS)
    store_multipliers = rng.uniform(0.7, 1.4, N_STORES)
    elasticity = rng.uniform(0.65, 1.45, N_PRODUCTS)
    base_prices = products["base_price"].to_numpy()
    promo_flags = rng.random((len(dates), N_STORES, N_PRODUCTS)) < 0.09
    rows = []

    holidays = {
        (1, 1): 1.55, (2, 14): 1.20, (7, 4): 1.30, (10, 31): 1.35,
        (11, 27): 1.85, (12, 24): 1.75, (12, 25): 1.65, (12, 31): 1.45,
    }
    for day_index, date in enumerate(dates):
        dow = date.dayofweek
        weekend_multiplier = 1.48 if dow >= 5 else 1.0
        yearly_multiplier = 1.0 + 0.20 * np.sin(
            2 * np.pi * (date.dayofyear - 1) / 365.25
        )
        holiday_multiplier = holidays.get((date.month, date.day), 1.0)
        for store_index, store in stores.iterrows():
            store_size_multiplier = {"Small": 0.8, "Medium": 1.0, "Large": 1.25}[
                store["store_size"]
            ]
            for product_index, product in products.iterrows():
                promo = int(promo_flags[day_index, store_index, product_index])
                price = base_prices[product_index] * (0.82 if promo else 1.0)
                price_effect = (price / base_prices[product_index]) ** (-elasticity[product_index])
                category_effect = 1.25 if product["category"] in ("Grocery", "Beverages") else 1.0
                mean_demand = (
                    product_demand[product_index] * store_multipliers[store_index]
                    * store_size_multiplier * weekend_multiplier * yearly_multiplier
                    * holiday_multiplier * category_effect * price_effect
                    * (1.35 if promo else 1.0)
                )
                units = int(rng.poisson(max(mean_demand, 0.1)))

                # Randomly omit some store/SKU/day observations to create genuine calendar gaps.
                if rng.random() < 0.0025:
                    continue
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "store_id": store["store_id"],
                    "product_id": product["product_id"],
                    "units_sold": units,
                    "unit_price": round(price, 2),
                    "promo_flag": promo,
                })

    sales = pd.DataFrame(rows)
    # One percent extra copies of existing transaction rows (including exact row duplicates).
    duplicate_count = max(1, int(round(len(sales) * 0.01)))
    duplicates = sales.iloc[rng.choice(len(sales), duplicate_count, replace=False)].copy()
    sales = pd.concat([sales, duplicates], ignore_index=True)

    # Mix ISO and US-style dates to require robust date normalization downstream.
    date_values = pd.to_datetime(sales["date"])
    us_style = rng.random(len(sales)) < 0.18
    sales.loc[us_style, "date"] = date_values[us_style].dt.strftime("%m/%d/%Y")
    sales.to_csv(output_path / "sales_history.csv", index=False)
    products.to_csv(output_path / "products.csv", index=False)
    stores.to_csv(output_path / "stores.csv", index=False)

    inventory = []
    latest = sales.copy()
    latest["date"] = pd.to_datetime(latest["date"], format="mixed")
    recent = latest[latest["date"] >= END_DATE - pd.Timedelta(days=27)]
    recent_demand = recent.groupby(["store_id", "product_id"])["units_sold"].mean()
    for store in stores["store_id"]:
        for product_index, product in products.iterrows():
            daily = float(recent_demand.get((store, product["product_id"]), 0.0))
            lead_time = int(product["lead_time_days"])
            reorder_point = max(1, int(np.ceil(daily * (lead_time + 3))))
            stock = max(0, int(np.ceil(daily * rng.integers(1, lead_time + 7))))
            inventory.append({
                "store_id": store,
                "product_id": product["product_id"],
                "current_stock_on_hand": stock,
                "reorder_point": reorder_point,
            })
    pd.DataFrame(inventory).to_csv(output_path / "current_inventory.csv", index=False)


if __name__ == "__main__":
    generate_retail_data()
