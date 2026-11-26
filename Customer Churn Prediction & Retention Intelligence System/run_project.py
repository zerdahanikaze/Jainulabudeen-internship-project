"""Execute the complete Customer Churn Intelligence project."""
from pathlib import Path

from generate_data import generate_data
from churn_pipeline import run_pipeline


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    generate_data(str(data_dir))
    run_pipeline(str(data_dir), str(base / "outputs"))
