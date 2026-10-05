"""
CatPriceAI - Loss Data Analysis

This script builds loss history, applies the treaty layer, computes burning cost,
creates a development triangle, and saves outputs for downstream pricing.
"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DB_PATH = DATA_DIR / "catprice.db"

ATTACHMENT = 50_000_000
LIMIT = 50_000_000
EXHAUSTION = ATTACHMENT + LIMIT


def load_data():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run python data/01_generate_data.py first.")
    with sqlite3.connect(DB_PATH) as conn:
        losses = pd.read_sql("SELECT * FROM loss_events", conn)
        treaty = pd.read_sql("SELECT * FROM treaty_terms", conn)
    return losses, treaty


def annual_gross_loss(losses: pd.DataFrame) -> pd.DataFrame:
    annual = (
        losses.groupby("year")["gross_loss_usd"]
        .sum()
        .reset_index()
        .rename(columns={"gross_loss_usd": "annual_gross_loss"})
    )
    all_years = pd.DataFrame({"year": range(int(losses["year"].min()), int(losses["year"].max()) + 1)})
    annual = all_years.merge(annual, on="year", how="left").fillna(0)
    annual["annual_gross_loss_m"] = (annual["annual_gross_loss"] / 1e6).round(2)
    return annual


def apply_cat_xl_layer(annual: pd.DataFrame) -> pd.DataFrame:
    df = annual.copy()
    df["loss_entering_layer"] = np.maximum(df["annual_gross_loss"] - ATTACHMENT, 0)
    df["loss_in_layer"] = np.minimum(df["loss_entering_layer"], LIMIT)
    df["layer_penetrated"] = df["loss_in_layer"] > 0
    df["exhausted_layer"] = df["annual_gross_loss"] >= EXHAUSTION
    df["loss_above_layer"] = np.maximum(df["annual_gross_loss"] - EXHAUSTION, 0)
    df["loss_in_layer_m"] = (df["loss_in_layer"] / 1e6).round(2)
    return df


def compute_burning_cost(layer_losses: pd.DataFrame) -> dict:
    years = len(layer_losses)
    total_loss_in_layer = layer_losses["loss_in_layer"].sum()
    aal = total_loss_in_layer / years
    burning_cost_rol = aal / LIMIT
    penetration_count = int(layer_losses["layer_penetrated"].sum())
    exhaustion_count = int(layer_losses["exhausted_layer"].sum())
    empirical_freq = penetration_count / years
    empirical_severity = (total_loss_in_layer / penetration_count) if penetration_count > 0 else 0
    return {
        "experience_years": years,
        "total_loss_in_layer_m": round(total_loss_in_layer / 1e6, 2),
        "aal_m": round(aal / 1e6, 2),
        "layer_limit_m": round(LIMIT / 1e6, 2),
        "burning_cost_rol": round(burning_cost_rol, 4),
        "burning_cost_rol_pct": round(burning_cost_rol * 100, 2),
        "layer_penetration_years": penetration_count,
        "layer_exhaustion_years": exhaustion_count,
        "empirical_freq_pct": round(empirical_freq * 100, 2),
        "empirical_severity_m": round(empirical_severity / 1e6, 2),
    }


def top_loss_events(losses: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return (
        losses.groupby(["year", "event_name", "peril"])["gross_loss_usd"]
        .sum()
        .reset_index()
        .sort_values("gross_loss_usd", ascending=False)
        .head(n)
        .assign(gross_loss_m=lambda x: (x["gross_loss_usd"] / 1e6).round(2))
    )


def simple_development_triangle(losses: pd.DataFrame) -> pd.DataFrame:
    annual = annual_gross_loss(losses)
    loss_years = annual[annual["annual_gross_loss"] > 0].copy()
    np.random.seed(7)
    rows = []
    for _, row in loss_years.iterrows():
        ultimate = row["annual_gross_loss"]
        dev_12 = ultimate * np.random.uniform(0.70, 0.85)
        dev_24 = ultimate * np.random.uniform(0.90, 0.98)
        dev_36 = ultimate * np.random.uniform(0.98, 1.00)
        rows.append({
            "loss_year": int(row["year"]),
            "12_months": round(dev_12, 0),
            "24_months": round(dev_24, 0),
            "36_months": round(dev_36, 0),
            "ultimate": round(ultimate, 0),
            "dev_to_ult_factor": round(ultimate / dev_12, 3) if dev_12 > 0 else None,
        })
    return pd.DataFrame(rows).sort_values("loss_year")


def main() -> None:
    losses, treaty = load_data()
    annual = annual_gross_loss(losses)
    layer_losses = apply_cat_xl_layer(annual)
    burning_cost = compute_burning_cost(layer_losses)
    triangle = simple_development_triangle(losses)
    annual.to_csv(DATA_DIR / "loss_summary.csv", index=False)
    layer_losses[layer_losses["layer_penetrated"]].to_csv(DATA_DIR / "loss_in_layer.csv", index=False)
    pd.DataFrame([burning_cost]).to_csv(DATA_DIR / "burning_cost.csv", index=False)
    print("Loss analysis complete.")
    print(layer_losses.to_string(index=False))
    print("\nBurning cost")
    print(burning_cost)
    print("\nTop losses")
    print(top_loss_events(losses).to_string(index=False))
    print("\nDevelopment triangle")
    print(triangle.to_string(index=False))


if __name__ == "__main__":
    main()
