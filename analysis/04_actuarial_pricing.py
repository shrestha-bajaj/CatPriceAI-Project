"""
CatPriceAI - Actuarial Pricing Model

This script compares three pricing methods and produces the final technical ROL.
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BURNING_COST_PATH = DATA_DIR / "burning_cost.csv"
EXPOSURE_SUMMARY_PATH = DATA_DIR / "exposure_summary.csv"
LOSS_SUMMARY_PATH = DATA_DIR / "loss_summary.csv"
LOSS_IN_LAYER_PATH = DATA_DIR / "loss_in_layer.csv"
MARKET_ROL_PATH = DATA_DIR / "market_rol.csv"
PRICING_OUTPUT_PATH = DATA_DIR / "pricing_output.csv"
ML_PRICING_PATH = DATA_DIR / "ml_pricing.csv"
METHOD_COMPARISON_PATH = DATA_DIR / "method_comparison.csv"

ATTACHMENT = 50_000_000
LIMIT = 50_000_000
EXPENSE_LOAD = 0.05
PROFIT_TARGET = 0.08
UNCERTAINTY_LOAD = 0.10
BURNING_COST_WEIGHT = 0.40
FREQ_SEV_WEIGHT = 0.35
EXPOSURE_RATING_WEIGHT = 0.25


def check_required_files():
    required_files = [
        BURNING_COST_PATH,
        EXPOSURE_SUMMARY_PATH,
        LOSS_SUMMARY_PATH,
        LOSS_IN_LAYER_PATH,
        MARKET_ROL_PATH,
    ]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError("Missing required files: " + ", ".join(str(p) for p in missing_files))


def load_inputs():
    burning_cost = pd.read_csv(BURNING_COST_PATH)
    exposure_summary = pd.read_csv(EXPOSURE_SUMMARY_PATH)
    loss_summary = pd.read_csv(LOSS_SUMMARY_PATH)
    loss_in_layer = pd.read_csv(LOSS_IN_LAYER_PATH)
    market_rol = pd.read_csv(MARKET_ROL_PATH)
    return burning_cost, exposure_summary, loss_summary, loss_in_layer, market_rol


def method_1_burning_cost(burning_cost):
    pure_rol_pct = float(burning_cost.loc[0, "burning_cost_rol_pct"])
    return {
        "method": "Burning Cost",
        "description": "Historical average annual loss in the treaty layer.",
        "pure_rol_pct": pure_rol_pct,
    }


def method_2_frequency_severity(loss_in_layer):
    np.random.seed(2026)
    number_of_simulations = 100_000
    historical_layer_losses = loss_in_layer["loss_in_layer"]
    years = len(loss_in_layer)
    years_hitting_layer = int((historical_layer_losses > 0).sum())
    annual_hit_frequency = years_hitting_layer / years
    positive_layer_losses = historical_layer_losses[historical_layer_losses > 0]
    if len(positive_layer_losses) == 0:
        simulated_average_loss = 0.0
    else:
        event_counts = np.random.poisson(lam=annual_hit_frequency, size=number_of_simulations)
        sampled_severities = np.random.choice(positive_layer_losses, size=number_of_simulations, replace=True)
        simulated_annual_losses = event_counts * sampled_severities
        simulated_average_loss = simulated_annual_losses.mean()
    pure_rol_pct = simulated_average_loss / LIMIT * 100
    return {
        "method": "Frequency-Severity",
        "description": "Monte Carlo simulation of annual layer losses.",
        "pure_rol_pct": round(pure_rol_pct, 4),
        "annual_hit_frequency": round(annual_hit_frequency, 4),
        "simulation_years": number_of_simulations,
    }


def method_3_exposure_rating(exposure_summary):
    row = exposure_summary.iloc[0]
    total_tiv = float(row["total_tiv"])
    base_loss_cost_pct_of_tiv = 0.006
    coastal_factor = 1 + (float(row["coastal_tiv_pct"]) / 100 * 0.35)
    flood_factor = 1 + (float(row["high_risk_flood_tiv_pct"]) / 100 * 0.30)
    wood_frame_pct = float(row["wood_frame_tiv_pct"])
    wood_frame_factor = 1 + (wood_frame_pct / 100 * 0.18)
    state_concentration = bool(row["state_concentration_flag"])
    concentration_factor = 1.12 if state_concentration else 1.00
    expected_ground_up_loss = (
        total_tiv
        * base_loss_cost_pct_of_tiv
        * coastal_factor
        * flood_factor
        * wood_frame_factor
        * concentration_factor
    )
    layer_conversion_factor = 0.35
    expected_layer_loss = expected_ground_up_loss * layer_conversion_factor
    pure_rol_pct = expected_layer_loss / LIMIT * 100
    return {
        "method": "Exposure Rating",
        "description": "Top-down exposure-based expected loss estimate.",
        "pure_rol_pct": round(pure_rol_pct, 4),
        "expected_ground_up_loss": round(expected_ground_up_loss, 2),
        "expected_layer_loss": round(expected_layer_loss, 2),
    }


def create_method_comparison(method_results):
    method_comparison = pd.DataFrame(method_results)
    weights = {
        "Burning Cost": BURNING_COST_WEIGHT,
        "Frequency-Severity": FREQ_SEV_WEIGHT,
        "Exposure Rating": EXPOSURE_RATING_WEIGHT,
    }
    method_comparison["weight"] = method_comparison["method"].map(weights)
    method_comparison["weighted_pure_rol_pct"] = method_comparison["pure_rol_pct"] * method_comparison["weight"]
    return method_comparison


def calculate_final_price(method_comparison, market_rol):
    blended_pure_rol_pct = method_comparison["weighted_pure_rol_pct"].sum()
    total_load = EXPENSE_LOAD + PROFIT_TARGET + UNCERTAINTY_LOAD
    technical_rol_pct = blended_pure_rol_pct * (1 + total_load)
    pure_premium = LIMIT * blended_pure_rol_pct / 100
    technical_premium = LIMIT * technical_rol_pct / 100
    latest_market_year = int(market_rol["year"].max())
    latest_market_rol = float(market_rol.loc[market_rol["year"] == latest_market_year, "market_rol"].iloc[0])
    latest_market_rol_pct = latest_market_rol * 100
    adequacy_vs_market_pct = (technical_rol_pct - latest_market_rol_pct) / latest_market_rol_pct * 100
    recommendation = "QUOTE" if technical_rol_pct >= latest_market_rol_pct * 0.95 else "PASS"
    return pd.DataFrame([
        {
            "treaty": "Gulf Coast Insurance Co Cat XL",
            "attachment": ATTACHMENT,
            "limit": LIMIT,
            "blended_pure_rol_pct": round(blended_pure_rol_pct, 4),
            "blended_pure_rol": round(blended_pure_rol_pct / 100, 4),
            "expense_load": EXPENSE_LOAD,
            "profit_target": PROFIT_TARGET,
            "uncertainty_load": UNCERTAINTY_LOAD,
            "total_load": total_load,
            "technical_rol_pct": round(technical_rol_pct, 4),
            "technical_rol": round(technical_rol_pct / 100, 4),
            "pure_premium": round(pure_premium, 2),
            "technical_premium": round(technical_premium, 2),
            "technical_premium_usd": round(technical_premium, 2),
            "market_year": latest_market_year,
            "market_rol_pct": round(latest_market_rol_pct, 4),
            "market_benchmark_rol": round(latest_market_rol_pct, 4),
            "adequacy_vs_market_pct": round(adequacy_vs_market_pct, 2),
            "recommendation": recommendation,
            "burning_cost_rol": round(method_comparison.loc[method_comparison["method"] == "Burning Cost", "pure_rol_pct"].iloc[0] / 100, 4),
            "freq_severity_rol": round(method_comparison.loc[method_comparison["method"] == "Frequency-Severity", "pure_rol_pct"].iloc[0] / 100, 4),
            "exposure_rating_rol": round(method_comparison.loc[method_comparison["method"] == "Exposure Rating", "pure_rol_pct"].iloc[0] / 100, 4),
        }
    ])


def print_results(method_comparison, pricing_output):
    print("\nCatPriceAI Actuarial Pricing Engine")
    print("-----------------------------------")
    print("\nTreaty")
    print("------")
    print(f"Layer: ${LIMIT:,.0f} excess of ${ATTACHMENT:,.0f}")
    print("\nPricing Loads")
    print("-------------")
    print(f"Expense Load: {EXPENSE_LOAD:.0%}")
    print(f"Profit Target: {PROFIT_TARGET:.0%}")
    print(f"Uncertainty Load: {UNCERTAINTY_LOAD:.0%}")
    print("\nThree-Method Comparison")
    print("-----------------------")
    print(method_comparison[["method", "pure_rol_pct", "weight", "weighted_pure_rol_pct"]].to_string(index=False))
    row = pricing_output.iloc[0]
    print("\nFinal Pricing Output")
    print("--------------------")
    print(f"Blended Pure ROL: {row['blended_pure_rol_pct']}%")
    print(f"Technical ROL: {row['technical_rol_pct']}%")
    print(f"Pure Premium: ${row['pure_premium']:,.2f}")
    print(f"Technical Premium: ${row['technical_premium']:,.2f}")
    print(f"Market ROL ({row['market_year']}): {row['market_rol_pct']}%")
    print(f"Adequacy vs Market: {row['adequacy_vs_market_pct']}%")
    print(f"Recommendation: {row['recommendation']}")
    print("\nFiles saved:")
    print(METHOD_COMPARISON_PATH)
    print(PRICING_OUTPUT_PATH)


def main() -> None:
    check_required_files()
    burning_cost, exposure_summary, loss_summary, loss_in_layer, market_rol = load_inputs()
    method_results = [
        method_1_burning_cost(burning_cost),
        method_2_frequency_severity(loss_in_layer),
        method_3_exposure_rating(exposure_summary),
    ]
    method_comparison = create_method_comparison(method_results)
    pricing_output = calculate_final_price(method_comparison=method_comparison, market_rol=market_rol)
    method_comparison.to_csv(METHOD_COMPARISON_PATH, index=False)
    pricing_output.to_csv(PRICING_OUTPUT_PATH, index=False)
    pricing_output.to_csv(ML_PRICING_PATH, index=False)
    print_results(method_comparison, pricing_output)


if __name__ == "__main__":
    main()
