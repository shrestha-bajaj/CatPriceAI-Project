"""
CatPriceAI - Exposure Analysis

This script analyzes the synthetic cedant portfolio and produces a
single-row exposure summary used by downstream pricing and ML modules.
"""

from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "catprice.db"
OUTPUT_PATH = DATA_DIR / "exposure_summary.csv"

STATE_CONCENTRATION_THRESHOLD = 40.0
COASTAL_DISTANCE_KM = 50.0
HIGH_RISK_FLOOD_ZONES = ["A", "AE", "VE"]


def load_portfolio() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run python data/01_generate_data.py first."
        )
    with sqlite3.connect(DB_PATH) as connection:
        portfolio = pd.read_sql_query("SELECT * FROM portfolio", connection)
    return portfolio


def analyze_tiv_by_state(portfolio: pd.DataFrame) -> pd.DataFrame:
    total_tiv = portfolio["tiv_usd"].sum()
    state_summary = (
        portfolio.groupby("state")
        .agg(
            location_count=("location_id", "count"),
            total_tiv=("tiv_usd", "sum"),
            average_tiv=("tiv_usd", "mean"),
        )
        .reset_index()
    )
    state_summary["tiv_share_pct"] = state_summary["total_tiv"] / total_tiv * 100
    state_summary["concentration_flag"] = state_summary["tiv_share_pct"] > STATE_CONCENTRATION_THRESHOLD
    return state_summary.sort_values("total_tiv", ascending=False)


def analyze_coastal_exposure(portfolio: pd.DataFrame) -> dict:
    total_tiv = portfolio["tiv_usd"].sum()
    coastal_portfolio = portfolio[portfolio["dist_to_coast_km"] <= COASTAL_DISTANCE_KM]
    coastal_tiv = coastal_portfolio["tiv_usd"].sum()
    return {
        "coastal_location_count": len(coastal_portfolio),
        "coastal_tiv": coastal_tiv,
        "coastal_tiv_pct": coastal_tiv / total_tiv * 100,
    }


def analyze_flood_zone_exposure(portfolio: pd.DataFrame) -> pd.DataFrame:
    total_tiv = portfolio["tiv_usd"].sum()
    flood_summary = (
        portfolio.groupby("flood_zone")
        .agg(
            location_count=("location_id", "count"),
            total_tiv=("tiv_usd", "sum"),
        )
        .reset_index()
    )
    flood_summary["tiv_share_pct"] = flood_summary["total_tiv"] / total_tiv * 100
    flood_summary["high_risk_flag"] = flood_summary["flood_zone"].isin(HIGH_RISK_FLOOD_ZONES)
    return flood_summary.sort_values("total_tiv", ascending=False)


def analyze_construction_exposure(portfolio: pd.DataFrame) -> pd.DataFrame:
    total_tiv = portfolio["tiv_usd"].sum()
    construction_summary = (
        portfolio.groupby("construction_type")
        .agg(
            location_count=("location_id", "count"),
            total_tiv=("tiv_usd", "sum"),
            average_tiv=("tiv_usd", "mean"),
        )
        .reset_index()
    )
    construction_summary["tiv_share_pct"] = construction_summary["total_tiv"] / total_tiv * 100
    return construction_summary.sort_values("total_tiv", ascending=False)


def create_single_row_summary(
    portfolio: pd.DataFrame,
    state_summary: pd.DataFrame,
    coastal_summary: dict,
    flood_summary: pd.DataFrame,
    construction_summary: pd.DataFrame,
) -> pd.DataFrame:
    total_tiv = portfolio["tiv_usd"].sum()
    total_locations = len(portfolio)
    top_state = state_summary.iloc[0]
    highest_state_tiv_share_pct = top_state["tiv_share_pct"]
    concentration_flag = top_state["concentration_flag"]
    high_risk_flood_tiv = flood_summary.loc[flood_summary["high_risk_flag"], "total_tiv"].sum()
    high_risk_flood_tiv_pct = high_risk_flood_tiv / total_tiv * 100
    wood_frame_row = construction_summary[construction_summary["construction_type"] == "Wood Frame"]
    top_construction = construction_summary.iloc[0]["construction_type"] if not construction_summary.empty else "Unknown"
    wood_frame_tiv_pct = float(wood_frame_row["tiv_share_pct"].iloc[0]) if not wood_frame_row.empty else 0.0
    coastal_flag = coastal_summary["coastal_tiv_pct"] > 50.0
    flood_flag = high_risk_flood_tiv_pct > 40.0
    wood_frame_flag = wood_frame_tiv_pct > 50.0
    risk_flag_count = sum([concentration_flag, coastal_flag, flood_flag, wood_frame_flag])
    if risk_flag_count >= 3:
        overall_risk_level = "High"
    elif risk_flag_count == 2:
        overall_risk_level = "Medium"
    else:
        overall_risk_level = "Low"
    return pd.DataFrame([
        {
            "total_locations": total_locations,
            "total_tiv": round(total_tiv, 2),
            "Total_TIV": round(total_tiv, 2),
            "highest_tiv_state": top_state["state"],
            "Top_State": top_state["state"],
            "highest_state_tiv_share_pct": round(highest_state_tiv_share_pct, 2),
            "Top_State_TIV_Percentage": round(highest_state_tiv_share_pct, 2),
            "state_concentration_flag": bool(concentration_flag),
            "Top_State_Concentration_Flag": bool(concentration_flag),
            "coastal_location_count": coastal_summary["coastal_location_count"],
            "coastal_tiv": round(coastal_summary["coastal_tiv"], 2),
            "coastal_tiv_pct": round(coastal_summary["coastal_tiv_pct"], 2),
            "Coastal_50km_TIV_Percentage": round(coastal_summary["coastal_tiv_pct"], 2),
            "coastal_exposure_flag": bool(coastal_flag),
            "high_risk_flood_tiv": round(high_risk_flood_tiv, 2),
            "high_risk_flood_tiv_pct": round(high_risk_flood_tiv_pct, 2),
            "High_Risk_TIV_Percentage": round(high_risk_flood_tiv_pct, 2),
            "flood_zone_flag": bool(flood_flag),
            "wood_frame_tiv_pct": round(wood_frame_tiv_pct, 2),
            "Top_Construction": top_construction,
            "Top_Construction_TIV_Percentage": round(wood_frame_tiv_pct, 2),
            "wood_frame_flag": bool(wood_frame_flag),
            "risk_flag_count": risk_flag_count,
            "overall_risk_level": overall_risk_level,
        }
    ])


def print_results(
    state_summary: pd.DataFrame,
    coastal_summary: dict,
    flood_summary: pd.DataFrame,
    construction_summary: pd.DataFrame,
    exposure_summary: pd.DataFrame,
) -> None:
    print("\nCatPriceAI Exposure Analysis")
    print("----------------------------")
    print("\nTable 1: TIV by State")
    print(state_summary.to_string(index=False))
    print("\nTable 2: Construction Type Exposure")
    print(construction_summary.to_string(index=False))
    print("\nTable 3: Coastal Exposure")
    print(f"Coastal distance threshold: {COASTAL_DISTANCE_KM} km")
    print(f"Coastal location count: {coastal_summary['coastal_location_count']}")
    print(f"Coastal TIV: ${coastal_summary['coastal_tiv']:,.2f}")
    print(f"Coastal TIV %: {coastal_summary['coastal_tiv_pct']:.2f}%")
    print("\nTable 4: Flood Zone Exposure")
    print(flood_summary.to_string(index=False))
    print("\nSingle Row Exposure Summary")
    print(exposure_summary.to_string(index=False))
    print("\nKey Risk Observations")
    row = exposure_summary.iloc[0]
    if row["state_concentration_flag"]:
        print(f"- RED FLAG: {row['highest_tiv_state']} has {row['highest_state_tiv_share_pct']}% of total TIV.")
    else:
        print(f"- State concentration is below the {STATE_CONCENTRATION_THRESHOLD}% red flag threshold.")
    if row["coastal_exposure_flag"]:
        print(f"- RED FLAG: {row['coastal_tiv_pct']}% of TIV is within {COASTAL_DISTANCE_KM} km of the coast.")
    else:
        print(f"- Coastal exposure is {row['coastal_tiv_pct']}% of TIV.")
    if row["flood_zone_flag"]:
        print(f"- RED FLAG: {row['high_risk_flood_tiv_pct']}% of TIV is in high-risk flood zones.")
    else:
        print(f"- High-risk flood zone exposure is {row['high_risk_flood_tiv_pct']}% of TIV.")
    print(f"- Overall risk level: {row['overall_risk_level']}")
    print(f"\nSaved output to: {OUTPUT_PATH}")


def main() -> None:
    portfolio = load_portfolio()
    state_summary = analyze_tiv_by_state(portfolio)
    coastal_summary = analyze_coastal_exposure(portfolio)
    flood_summary = analyze_flood_zone_exposure(portfolio)
    construction_summary = analyze_construction_exposure(portfolio)
    exposure_summary = create_single_row_summary(
        portfolio=portfolio,
        state_summary=state_summary,
        coastal_summary=coastal_summary,
        flood_summary=flood_summary,
        construction_summary=construction_summary,
    )
    exposure_summary.to_csv(OUTPUT_PATH, index=False)
    print_results(
        state_summary=state_summary,
        coastal_summary=coastal_summary,
        flood_summary=flood_summary,
        construction_summary=construction_summary,
        exposure_summary=exposure_summary,
    )


if __name__ == "__main__":
    main()
