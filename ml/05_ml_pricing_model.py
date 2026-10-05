"""
CatPriceAI - ML Pricing Model

This script builds a simple adjustment factor for the actuarial ROL using
market benchmark data and the current cedant exposure summary.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ML_DIR = BASE_DIR / "ml"
ML_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "pct_wood_frame",
    "pct_coastal_10km",
    "pct_flood_zone_A_AE",
    "average_building_age",
    "pct_commercial",
    "deductible_average",
    "historical_loss_ratio",
]
TEST_SIZE = 0.25
RANDOM_STATE = 42


def require_file(path: Path) -> None:
    if not path.exists():
        print(f"Missing required file: {path}")
        sys.exit(1)


def pct_to_decimal(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return default
        value = float(value)
        return value / 100 if value > 1 else value
    except Exception:
        return default


def first_value(df: pd.DataFrame, columns, default=np.nan):
    for col in columns:
        if col in df.columns:
            return df[col].iloc[0]
    return default


def get_base_rol(pricing_df: pd.DataFrame) -> float:
    candidates = [
        "pure_actuarial_rol",
        "technical_rol",
        "technical_rol_decimal",
        "blended_technical_rol",
        "final_rol",
        "recommended_rol",
        "technical_rol_pct",
        "final_rol_pct",
        "recommended_rol_pct",
    ]
    for col in candidates:
        if col in pricing_df.columns:
            value = float(pricing_df[col].iloc[0])
            if col.endswith("_pct") or value > 1:
                return value / 100
            return value
    raise KeyError("No usable ROL column found in pricing_output.csv.")


def build_current_portfolio_features(exposure_df: pd.DataFrame) -> pd.DataFrame:
    if all(col in exposure_df.columns for col in FEATURE_COLS):
        return exposure_df[FEATURE_COLS].copy()
    top_construction = str(first_value(exposure_df, ["Top_Construction"], "")).lower()
    top_occupancy = str(first_value(exposure_df, ["Top_Occupancy"], "")).lower()
    if "wood" in top_construction:
        pct_wood = pct_to_decimal(first_value(exposure_df, ["Top_Construction_TIV_Percentage"], 53.93), default=0.5393)
    else:
        pct_wood = pct_to_decimal(first_value(exposure_df, ["Top_Construction_TIV_Percentage"], 53.93), default=0.5393)
    pct_coastal = pct_to_decimal(first_value(exposure_df, ["Coastal_50km_TIV_Percentage"], 32.53), default=0.3253)
    pct_flood = pct_to_decimal(first_value(exposure_df, ["High_Risk_TIV_Percentage"], 47.97), default=0.4797)
    avg_age = 24.5
    if "commercial" in top_occupancy:
        pct_commercial = pct_to_decimal(first_value(exposure_df, ["Top_Occupancy_TIV_Percentage"], 15.0), default=0.15)
    else:
        pct_commercial = pct_to_decimal(first_value(exposure_df, ["Top_Occupancy_TIV_Percentage"], 15.0), default=0.15)
    deductible_avg = 5000.0
    historical_loss_ratio = 0.62
    mapped = pd.DataFrame([
        {
            "pct_wood_frame": pct_wood,
            "pct_coastal_10km": pct_coastal,
            "pct_flood_zone_A_AE": pct_flood,
            "average_building_age": avg_age,
            "pct_commercial": pct_commercial,
            "deductible_average": deductible_avg,
            "historical_loss_ratio": historical_loss_ratio,
        }
    ])
    print("\nMapped current portfolio features:")
    print(mapped.to_string(index=False))
    return mapped


def build_training_data_from_market(market_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"year", "region", "peril", "market_rol"}
    missing = required_cols - set(market_df.columns)
    if missing:
        raise KeyError(f"market_rol.csv missing columns: {sorted(missing)}")
    df = market_df.copy()
    df["market_rol"] = pd.to_numeric(df["market_rol"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["market_rol", "year", "region", "peril"]).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("market_rol.csv has too few valid rows to train the ML model.")
    region = df["region"].astype(str).str.lower()
    peril = df["peril"].astype(str).str.lower()
    year = df["year"].astype(int)
    out = pd.DataFrame()
    out["pct_wood_frame"] = np.select(
        [region.str.contains("florida"), region.str.contains("gulf"), region.str.contains("southeast")],
        [0.58, 0.54, 0.48],
        default=0.50,
    )
    out["pct_coastal_10km"] = np.select(
        [region.str.contains("florida"), region.str.contains("gulf"), region.str.contains("southeast")],
        [0.60, 0.50, 0.35],
        default=0.40,
    )
    out["pct_flood_zone_A_AE"] = np.select(
        [peril.str.contains("flood"), peril.str.contains("all"), peril.str.contains("hurricane")],
        [0.48, 0.38, 0.28],
        default=0.32,
    )
    out["average_building_age"] = 18 + ((year - year.min()) / max(year.max() - year.min(), 1)) * 14
    out["pct_commercial"] = np.select(
        [region.str.contains("gulf"), region.str.contains("florida"), region.str.contains("southeast")],
        [0.15, 0.12, 0.18],
        default=0.15,
    )
    out["deductible_average"] = np.select(
        [peril.str.contains("flood"), peril.str.contains("hurricane"), peril.str.contains("all")],
        [5000, 10000, 7500],
        default=5000,
    )
    rol_min = df["market_rol"].min()
    rol_max = df["market_rol"].max()
    if rol_max == rol_min:
        out["historical_loss_ratio"] = 0.62
    else:
        out["historical_loss_ratio"] = 0.30 + 0.60 * ((df["market_rol"] - rol_min) / (rol_max - rol_min))
    out["target_adjustment_factor"] = (df["market_rol"] / df["market_rol"].mean()).clip(0.85, 1.55)
    return out


def main() -> None:
    market_path = DATA_DIR / "market_rol.csv"
    exposure_path = DATA_DIR / "exposure_summary.csv"
    pricing_path = DATA_DIR / "pricing_output.csv"
    for path in [market_path, exposure_path, pricing_path]:
        require_file(path)
    print("Loading input datasets...")
    market_df = pd.read_csv(market_path)
    exposure_df = pd.read_csv(exposure_path)
    pricing_df = pd.read_csv(pricing_path)
    target_features = build_current_portfolio_features(exposure_df)
    training_df = build_training_data_from_market(market_df)
    X = training_df[FEATURE_COLS]
    y = training_df["target_adjustment_factor"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"\nTraining models on {len(training_df)} market benchmark rows...")
    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, min_samples_leaf=2)
    gb = GradientBoostingRegressor(n_estimators=250, random_state=RANDOM_STATE, learning_rate=0.05, max_depth=3)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    rf_cv = float(np.mean(cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")))
    gb_cv = float(np.mean(cross_val_score(gb, X_train, y_train, cv=5, scoring="r2")))
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)
    rf_test_r2 = float(r2_score(y_test, rf_pred))
    gb_test_r2 = float(r2_score(y_test, gb_pred))
    rf_mae = float(mean_absolute_error(y_test, rf_pred))
    gb_mae = float(mean_absolute_error(y_test, gb_pred))
    best_model = rf if rf_cv > gb_cv else gb
    selected_model = "Random Forest" if rf_cv > gb_cv else "Gradient Boosting"
    model_results = pd.DataFrame({
        "model_name": ["Random Forest", "Gradient Boosting"],
        "mean_cv_r2": [rf_cv, gb_cv],
        "test_r2": [rf_test_r2, gb_test_r2],
        "test_mae": [rf_mae, gb_mae],
        "is_production_winner": [selected_model == "Random Forest", selected_model == "Gradient Boosting"],
    })
    model_results.to_csv(ML_DIR / "ml_model_results.csv", index=False)
    feature_importance = pd.DataFrame({
        "feature_name": FEATURE_COLS,
        "importance_score": best_model.feature_importances_,
    }).sort_values("importance_score", ascending=False)
    feature_importance.to_csv(ML_DIR / "feature_importance.csv", index=False)
    adjustment_factor = float(best_model.predict(target_features)[0])
    base_rol = get_base_rol(pricing_df)
    ml_adjusted_rol = base_rol * adjustment_factor
    cedant_name = pricing_df["cedant_name"].iloc[0] if "cedant_name" in pricing_df.columns else "Gulf Coast Insurance Co"
    ml_pricing = pd.DataFrame([
        {
            "cedant_name": cedant_name,
            "base_actuarial_rol": base_rol,
            "ml_adjustment_factor": adjustment_factor,
            "ml_adjusted_rol": ml_adjusted_rol,
            "base_actuarial_rol_pct": base_rol * 100,
            "ml_adjusted_rol_pct": ml_adjusted_rol * 100,
            "selected_model": selected_model,
        }
    ])
    ml_pricing.to_csv(DATA_DIR / "ml_pricing.csv", index=False)
    print("\n=======================================================")
    print("ML PRICING PIPELINE COMPLETE")
    print("=======================================================")
    print(f"Selected Model          : {selected_model}")
    print(f"Random Forest CV R²     : {rf_cv:.4f}")
    print(f"Gradient Boosting CV R² : {gb_cv:.4f}")
    print("-------------------------------------------------------")
    print(f"ML Adjustment Factor    : {adjustment_factor:.4f}x")
    print(f"Base Actuarial ROL      : {base_rol * 100:.2f}%")
    print(f"ML-Adjusted ROL         : {ml_adjusted_rol * 100:.2f}%")
    print("\nSaved outputs:")
    print("  ml/feature_importance.csv")
    print("  ml/ml_model_results.csv")
    print("  data/ml_pricing.csv")


if __name__ == "__main__":
    main()
