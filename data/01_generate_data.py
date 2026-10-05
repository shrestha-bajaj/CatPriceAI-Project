"""
STUDENT 1 — Data Engineer
=========================
Project: CatPriceAI — Property Cat XL Treaty Pricing Workbench
Module:  Synthetic Data Generation + SQLite Database Setup

BUSINESS CONTEXT
----------------
A reinsurer needs cedant data to price a Cat XL treaty. Before real data arrives,
we generate a realistic synthetic portfolio for "Gulf Coast Insurance Co."
The cedant writes homeowners insurance across 5 Gulf Coast states (TX, LA, MS, AL, FL).

WHAT YOU WILL BUILD
--------------------
1. A synthetic cedant portfolio: 500 insured locations with TIV, construction, peril exposure
2. 25 years of historical cat loss events (hurricanes, floods)
3. A SQLite database (catprice.db) with 4 tables

YOUR DELIVERABLE
----------------
- data/catprice.db          (SQLite database)
- data/portfolio.csv        (raw export for other students)
- data/losses.csv           (raw export for other students)
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
DB_PATH  = DATA_DIR / "catprice.db"

# ── 1. CEDANT PORTFOLIO ───────────────────────────────────────────────────────
N_LOCATIONS = 500

STATES = {
    "TX": {"weight": 0.35, "lat_range": (26.0, 30.5), "lon_range": (-97.5, -93.5), "wind_zone": 3},
    "LA": {"weight": 0.25, "lat_range": (28.5, 31.5), "lon_range": (-94.0, -89.0), "wind_zone": 4},
    "MS": {"weight": 0.10, "lat_range": (30.0, 32.5), "lon_range": (-89.5, -88.0), "wind_zone": 3},
    "AL": {"weight": 0.10, "lat_range": (30.0, 32.5), "lon_range": (-88.5, -85.5), "wind_zone": 3},
    "FL": {"weight": 0.20, "lat_range": (24.5, 31.0), "lon_range": (-87.5, -80.0), "wind_zone": 5},
}

CONSTRUCTION_TYPES = ["Wood Frame", "Masonry", "Steel Frame", "Manufactured"]
OCCUPANCY_TYPES    = ["Residential", "Commercial", "Industrial", "Mixed Use"]

def generate_portfolio():
    rows = []
    state_keys    = list(STATES.keys())
    state_weights = [STATES[s]["weight"] for s in state_keys]
    state_assigns = np.random.choice(state_keys, size=N_LOCATIONS, p=state_weights)

    for i, state in enumerate(state_assigns):
        info = STATES[state]
        lat  = np.random.uniform(*info["lat_range"])
        lon  = np.random.uniform(*info["lon_range"])

        # TIV: log-normal distribution ($200k – $5M typical homeowner block)
        tiv = np.random.lognormal(mean=13.1, sigma=0.8)   # mean ~$500k
        tiv = np.clip(tiv, 100_000, 5_000_000)

        construction = np.random.choice(
            CONSTRUCTION_TYPES,
            p=[0.55, 0.25, 0.10, 0.10]  # Gulf Coast is mostly wood frame
        )
        occupancy = np.random.choice(
            OCCUPANCY_TYPES,
            p=[0.70, 0.20, 0.05, 0.05]
        )
        year_built = int(np.random.choice(
            range(1950, 2024),
            p=np.array([1.0] * 74) / 74  # uniform — simplification
        ))

        # Wind vulnerability index: 1 (low) – 5 (high)
        # Depends on wind zone, construction, and age
        age_factor = max(0, (2024 - year_built) / 74)  # 0=new, 1=old
        const_factor = {"Wood Frame": 1.3, "Masonry": 0.8, "Steel Frame": 0.6, "Manufactured": 1.8}[construction]
        wind_vuln = round(
            min(5.0, info["wind_zone"] * 0.5 + age_factor * 1.5 + const_factor * 0.5),
            2
        )

        # Distance to coast (simplified — correlates with longitude for Gulf)
        dist_to_coast_km = round(abs(lon + 90) * 10 + np.random.uniform(0, 50), 1)

        rows.append({
            "location_id":      f"LOC{i+1:04d}",
            "state":            state,
            "latitude":         round(lat, 4),
            "longitude":        round(lon, 4),
            "tiv_usd":          round(tiv, 0),
            "construction_type": construction,
            "occupancy_type":   occupancy,
            "year_built":       year_built,
            "wind_vuln_index":  wind_vuln,
            "dist_to_coast_km": dist_to_coast_km,
            "flood_zone":       np.random.choice(["A", "AE", "X", "AH"], p=[0.20, 0.30, 0.40, 0.10]),
        })

    return pd.DataFrame(rows)


# ── 2. HISTORICAL LOSS EVENTS ────────────────────────────────────────────────
def generate_losses(portfolio: pd.DataFrame):
    """
    Simulate 25 years (1999–2023) of cat loss events.
    Each event hits a subset of locations based on peril/region logic.
    """
    events = []
    total_tiv = portfolio["tiv_usd"].sum()

    # Named storms (simplified track catalogue)
    named_storms = [
        # (year, name, peril, landfall_state, industry_loss_bn, affected_states)
        (1999, "Floyd",    "Hurricane", "FL", 4.5,  ["FL", "AL"]),
        (2001, "Allison",  "Flood",     "TX", 2.5,  ["TX"]),
        (2004, "Charley",  "Hurricane", "FL", 7.5,  ["FL"]),
        (2004, "Ivan",     "Hurricane", "AL", 12.8, ["AL", "FL", "MS"]),
        (2005, "Katrina",  "Hurricane", "LA", 45.0, ["LA", "MS", "AL"]),
        (2005, "Rita",     "Hurricane", "TX", 9.5,  ["TX", "LA"]),
        (2005, "Wilma",    "Hurricane", "FL", 12.0, ["FL"]),
        (2008, "Ike",      "Hurricane", "TX", 19.0, ["TX", "LA"]),
        (2008, "Gustav",   "Hurricane", "LA", 4.5,  ["LA", "MS"]),
        (2012, "Isaac",    "Hurricane", "LA", 1.5,  ["LA", "MS"]),
        (2016, "Matthew",  "Hurricane", "FL", 6.5,  ["FL"]),
        (2017, "Harvey",   "Flood",     "TX", 30.0, ["TX"]),
        (2017, "Irma",     "Hurricane", "FL", 19.0, ["FL"]),
        (2018, "Michael",  "Hurricane", "FL", 7.5,  ["FL", "AL"]),
        (2020, "Laura",    "Hurricane", "LA", 10.0, ["LA", "TX"]),
        (2020, "Zeta",     "Hurricane", "LA", 3.0,  ["LA", "MS", "AL"]),
        (2021, "Ida",      "Hurricane", "LA", 15.0, ["LA", "MS"]),
        (2022, "Ian",      "Hurricane", "FL", 50.0, ["FL"]),
        (2023, "Idalia",   "Hurricane", "FL", 3.5,  ["FL"]),
    ]

    event_id = 1
    for year, name, peril, landfall, industry_loss_bn, affected_states in named_storms:
        # Scale industry loss to our portfolio (market share ~0.03%)
        market_share = total_tiv / 3e12  # crude market share
        gross_loss = industry_loss_bn * 1.2e10 * market_share * np.random.uniform(0.5, 1.5)

        # Split loss across affected states
        affected_portfolio = portfolio[portfolio["state"].isin(affected_states)]
        if len(affected_portfolio) == 0:
            continue

        state_tiv = affected_portfolio.groupby("state")["tiv_usd"].sum()
        for state in affected_states:
            if state not in state_tiv.index:
                continue
            state_share = state_tiv[state] / state_tiv.sum()
            state_loss = gross_loss * state_share * np.random.uniform(0.8, 1.2)

            events.append({
                "event_id":     f"EVT{event_id:04d}",
                "year":         year,
                "event_name":   name,
                "peril":        peril,
                "landfall_state": landfall,
                "affected_state": state,
                "industry_loss_bn": industry_loss_bn,
                "gross_loss_usd": round(state_loss, 0),
                "cedant_name":  "Gulf Coast Insurance Co",
            })
        event_id += 1

    # Add some flood-only years with smaller events
    for year in [2000, 2002, 2003, 2006, 2007, 2009, 2010, 2011, 2013, 2014, 2015, 2019]:
        state = np.random.choice(list(STATES.keys()))
        events.append({
            "event_id":       f"EVT{event_id:04d}",
            "year":           year,
            "event_name":     f"Flood_{year}",
            "peril":          "Flood",
            "landfall_state": state,
            "affected_state": state,
            "industry_loss_bn": round(np.random.uniform(0.3, 2.0), 2),
            "gross_loss_usd": round(np.random.uniform(500_000, 5_000_000), 0),
            "cedant_name":    "Gulf Coast Insurance Co",
        })
        event_id += 1

    return pd.DataFrame(events).sort_values(["year", "event_id"]).reset_index(drop=True)


# ── 3. TREATY TERMS TABLE ─────────────────────────────────────────────────────
def generate_treaty_terms():
    return pd.DataFrame([{
        "treaty_id":           "TRT-2024-001",
        "cedant_name":         "Gulf Coast Insurance Co",
        "treaty_type":         "Cat XL",
        "layer_attachment_usd": 50_000_000,
        "layer_limit_usd":     50_000_000,
        "layer_exhaustion_usd": 100_000_000,
        "perils_covered":      "Hurricane;Flood",
        "territories":         "TX;LA;MS;AL;FL",
        "inception_date":      "2024-01-01",
        "expiry_date":         "2024-12-31",
        "reinstatements":      2,
        "aggregate_retention": 0,
        "cession_pct":         1.0,    # 100% quota — Cat XL is typically full cession
        "provisional_rate_pct": None,  # to be filled by the model
    }])


# ── 4. MARKET ROL BENCHMARKS ─────────────────────────────────────────────────
def generate_market_rol():
    """Historical market Rate-on-Line data by peril/region — used for ML training."""
    rows = []
    np.random.seed(99)
    for year in range(2005, 2024):
        for region in ["Gulf Coast", "Southeast", "Florida Only"]:
            for peril in ["Hurricane", "Flood", "All Perils"]:
                # ROL cycles: post-Katrina spike (2006–2008), softening, Ian spike
                base_rol = {"Hurricane": 0.18, "Flood": 0.08, "All Perils": 0.22}[peril]
                katrina_bump = 0.12 if year in [2006, 2007, 2008] else 0
                ian_bump     = 0.08 if year in [2023] else 0
                rol = base_rol + katrina_bump + ian_bump + np.random.uniform(-0.02, 0.02)
                rows.append({
                    "year":       year,
                    "region":     region,
                    "peril":      peril,
                    "market_rol": round(max(0.04, rol), 4),
                })
    return pd.DataFrame(rows)


# ── 5. WRITE TO SQLITE ────────────────────────────────────────────────────────
def write_to_db(portfolio, losses, treaty, market_rol):
    conn = sqlite3.connect(DB_PATH)

    portfolio.to_sql("portfolio",   conn, if_exists="replace", index=False)
    losses.to_sql("loss_events",    conn, if_exists="replace", index=False)
    treaty.to_sql("treaty_terms",   conn, if_exists="replace", index=False)
    market_rol.to_sql("market_rol", conn, if_exists="replace", index=False)

    # Create useful indices
    conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_state ON portfolio(state)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_losses_year ON loss_events(year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_losses_peril ON loss_events(peril)")
    conn.commit()
    conn.close()
    print(f"Database written to: {DB_PATH}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating cedant portfolio...")
    portfolio = generate_portfolio()
    print(f"  {len(portfolio)} locations | TIV = ${portfolio['tiv_usd'].sum()/1e9:.2f}B")

    print("Generating historical loss events...")
    losses = generate_losses(portfolio)
    print(f"  {len(losses)} loss records across {losses['year'].nunique()} years")

    print("Generating treaty terms...")
    treaty = generate_treaty_terms()

    print("Generating market ROL benchmarks...")
    market_rol = generate_market_rol()

    print("Writing to SQLite database...")
    write_to_db(portfolio, losses, treaty, market_rol)

    # Export CSVs for other students
    portfolio.to_csv(DATA_DIR / "portfolio.csv", index=False)
    losses.to_csv(DATA_DIR / "losses.csv", index=False)
    market_rol.to_csv(DATA_DIR / "market_rol.csv", index=False)
    print("CSVs exported.")

    # Quick sanity check
    print("\n── Quick Stats ──")
    print(portfolio.groupby("state")["tiv_usd"].agg(["count", "sum"]).assign(
        sum=lambda x: x["sum"].map(lambda v: f"${v/1e6:.1f}M")
    ))
    print(f"\nTotal gross losses: ${losses['gross_loss_usd'].sum()/1e6:.1f}M over 25 years")
    print("Done ✓")
