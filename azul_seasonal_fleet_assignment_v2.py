
"""
Azul Seasonal Fleet Assignment Engine v2
========================================

This version combines two datasets:

1. Operational flight records:
   - STDUTC
   - DepartureStation / ArrivalStation
   - FlightNumber
   - Equipment
   - TripFuelBurnTotal [KG]
   - TakeOffWeight [KG]
   - TripTimeSec
   - GroundDistance [NM]

2. Load / demand summary sheet:
   - DepartureStation / ArrivalStation
   - FlightNumber
   - EquipmentType
   - AVG_PassengerTotal
   - AVG_PassengerWeight
   - AVG_ZeroFuelWeight
   - AVG_DryOperatingWeight
   - AVG_CargoLoadKg
   - AVG_BaggageLoadKg

Main idea:
- Use the operational data for seasonality.
- Use the new load sheet to estimate typical passenger/cargo demand for each flight/route/equipment.
- Identify route-months where demand increases but aircraft size does not.
- Highlight E1 -> E2 opportunities where E2 has evidence of lower fuel intensity or better scaling.

Run:
python azul_seasonal_fleet_assignment_v2.py \
    --ops Azul_DataSample.xlsx \
    --load Azul_Project_Data_2.xlsx \
    --output_dir azul_v2_outputs
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Helpers
# ==========================================================

def clean_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def find_column(df: pd.DataFrame, candidates, required=True):
    lookup = {clean_colname(c): c for c in df.columns}

    for cand in candidates:
        key = clean_colname(cand)
        if key in lookup:
            return lookup[key]

    for cand in candidates:
        key = clean_colname(cand)
        for ncol, original in lookup.items():
            if key in ncol or ncol in key:
                return original

    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def robust_mode(series: pd.Series):
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else np.nan


def pct_change_safe(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return (a - b) / b


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")[:90]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .xlsx, .xls, or .csv")


def setup_plot_style():
    plt.rcParams.update({
        "figure.figsize": (11, 6),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "font.size": 11,
    })


# ==========================================================
# Load and clean operational data
# ==========================================================

def prepare_ops(df_raw: pd.DataFrame, strict=False):
    df = df_raw.copy()
    df.columns = df.columns.astype(str).str.strip()

    cols = {
        "date": find_column(df, ["STDUTC", "FlightDate", "Date"]),
        "dep": find_column(df, ["DepartureStation", "Origin", "From"]),
        "arr": find_column(df, ["ArrivalStation", "Destination", "To"]),
        "flight": find_column(df, ["FlightNumber", "FlightNo", "FltNum"]),
        "equipment": find_column(df, ["Equipment", "Aircraft", "AircraftType", "Fleet"]),
        "fuel": find_column(df, ["TripFuelBurnTotal [KG]", "TripFuelBurnTotalKG", "FuelBurn", "Fuel"], required=False),
        "tow": find_column(df, ["TakeOffWeight  [KG]", "TakeOffWeight [KG]", "TakeoffWeightKG", "TOW"], required=False),
        "time": find_column(df, ["TripTimeSec", "TripTime", "FlightTimeSec", "BlockTimeSec"], required=False),
        "distance": find_column(df, ["GroundDistance [NM]", "GroundDistanceNM", "DistanceNM", "Distance"], required=False),
    }

    df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce")
    df = df.dropna(subset=[cols["date"], cols["dep"], cols["arr"], cols["equipment"], cols["flight"]]).copy()

    df["DepartureStation"] = df[cols["dep"]].astype(str).str.strip()
    df["ArrivalStation"] = df[cols["arr"]].astype(str).str.strip()
    df["FlightNumber"] = df[cols["flight"]]
    df["Equipment"] = df[cols["equipment"]].astype(str).str.strip()
    df["Route"] = df["DepartureStation"] + " -> " + df["ArrivalStation"]
    df["MonthStart"] = df[cols["date"]].dt.to_period("M").dt.to_timestamp()
    df["Year"] = df[cols["date"]].dt.year
    df["Month"] = df[cols["date"]].dt.month
    df["MonthName"] = df[cols["date"]].dt.strftime("%b")

    report = [("initial_rows_after_required_fields", len(df))]

    def apply_filter(mask, label):
        nonlocal df
        before = len(df)
        df = df[mask].copy()
        report.append((label, before - len(df)))

    if cols["fuel"]:
        apply_filter(df[cols["fuel"]].notna(), "missing_fuel_removed")
        apply_filter(df[cols["fuel"]] > 0, "nonpositive_fuel_removed")

    if cols["distance"]:
        apply_filter(df[cols["distance"]].notna(), "missing_distance_removed")
        apply_filter(df[cols["distance"]] > 0, "nonpositive_distance_removed")

    if cols["time"]:
        min_time = 900 if strict else 600
        apply_filter(df[cols["time"]].notna(), "missing_time_removed")
        apply_filter(df[cols["time"]] >= min_time, f"trip_time_below_{min_time}_sec_removed")

    if cols["tow"]:
        apply_filter(df[cols["tow"]].notna(), "missing_tow_removed")
        apply_filter(df[cols["tow"]].between(5000, 350000), "implausible_tow_removed")

    if cols["fuel"] and cols["distance"]:
        df["FuelPerNM"] = df[cols["fuel"]] / df[cols["distance"]].replace(0, np.nan)
        lo, hi = (0.5, 35) if strict else (0.2, 50)
        apply_filter(df["FuelPerNM"].between(lo, hi), "implausible_fuel_per_nm_removed")

    if cols["distance"] and cols["time"]:
        df["GroundSpeedApprox"] = df[cols["distance"]] / (df[cols["time"]] / 3600)
        lo, hi = (100, 650) if strict else (80, 700)
        apply_filter(df["GroundSpeedApprox"].between(lo, hi), "implausible_ground_speed_removed")

    report.append(("final_rows", len(df)))
    return df, cols, pd.DataFrame(report, columns=["Step", "RowsRemoved_or_Count"])


# ==========================================================
# Load and clean demand/load sheet
# ==========================================================

def prepare_load(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.columns = df.columns.astype(str).str.strip()

    cols = {
        "dep": find_column(df, ["DepartureStation", "Origin", "From"]),
        "arr": find_column(df, ["ArrivalStation", "Destination", "To"]),
        "flight": find_column(df, ["FlightNumber", "FlightNo", "FltNum"]),
        "equipment": find_column(df, ["EquipmentType", "Equipment", "AircraftType"]),
        "pax_total": find_column(df, ["AVG_PassengerTotal", "PassengerTotal", "AvgPassengers"], required=False),
        "pax_weight": find_column(df, ["AVG_PassengerWeight", "PassengerWeight"], required=False),
        "zfw": find_column(df, ["AVG_ZeroFuelWeight", "ZeroFuelWeight"], required=False),
        "dow": find_column(df, ["AVG_DryOperatingWeight", "DryOperatingWeight"], required=False),
        "cargo": find_column(df, ["AVG_CargoLoadKg", "CargoLoadKg"], required=False),
        "bags": find_column(df, ["AVG_BaggageLoadKg", "BaggageLoadKg"], required=False),
    }

    df["DepartureStation"] = df[cols["dep"]].astype(str).str.strip()
    df["ArrivalStation"] = df[cols["arr"]].astype(str).str.strip()
    df["Route"] = df["DepartureStation"] + " -> " + df["ArrivalStation"]
    df["FlightNumber"] = df[cols["flight"]]
    df["Equipment"] = df[cols["equipment"]].astype(str).str.strip()

    # Replace missing cargo/bags with zero only because no cargo/bags is operationally plausible.
    # Do NOT replace passenger totals with zero unless the source says zero.
    for field in ["cargo", "bags"]:
        if cols[field]:
            df[cols[field]] = df[cols[field]].fillna(0)

    if cols["pax_total"]:
        df["PassengerRecordValid"] = df[cols["pax_total"]].notna() & (df[cols["pax_total"]] >= 0)
    else:
        df["PassengerRecordValid"] = False

    # Payload proxy: passenger weight + cargo + bags.
    pieces = []
    for field in ["pax_weight", "cargo", "bags"]:
        if cols[field]:
            pieces.append(df[cols[field]].fillna(0))
    df["EstimatedPayloadKg"] = sum(pieces) if pieces else np.nan

    return df, cols


# ==========================================================
# Join operational data with load evidence
# ==========================================================

def attach_load_evidence(ops: pd.DataFrame, load: pd.DataFrame, load_cols: dict):
    """
    Hierarchical join:
    1. exact route + flight number + equipment
    2. route + equipment aggregate
    3. route aggregate
    4. equipment aggregate
    """

    fields = [
        "AVG_PassengerTotal",
        "AVG_PassengerWeight",
        "AVG_ZeroFuelWeight",
        "AVG_DryOperatingWeight",
        "AVG_CargoLoadKg",
        "AVG_BaggageLoadKg",
        "EstimatedPayloadKg",
    ]
    fields = [f for f in fields if f in load.columns]

    # 1 exact
    exact = load[["Route", "FlightNumber", "Equipment"] + fields].drop_duplicates(
        ["Route", "FlightNumber", "Equipment"]
    )
    out = ops.merge(exact, on=["Route", "FlightNumber", "Equipment"], how="left", suffixes=("", "_exact"))

    out["LoadJoinLevel"] = pd.Series(pd.NA, index=out.index, dtype="object")

    if fields:
        out.loc[out[fields[0]].notna(), "LoadJoinLevel"] = "route_flight_equipment"

    # 2 route + equipment fallback
    route_eq = (
        load.groupby(["Route", "Equipment"], as_index=False)[fields]
        .mean(numeric_only=True)
    )
    route_eq = route_eq.rename(columns={f: f"{f}_route_eq" for f in fields})
    out = out.merge(route_eq, on=["Route", "Equipment"], how="left")

    for f in fields:
        fallback = f"{f}_route_eq"
        out[f] = out[f].combine_first(out[fallback])
        out["LoadJoinLevel"] = np.where(
            out["LoadJoinLevel"].isna() & out[fallback].notna(),
            "route_equipment",
            out["LoadJoinLevel"],
        )

    # 3 route fallback
    route = (
        load.groupby("Route", as_index=False)[fields]
        .mean(numeric_only=True)
    )
    route = route.rename(columns={f: f"{f}_route" for f in fields})
    out = out.merge(route, on="Route", how="left")

    for f in fields:
        fallback = f"{f}_route"
        out[f] = out[f].combine_first(out[fallback])
        out["LoadJoinLevel"] = np.where(
            out["LoadJoinLevel"].isna() & out[fallback].notna(),
            "route",
            out["LoadJoinLevel"],
        )

    # 4 equipment fallback
    equip = (
        load.groupby("Equipment", as_index=False)[fields]
        .mean(numeric_only=True)
    )
    equip = equip.rename(columns={f: f"{f}_equipment" for f in fields})
    out = out.merge(equip, on="Equipment", how="left")

    for f in fields:
        fallback = f"{f}_equipment"
        out[f] = out[f].combine_first(out[fallback])
        out["LoadJoinLevel"] = np.where(
            out["LoadJoinLevel"].isna() & out[fallback].notna(),
            "equipment",
            out["LoadJoinLevel"],
        )

    out["LoadJoinLevel"] = out["LoadJoinLevel"].fillna("unmatched")

    # Remove intermediate fallback columns
    drop_cols = [c for c in out.columns if c.endswith("_route_eq") or c.endswith("_route") or c.endswith("_equipment")]
    out = out.drop(columns=drop_cols)

    return out

def add_fuel_savings_estimates(recs: pd.DataFrame, eq_route_fuel: pd.DataFrame):
    recs = recs.copy()

    # Default values
    recs["RecommendedAircraft"] = recs["PrimaryAircraft"]
    recs["RecommendedAircraftFuelPerNM"] = np.nan
    recs["EstimatedFuelSavedKgPerRouteMonth"] = np.nan
    recs["EstimatedFuelSavedPercentOfRouteMonthFuel"] = np.nan
    # Backward-compatible aliases retained for downstream files that already read these names.
    recs["EstimatedFuelSavedKg"] = np.nan
    recs["EstimatedFuelSavedPercent"] = np.nan

    # Recommendation mapping
    # Main focus: E1 -> E2
    e1_to_e2_mask = recs["Recommendation"].isin([
        "High-priority E1 -> E2 candidate",
        "E1 -> E2 review candidate"
    ])

    recs.loc[e1_to_e2_mask, "RecommendedAircraft"] = "E2"

    # Build lookup table:
    # route + aircraft -> median fuel per NM
    fuel_lookup = eq_route_fuel.set_index(["Route", "Equipment"])["MedianFuelPerNM"].to_dict()

    def get_recommended_fuel(row):
        key = (row["Route"], row["RecommendedAircraft"])
        return fuel_lookup.get(key, np.nan)

    recs["RecommendedAircraftFuelPerNM"] = recs.apply(get_recommended_fuel, axis=1)

    # Current observed fuel per NM
    if "AvgFuelPerNM" not in recs.columns:
        return recs

    # Need distance and flights
    required = ["AvgFuelPerNM", "RecommendedAircraftFuelPerNM", "AvgDistanceNM", "Flights"]
    for col in required:
        if col not in recs.columns:
            return recs

    valid_savings_mask = (
        e1_to_e2_mask
        & recs["AvgFuelPerNM"].notna()
        & recs["RecommendedAircraftFuelPerNM"].notna()
        & recs["AvgDistanceNM"].notna()
        & recs["Flights"].notna()
    )

    recs.loc[valid_savings_mask, "EstimatedFuelSavedKgPerRouteMonth"] = (
        (recs["AvgFuelPerNM"] - recs["RecommendedAircraftFuelPerNM"])
        * recs["AvgDistanceNM"]
        * recs["Flights"]
    )

    # Do not allow negative savings to appear as positive recommendation.
    recs.loc[
        recs["EstimatedFuelSavedKgPerRouteMonth"] < 0,
        "EstimatedFuelSavedKgPerRouteMonth",
    ] = 0

    # Percent savings compared to current route-month fuel burn.
    if "FuelBurnKg" in recs.columns:
        recs.loc[valid_savings_mask, "EstimatedFuelSavedPercentOfRouteMonthFuel"] = (
            recs.loc[valid_savings_mask, "EstimatedFuelSavedKgPerRouteMonth"]
            / recs.loc[valid_savings_mask, "FuelBurnKg"]
        )

    recs["EstimatedFuelSavedKg"] = recs["EstimatedFuelSavedKgPerRouteMonth"]
    recs["EstimatedFuelSavedPercent"] = recs["EstimatedFuelSavedPercentOfRouteMonthFuel"]

    return recs
# ==========================================================
# Aircraft regimes
# ==========================================================

def build_tow_regimes(df: pd.DataFrame, tow_col: str, min_flights=30):
    rows = []
    for aircraft, g in df.groupby("Equipment"):
        tow = g[tow_col].dropna()
        if len(tow) < min_flights:
            continue

        q10, q25, q50, q75, q90 = tow.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        iqr = q75 - q25

        # Central operating band: tight enough to ignore abnormal operations.
        low = max(q10, q25 - iqr)
        high = min(q90, q75 + iqr)

        rows.append({
            "Aircraft": aircraft,
            "Flights": len(tow),
            "OperationalLowTOW": low,
            "MedianTOW": q50,
            "OperationalHighTOW": high,
            "P10": q10,
            "P25": q25,
            "P75": q75,
            "P90": q90,
        })

    regimes = pd.DataFrame(rows).sort_values("MedianTOW").reset_index(drop=True)
    regimes["TowRegimeRank"] = np.arange(len(regimes))
    return regimes


def build_load_regimes(load: pd.DataFrame, min_records=20):
    rows = []
    if "AVG_PassengerTotal" not in load.columns:
        return pd.DataFrame()

    valid = load[load["AVG_PassengerTotal"].notna() & (load["AVG_PassengerTotal"] > 0)].copy()

    for aircraft, g in valid.groupby("Equipment"):
        if len(g) < min_records:
            continue

        pax = g["AVG_PassengerTotal"]
        payload = g["EstimatedPayloadKg"] if "EstimatedPayloadKg" in g.columns else pd.Series(np.nan, index=g.index)

        rows.append({
            "Aircraft": aircraft,
            "Records": len(g),
            "PaxP10": pax.quantile(0.10),
            "PaxP25": pax.quantile(0.25),
            "PaxMedian": pax.quantile(0.50),
            "PaxP75": pax.quantile(0.75),
            "PaxP90": pax.quantile(0.90),
            "PayloadMedianKg": payload.median(),
            "PayloadP75Kg": payload.quantile(0.75),
        })

    return pd.DataFrame(rows).sort_values("PaxMedian").reset_index(drop=True)


def classify_tow(avg_tow, regimes):
    if pd.isna(avg_tow) or regimes.empty:
        return pd.Series({"TowRegime": "Unknown", "TowRegimeRank": np.nan})

    m = regimes[(regimes["OperationalLowTOW"] <= avg_tow) & (avg_tow <= regimes["OperationalHighTOW"])]
    if len(m):
        chosen = m.iloc[(m["MedianTOW"] - avg_tow).abs().argmin()]
    else:
        chosen = regimes.iloc[(regimes["MedianTOW"] - avg_tow).abs().argmin()]

    return pd.Series({"TowRegime": chosen["Aircraft"], "TowRegimeRank": chosen["TowRegimeRank"]})


# ==========================================================
# Route-month analysis
# ==========================================================

def build_route_monthly(df: pd.DataFrame, ops_cols: dict, tow_regimes: pd.DataFrame):
    tow_col = ops_cols["tow"]
    fuel_col = ops_cols["fuel"]
    dist_col = ops_cols["distance"]

    agg = {
        "Flights": ("Route", "size"),
        "PrimaryAircraft": ("Equipment", robust_mode),
        "DistinctAircraft": ("Equipment", "nunique"),
        "AvgTOW_kg": (tow_col, "mean"),
        "MedianTOW_kg": (tow_col, "median"),
        "AvgPassengers": ("AVG_PassengerTotal", "mean"),
        "TotalPassengerProxy": ("AVG_PassengerTotal", "sum"),
        "AvgPayloadKg": ("EstimatedPayloadKg", "mean"),
        "TotalPayloadKg": ("EstimatedPayloadKg", "sum"),
        "ExactLoadJoinShare": ("LoadJoinLevel", lambda s: (s == "route_flight_equipment").mean()),
    }

    if fuel_col:
        agg["FuelBurnKg"] = (fuel_col, "sum")
        agg["AvgFuelBurnKg"] = (fuel_col, "mean")

    if dist_col:
        agg["AvgDistanceNM"] = (dist_col, "mean")

    if "FuelPerNM" in df.columns:
        agg["AvgFuelPerNM"] = ("FuelPerNM", "mean")

    route_monthly = (
        df.groupby(["Route", "Year", "Month", "MonthStart"], as_index=False)
        .agg(**agg)
        .sort_values(["Route", "MonthStart"])
    )

    if fuel_col:
        route_monthly["FuelPerFlightKg"] = route_monthly["FuelBurnKg"] / route_monthly["Flights"]
        route_monthly["FuelPerPassengerProxy"] = route_monthly["FuelBurnKg"] / route_monthly["TotalPassengerProxy"].replace(0, np.nan)

    if fuel_col and dist_col:
        route_monthly["FuelPerPassengerNMProxy"] = (
            route_monthly["FuelBurnKg"] /
            (route_monthly["TotalPassengerProxy"] * route_monthly["AvgDistanceNM"]).replace(0, np.nan)
        )

    route_monthly[["TowRegime", "TowRegimeRank"]] = route_monthly["AvgTOW_kg"].apply(
        lambda x: classify_tow(x, tow_regimes)
    )

    baseline_agg = {
        "MedianMonthlyFlights": ("Flights", "median"),
        "AvgMonthlyFlights": ("Flights", "mean"),
        "MedianRouteTOW_kg": ("AvgTOW_kg", "median"),
        "MedianRoutePassengers": ("AvgPassengers", "median"),
        "MedianRoutePayloadKg": ("AvgPayloadKg", "median"),
        "MedianTowRegimeRank": ("TowRegimeRank", "median"),
    }

    if "AvgFuelPerNM" in route_monthly.columns:
        baseline_agg["MedianFuelPerNM"] = ("AvgFuelPerNM", "median")
    if "FuelPerPassengerProxy" in route_monthly.columns:
        baseline_agg["MedianFuelPerPassengerProxy"] = ("FuelPerPassengerProxy", "median")
    if "FuelPerPassengerNMProxy" in route_monthly.columns:
        baseline_agg["MedianFuelPerPassengerNMProxy"] = ("FuelPerPassengerNMProxy", "median")

    baseline = route_monthly.groupby("Route", as_index=False).agg(**baseline_agg)
    out = route_monthly.merge(baseline, on="Route", how="left")

    out["FlightsPctVsMedian"] = out.apply(lambda r: pct_change_safe(r["Flights"], r["MedianMonthlyFlights"]), axis=1)
    out["TowPctVsMedian"] = out.apply(lambda r: pct_change_safe(r["AvgTOW_kg"], r["MedianRouteTOW_kg"]), axis=1)
    out["PassengerPctVsMedian"] = out.apply(lambda r: pct_change_safe(r["AvgPassengers"], r["MedianRoutePassengers"]), axis=1)
    out["PayloadPctVsMedian"] = out.apply(lambda r: pct_change_safe(r["AvgPayloadKg"], r["MedianRoutePayloadKg"]), axis=1)
    out["RegimeShiftVsMedian"] = out["TowRegimeRank"] - out["MedianTowRegimeRank"]

    if "AvgFuelPerNM" in out.columns:
        out["FuelPerNMPctVsMedian"] = out.apply(lambda r: pct_change_safe(r["AvgFuelPerNM"], r["MedianFuelPerNM"]), axis=1)
    if "FuelPerPassengerProxy" in out.columns:
        out["FuelPerPassengerPctVsMedian"] = out.apply(
            lambda r: pct_change_safe(r["FuelPerPassengerProxy"], r["MedianFuelPerPassengerProxy"]), axis=1
        )
    if "FuelPerPassengerNMProxy" in out.columns:
        out["FuelPerPassengerNMPctVsMedian"] = out.apply(
            lambda r: pct_change_safe(r["FuelPerPassengerNMProxy"], r["MedianFuelPerPassengerNMProxy"]), axis=1
        )

    return out


def build_equipment_route_fuel(df: pd.DataFrame):
    if "FuelPerNM" not in df.columns:
        return pd.DataFrame()

    eq = (
        df.groupby(["Route", "Equipment"], as_index=False)
        .agg(
            Flights=("Route", "size"),
            AvgFuelPerNM=("FuelPerNM", "mean"),
            MedianFuelPerNM=("FuelPerNM", "median"),
            AvgTOW_kg=("TakeOffWeight  [KG]", "mean") if "TakeOffWeight  [KG]" in df.columns else ("Route", "size"),
            AvgPassengers=("AVG_PassengerTotal", "mean"),
        )
    )

    return eq


# ==========================================================
# Recommendation logic
# ==========================================================

def recommend(route_monthly: pd.DataFrame, eq_route_fuel: pd.DataFrame):
    r = route_monthly.copy()

    high_frequency = r["FlightsPctVsMedian"] >= 0.25
    very_high_frequency = r["FlightsPctVsMedian"] >= 0.50
    low_frequency = r["FlightsPctVsMedian"] <= -0.25

    demand_up = (r["PassengerPctVsMedian"] >= 0.08) | (r["PayloadPctVsMedian"] >= 0.08)
    demand_down = (r["PassengerPctVsMedian"] <= -0.08) & (r["PayloadPctVsMedian"] <= 0.05)

    tow_up = r["TowPctVsMedian"] >= 0.05
    tow_down = r["TowPctVsMedian"] <= -0.05
    regime_up = r["RegimeShiftVsMedian"] >= 1
    regime_down = r["RegimeShiftVsMedian"] <= -1

    fuel_worse = r["FuelPerNMPctVsMedian"] >= 0.04 if "FuelPerNMPctVsMedian" in r.columns else pd.Series(False, index=r.index)
    fuel_pax_worse = r["FuelPerPassengerNMPctVsMedian"] >= 0.04 if "FuelPerPassengerNMPctVsMedian" in r.columns else pd.Series(False, index=r.index)

    r["Recommendation"] = "Maintain current aircraft assignment"
    r["RecommendationReason"] = "No strong seasonal demand, frequency, TOW-regime, or fuel-intensity signal."

    # General upgauge: high demand/frequency but TOW regime not increasing.
    mask = high_frequency & demand_up & ~(tow_up | regime_up)
    r.loc[mask, "Recommendation"] = "Review larger aircraft"
    r.loc[mask, "RecommendationReason"] = (
        "Monthly frequency and load proxy increase, but average TOW/regime does not. "
        "This suggests seasonal demand may be handled by extra flights instead of larger aircraft."
    )

    mask = very_high_frequency & demand_up & ~(regime_up)
    r.loc[mask, "Recommendation"] = "Strong review for larger aircraft"
    r.loc[mask, "RecommendationReason"] = (
        "Flight frequency rises sharply with higher passenger/payload proxy, without a clear shift into a heavier aircraft regime."
    )

    # If fuel worsens at the same time, prioritize.
    mask = high_frequency & demand_up & (fuel_worse | fuel_pax_worse) & ~(regime_up)
    r.loc[mask, "Recommendation"] = "High-priority larger-aircraft review"
    r.loc[mask, "RecommendationReason"] = (
        "Peak month shows higher frequency/load proxy and worse fuel intensity. "
        "A larger aircraft may reduce fuel intensity if service frequency can remain acceptable."
    )

    # Already upgauging effectively.
    mask = high_frequency & demand_up & (tow_up | regime_up) & ~(fuel_worse | fuel_pax_worse)
    r.loc[mask, "Recommendation"] = "Current seasonal upgauge appears effective"
    r.loc[mask, "RecommendationReason"] = (
        "Demand and frequency increase while aircraft TOW/regime also increases without worse fuel intensity."
    )

    # Downgauge.
    mask = low_frequency & demand_down & (tow_up | regime_up)
    r.loc[mask, "Recommendation"] = "Review smaller aircraft"
    r.loc[mask, "RecommendationReason"] = (
        "Low-demand month still shows elevated TOW/regime. Smaller aircraft may better match seasonal demand."
    )

    mask = low_frequency & demand_down & (tow_down | regime_down)
    r.loc[mask, "Recommendation"] = "Current seasonal downgauge appears effective"
    r.loc[mask, "RecommendationReason"] = (
        "Low-demand month shows lower frequency and lower TOW/regime, suggesting current fleet assignment is already responsive."
    )

    # E1 -> E2 specific overlay.
    r["E1toE2Evidence"] = "No specific E1/E2 signal"

    if not eq_route_fuel.empty:
        route_eq = eq_route_fuel.pivot_table(index="Route", columns="Equipment", values="MedianFuelPerNM", aggfunc="first")
        if "E1" in route_eq.columns and "E2" in route_eq.columns:
            route_eq["E2FuelPerNMAdvantageVsE1"] = (route_eq["E1"] - route_eq["E2"]) / route_eq["E1"]
            route_eq = route_eq[["E2FuelPerNMAdvantageVsE1"]].reset_index()
            r = r.merge(route_eq, on="Route", how="left")
        else:
            r["E2FuelPerNMAdvantageVsE1"] = np.nan
    else:
        r["E2FuelPerNMAdvantageVsE1"] = np.nan

    e1_month = r["PrimaryAircraft"].astype(str).str.upper().eq("E1")
    e2_advantage = r["E2FuelPerNMAdvantageVsE1"].fillna(0) >= 0.03

    mask = e1_month & high_frequency & demand_up & e2_advantage
    r.loc[mask, "Recommendation"] = "E1 -> E2 review candidate"
    r.loc[mask, "E1toE2Evidence"] = (
        "Primary aircraft is E1, seasonal frequency/load rises, and route-level E2 fuel per NM is at least 3% better than E1."
    )
    r.loc[mask, "RecommendationReason"] = (
        "This route-month is an E1 to E2 reassignment candidate based on demand increase and observed E2 fuel advantage."
    )

    mask = e1_month & very_high_frequency & demand_up & e2_advantage & (fuel_worse | fuel_pax_worse)
    r.loc[mask, "Recommendation"] = "High-priority E1 -> E2 candidate"
    r.loc[mask, "E1toE2Evidence"] = (
        "E1-operated peak month has strong frequency/load increase, worse fuel intensity, and observed E2 fuel advantage on the route."
    )
    r.loc[mask, "RecommendationReason"] = (
        "Prioritize this route-month for E1 to E2 reassignment review; it combines peak demand, fuel penalty, and E2 efficiency evidence."
    )

    # Score for sorting.
    r["ActionPriorityScore"] = 0.0
    r["ActionPriorityScore"] += r["FlightsPctVsMedian"].fillna(0).clip(lower=0) * 35
    r["ActionPriorityScore"] += r["PassengerPctVsMedian"].fillna(0).clip(lower=0) * 25
    r["ActionPriorityScore"] += r["PayloadPctVsMedian"].fillna(0).clip(lower=0) * 15
    if "FuelPerNMPctVsMedian" in r.columns:
        r["ActionPriorityScore"] += r["FuelPerNMPctVsMedian"].fillna(0).clip(lower=0) * 20
    r["ActionPriorityScore"] += r["E2FuelPerNMAdvantageVsE1"].fillna(0).clip(lower=0) * 60

    return r.sort_values(["ActionPriorityScore", "Flights"], ascending=[False, False])


def summarize_routes(recs: pd.DataFrame):
    agg_dict = {
        "Months": ("MonthStart", "count"),
        "TotalFlights": ("Flights", "sum"),
        "AvgPriorityScore": ("ActionPriorityScore", "mean"),
        "MaxPriorityScore": ("ActionPriorityScore", "max"),
        "AvgPassengers": ("AvgPassengers", "mean"),
        "AvgTOW_kg": ("AvgTOW_kg", "mean"),
        "E2FuelAdvantage": ("E2FuelPerNMAdvantageVsE1", "mean"),
    }

    if "AvgFuelPerNM" in recs.columns:
        agg_dict["AvgFuelPerNM"] = ("AvgFuelPerNM", "mean")

    savings_col = None
    if "EstimatedFuelSavedKgPerRouteMonth" in recs.columns:
        savings_col = "EstimatedFuelSavedKgPerRouteMonth"
        agg_dict["EstimatedFuelSavedKgOverObservedRouteMonths"] = (
            savings_col,
            lambda s: s.sum(min_count=1),
        )
    elif "EstimatedFuelSavedKg" in recs.columns:
        savings_col = "EstimatedFuelSavedKg"
        agg_dict["EstimatedFuelSavedKgOverObservedRouteMonths"] = (
            savings_col,
            lambda s: s.sum(min_count=1),
        )

    sort_cols = ["MaxPriorityScore", "TotalFlights"]
    ascending = [False, False]
    if savings_col is not None:
        sort_cols = ["EstimatedFuelSavedKgOverObservedRouteMonths"] + sort_cols
        ascending = [False] + ascending

    return (
        recs.groupby(["Route", "Recommendation"], as_index=False)
        .agg(**agg_dict)
        .sort_values(sort_cols, ascending=ascending)
    )


# ==========================================================
# Plots
# ==========================================================

def plot_regimes(regimes, output_dir):
    if regimes.empty:
        return
    fig, ax = plt.subplots(figsize=(11, max(5, len(regimes) * 0.45)))
    for _, row in regimes.iterrows():
        ax.plot(
            [row["OperationalLowTOW"], row["OperationalHighTOW"]],
            [row["Aircraft"], row["Aircraft"]],
            linewidth=8,
            solid_capstyle="round",
        )
        ax.scatter(row["MedianTOW"], row["Aircraft"], s=60, zorder=3)
    ax.set_title("Aircraft TOW Operating Regimes (central operating band)")
    ax.set_xlabel("Take-off weight, TOW (kg)")
    ax.set_ylabel("Aircraft type")
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "aircraft_tow_regimes.png", dpi=200)
    plt.close(fig)


def plot_recommendations(recs, output_dir):
    counts = recs["Recommendation"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(counts.index, counts.values)
    ax.set_title("Route-Month Recommendation Counts")
    ax.set_xlabel("Number of route-months")
    ax.set_ylabel("Recommendation")
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "recommendation_counts.png", dpi=200)
    plt.close(fig)


def plot_top_candidates(recs, output_dir, title, filename, rec_names, top_n=15):
    p = recs[recs["Recommendation"].isin(rec_names)].head(top_n)
    if p.empty:
        return
    labels = p["Route"] + " | " + p["MonthStart"].dt.strftime("%Y-%m")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels.iloc[::-1], p["ActionPriorityScore"].iloc[::-1])
    ax.set_title(title)
    ax.set_xlabel("Action priority score (unitless index)")
    ax.set_ylabel("Route-month (city pair | YYYY-MM)")
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / filename, dpi=200)
    plt.close(fig)


def plot_route_profiles(recs, output_dir, top_n=6):
    focus = recs[
        recs["Recommendation"].isin([
            "High-priority E1 -> E2 candidate",
            "E1 -> E2 review candidate",
            "High-priority larger-aircraft review",
            "Strong review for larger aircraft",
            "Review larger aircraft",
            "Review smaller aircraft",
        ])
    ].copy()

    routes = focus.sort_values("ActionPriorityScore", ascending=False)["Route"].drop_duplicates().head(top_n).tolist()

    for route in routes:
        g = recs[recs["Route"] == route].sort_values("MonthStart")

        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(g["MonthStart"], g["Flights"], marker="o", label="Flights")
        ax1.set_ylabel("Number of flights per month")
        ax1.set_xlabel("Month")

        ax2 = ax1.twinx()
        ax2.plot(g["MonthStart"], g["AvgPassengers"], marker="s", linestyle="--", label="Avg Passengers")
        ax2.set_ylabel("Average passengers per flight (load proxy)")

        ax1.set_title(f"Monthly Frequency and Passenger Load Proxy: {route}")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / f"route_profile_{safe_filename(route)}.png", dpi=200)
        plt.close(fig)


# ==========================================================
# Main
# ==========================================================

def print_design_summary(joined, recommendations, tow_regimes, load_regimes):
    print("\nDesign summary")
    print("=" * 80)
    print("1. Seasonality is measured from dated operational records using route-month groups.")
    print("2. Demand/load evidence is attached from the new load sheet using a hierarchy:")
    print("   route + flight number + equipment -> route + equipment -> route -> equipment.")
    print("3. Aircraft operating regimes are inferred from central TOW distributions.")
    print("4. Recommendations compare each route-month against that route's own median behavior.")
    print("5. E1 -> E2 is flagged only when E1 is primary, demand/frequency rises, and E2 shows fuel advantage.")
    print("\nLoad join quality:")
    print(joined["LoadJoinLevel"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")
    print("\nTOW regimes:")
    print(tow_regimes[["Aircraft", "Flights", "OperationalLowTOW", "MedianTOW", "OperationalHighTOW"]].to_string(index=False))
    if not load_regimes.empty:
        print("\nPassenger/load regimes:")
        print(load_regimes[["Aircraft", "Records", "PaxP25", "PaxMedian", "PaxP75", "PayloadMedianKg"]].to_string(index=False))

    focus = recommendations[
        recommendations["Recommendation"].isin([
            "High-priority E1 -> E2 candidate",
            "E1 -> E2 review candidate",
            "High-priority larger-aircraft review",
            "Strong review for larger aircraft",
            "Review larger aircraft",
            "Review smaller aircraft",
        ])
    ].head(20)

    print("\nTop actionable route-months:")
    print("=" * 80)
    if focus.empty:
        print("No strong route-month recommendations under the current thresholds.")
    else:
        for _, r in focus.iterrows():
            print(f"\n{r['Route']} | {pd.Timestamp(r['MonthStart']).strftime('%Y-%m')}")
            print(f"Recommendation: {r['Recommendation']}")
            print(f"Primary aircraft: {r['PrimaryAircraft']} | TOW regime: {r['TowRegime']}")
            print(f"Flights: {r['Flights']:.0f} vs median {r['MedianMonthlyFlights']:.1f} ({r['FlightsPctVsMedian']:.1%})")
            print(f"Avg passengers proxy: {r['AvgPassengers']:.1f} vs median {r['MedianRoutePassengers']:.1f} ({r['PassengerPctVsMedian']:.1%})")
            print(f"Avg TOW: {r['AvgTOW_kg']:,.0f} kg vs median {r['MedianRouteTOW_kg']:,.0f} kg ({r['TowPctVsMedian']:.1%})")
            if pd.notna(r.get("E2FuelPerNMAdvantageVsE1", np.nan)):
                print(f"Observed route E2 fuel advantage vs E1: {r['E2FuelPerNMAdvantageVsE1']:.1%}")
            if pd.notna(r.get("EstimatedFuelSavedKgPerRouteMonth", np.nan)):
                print(f"Estimated fuel saved for route-month: {r['EstimatedFuelSavedKgPerRouteMonth']:,.0f} kg")
                if pd.notna(r.get("EstimatedFuelSavedPercentOfRouteMonthFuel", np.nan)):
                    print(f"Estimated route-month fuel reduction: {r['EstimatedFuelSavedPercentOfRouteMonthFuel']:.1%}")
            print(f"Reason: {r['RecommendationReason']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", required=True, help="Dated operational flight dataset")
    parser.add_argument("--load", required=True, help="Load/passenger summary dataset")
    parser.add_argument("--output_dir", default="azul_v2_outputs")
    parser.add_argument("--strict_cleaning", action="store_true")
    parser.add_argument("--min_aircraft_flights", type=int, default=30)
    args = parser.parse_args()

    setup_plot_style()

    out = Path(args.output_dir)
    (out / "csv").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    ops_raw = read_table(Path(args.ops))
    load_raw = read_table(Path(args.load))

    ops, ops_cols, cleaning_report = prepare_ops(ops_raw, strict=args.strict_cleaning)
    load, load_cols = prepare_load(load_raw)

    joined = attach_load_evidence(ops, load, load_cols)

    tow_regimes = build_tow_regimes(joined, ops_cols["tow"], min_flights=args.min_aircraft_flights)
    load_regimes = build_load_regimes(load)

    route_monthly = build_route_monthly(joined, ops_cols, tow_regimes)
    eq_route_fuel = build_equipment_route_fuel(joined)
    recommendations = recommend(route_monthly, eq_route_fuel)
    recommendations = add_fuel_savings_estimates(recommendations, eq_route_fuel)

    route_summary = summarize_routes(recommendations)

    cleaning_report.to_csv(out / "csv" / "cleaning_report.csv", index=False)
    joined.to_csv(out / "csv" / "joined_cleaned_flight_data.csv", index=False)
    tow_regimes.to_csv(out / "csv" / "aircraft_tow_regimes.csv", index=False)
    load_regimes.to_csv(out / "csv" / "aircraft_passenger_payload_regimes.csv", index=False)
    route_monthly.to_csv(out / "csv" / "route_monthly_metrics.csv", index=False)
    eq_route_fuel.to_csv(out / "csv" / "route_equipment_fuel_comparison.csv", index=False)
    recommendations.to_csv(out / "csv" / "route_month_recommendations.csv", index=False)
    route_summary.to_csv(out / "csv" / "route_recommendation_summary.csv", index=False)

    plot_regimes(tow_regimes, out)
    plot_recommendations(recommendations, out)
    plot_top_candidates(
        recommendations,
        out,
        "Top E1 to E2 Candidates",
        "top_e1_to_e2_candidates.png",
        ["High-priority E1 -> E2 candidate", "E1 -> E2 review candidate"],
    )
    plot_top_candidates(
        recommendations,
        out,
        "Top Larger-Aircraft Review Candidates",
        "top_larger_aircraft_candidates.png",
        ["High-priority larger-aircraft review", "Strong review for larger aircraft", "Review larger aircraft"],
    )
    plot_top_candidates(
        recommendations,
        out,
        "Top Smaller-Aircraft Review Candidates",
        "top_smaller_aircraft_candidates.png",
        ["Review smaller aircraft"],
    )
    plot_route_profiles(recommendations, out)

    print_design_summary(joined, recommendations, tow_regimes, load_regimes)

    print("\nSaved outputs to:")
    print(out.resolve())


if __name__ == "__main__":
    main()
