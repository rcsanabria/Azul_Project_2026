
"""
Azul Seasonal Fleet Assignment Decision Engine
==============================================

Usage:
python azul_seasonal_fleet_decision_engine.py --input Azul_DataSample.xlsx --output_dir azul_decision_outputs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clean_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def find_column(df: pd.DataFrame, candidates, required=True):
    lookup = {clean_colname(c): c for c in df.columns}

    for candidate in candidates:
        key = clean_colname(candidate)
        if key in lookup:
            return lookup[key]

    for candidate in candidates:
        key = clean_colname(candidate)
        for norm_col, original_col in lookup.items():
            if key in norm_col or norm_col in key:
                return original_col

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


def seasonality_index(values: pd.Series) -> float:
    total = values.sum()
    if total <= 0:
        return np.nan
    share = values / total
    uniform = 1 / len(values)
    return float(np.abs(share - uniform).sum())


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")[:80]


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .xlsx, .xls, or .csv")


def standardize_and_clean(df_raw: pd.DataFrame, strict=False):
    df = df_raw.copy()
    df.columns = df.columns.astype(str).str.strip()

    cols = {
        "date": find_column(df, ["STDUTC", "FlightDate", "Date"]),
        "dep": find_column(df, ["DepartureStation", "Origin", "From"]),
        "arr": find_column(df, ["ArrivalStation", "Destination", "To"]),
        "equipment": find_column(df, ["Equipment", "Aircraft", "AircraftType", "Fleet"]),
        "fuel": find_column(df, ["TripFuelBurnTotal [KG]", "TripFuelBurnTotalKG", "FuelBurn", "Fuel"], required=False),
        "distance": find_column(df, ["GroundDistance [NM]", "GroundDistanceNM", "DistanceNM", "Distance"], required=False),
        "time": find_column(df, ["TripTimeSec", "TripTime", "FlightTimeSec", "BlockTimeSec"], required=False),
        "tow": find_column(df, ["TakeOffWeight  [KG]", "TakeOffWeight [KG]", "TakeoffWeightKG", "TOW"], required=False),
    }

    df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce")
    df = df.dropna(subset=[cols["date"], cols["dep"], cols["arr"], cols["equipment"]]).copy()

    df["Route"] = df[cols["dep"]].astype(str).str.strip() + " -> " + df[cols["arr"]].astype(str).str.strip()
    df["MonthStart"] = df[cols["date"]].dt.to_period("M").dt.to_timestamp()
    df["Year"] = df[cols["date"]].dt.year
    df["Month"] = df[cols["date"]].dt.month
    df["MonthName"] = df[cols["date"]].dt.strftime("%b")

    cleaning_report = []
    cleaning_report.append(("initial_rows_after_required_fields", len(df)))

    def apply_filter(condition, label):
        nonlocal df
        before = len(df)
        df = df[condition].copy()
        after = len(df)
        cleaning_report.append((label, before - after))

    if cols["fuel"]:
        apply_filter(df[cols["fuel"]].notna(), "missing_fuel_removed")
        apply_filter(df[cols["fuel"]] > 0, "nonpositive_fuel_removed")

    if cols["distance"]:
        apply_filter(df[cols["distance"]].notna(), "missing_distance_removed")
        apply_filter(df[cols["distance"]] > 0, "nonpositive_distance_removed")

    if cols["time"]:
        apply_filter(df[cols["time"]].notna(), "missing_time_removed")
        min_time = 900 if strict else 600
        apply_filter(df[cols["time"]] >= min_time, f"trip_time_below_{min_time}_sec_removed")

    if cols["tow"]:
        apply_filter(df[cols["tow"]].notna(), "missing_tow_removed")
        apply_filter(df[cols["tow"]].between(5000, 350000), "implausible_tow_removed")

    if cols["fuel"] and cols["distance"]:
        df["FuelPerNM"] = df[cols["fuel"]] / df[cols["distance"]].replace(0, np.nan)
        low, high = (0.5, 35) if strict else (0.2, 50)
        apply_filter(df["FuelPerNM"].between(low, high), "implausible_fuel_per_nm_removed")

    if cols["fuel"] and cols["time"]:
        df["FuelPerHour"] = df[cols["fuel"]] * 3600 / df[cols["time"]].replace(0, np.nan)
        low, high = (200, 40000) if strict else (100, 50000)
        apply_filter(df["FuelPerHour"].between(low, high), "implausible_fuel_per_hour_removed")

    if cols["distance"] and cols["time"]:
        df["GroundSpeedApprox"] = df[cols["distance"]] / (df[cols["time"]] / 3600)
        low, high = (100, 650) if strict else (80, 700)
        apply_filter(df["GroundSpeedApprox"].between(low, high), "implausible_ground_speed_removed")

    cleaning_report.append(("final_rows", len(df)))
    cleaning_report = pd.DataFrame(cleaning_report, columns=["Step", "RowsRemoved_or_Count"])
    return df, cols, cleaning_report


def build_aircraft_regimes(df: pd.DataFrame, cols: dict, min_aircraft_flights: int = 30):
    tow_col = cols["tow"]
    eq_col = cols["equipment"]

    if tow_col is None:
        raise ValueError("TOW column is required for regime-based analysis.")

    rows = []

    for aircraft, g in df.groupby(eq_col):
        tow = g[tow_col].dropna()
        if len(tow) < min_aircraft_flights:
            continue

        q05 = tow.quantile(0.05)
        q10 = tow.quantile(0.10)
        q25 = tow.quantile(0.25)
        q50 = tow.quantile(0.50)
        q75 = tow.quantile(0.75)
        q90 = tow.quantile(0.90)
        q95 = tow.quantile(0.95)

        iqr = q75 - q25
        tight_low = q25 - 1.0 * iqr
        tight_high = q75 + 1.0 * iqr

        operational_low = max(q10, tight_low)
        operational_high = min(q90, tight_high)

        rows.append({
            "Aircraft": aircraft,
            "Flights": len(tow),
            "P05": q05,
            "P10": q10,
            "P25": q25,
            "MedianTOW": q50,
            "P75": q75,
            "P90": q90,
            "P95": q95,
            "IQR": iqr,
            "OperationalLowTOW": operational_low,
            "OperationalHighTOW": operational_high,
            "OperationalWidth": operational_high - operational_low,
        })

    regimes = pd.DataFrame(rows).sort_values("MedianTOW").reset_index(drop=True)
    regimes["RegimeRank"] = np.arange(len(regimes))
    return regimes


def classify_tow_to_regime(avg_tow, regimes: pd.DataFrame):
    if pd.isna(avg_tow):
        return pd.Series({"TowRegime": "Unknown", "RegimeRank": np.nan, "RegimeDistanceKg": np.nan})

    matches = regimes[
        (regimes["OperationalLowTOW"] <= avg_tow) &
        (avg_tow <= regimes["OperationalHighTOW"])
    ]

    if len(matches) > 0:
        chosen = matches.iloc[(matches["MedianTOW"] - avg_tow).abs().argmin()]
        return pd.Series({
            "TowRegime": chosen["Aircraft"],
            "RegimeRank": chosen["RegimeRank"],
            "RegimeDistanceKg": 0.0,
        })

    nearest = regimes.iloc[(regimes["MedianTOW"] - avg_tow).abs().argmin()]
    return pd.Series({
        "TowRegime": f"Nearest: {nearest['Aircraft']}",
        "RegimeRank": nearest["RegimeRank"],
        "RegimeDistanceKg": avg_tow - nearest["MedianTOW"],
    })


def build_route_monthly(df: pd.DataFrame, cols: dict, regimes: pd.DataFrame):
    eq_col = cols["equipment"]
    tow_col = cols["tow"]

    agg = {
        "Flights": ("Route", "size"),
        "PrimaryAircraft": (eq_col, robust_mode),
        "DistinctAircraft": (eq_col, "nunique"),
        "AvgTOW_kg": (tow_col, "mean"),
        "MedianTOW_kg": (tow_col, "median"),
        "MaxTOW_kg": (tow_col, "max"),
        "MinTOW_kg": (tow_col, "min"),
        "TotalTOW_kg": (tow_col, "sum"),
    }

    if cols["fuel"]:
        agg["FuelBurnKg"] = (cols["fuel"], "sum")
        agg["AvgFuelBurnKg"] = (cols["fuel"], "mean")

    if cols["distance"]:
        agg["AvgDistanceNM"] = (cols["distance"], "mean")

    if cols["time"]:
        agg["AvgTripTimeSec"] = (cols["time"], "mean")

    if "FuelPerNM" in df.columns:
        agg["AvgFuelPerNM"] = ("FuelPerNM", "mean")

    if "FuelPerHour" in df.columns:
        agg["AvgFuelPerHour"] = ("FuelPerHour", "mean")

    route_monthly = (
        df.groupby(["Route", "Year", "Month", "MonthStart"], as_index=False)
        .agg(**agg)
        .sort_values(["Route", "MonthStart"])
    )

    if cols["fuel"]:
        route_monthly["FuelPerFlightKg"] = route_monthly["FuelBurnKg"] / route_monthly["Flights"]

    route_monthly[["TowRegime", "TowRegimeRank", "RegimeDistanceKg"]] = route_monthly["AvgTOW_kg"].apply(
        lambda x: classify_tow_to_regime(x, regimes)
    )

    baseline = (
        route_monthly.groupby("Route")
        .agg(
            MedianMonthlyFlights=("Flights", "median"),
            AvgMonthlyFlights=("Flights", "mean"),
            StdMonthlyFlights=("Flights", "std"),
            MedianRouteAvgTOW_kg=("AvgTOW_kg", "median"),
            MedianTowRegimeRank=("TowRegimeRank", "median"),
        )
        .reset_index()
    )

    if "AvgFuelPerNM" in route_monthly.columns:
        fuel_baseline = (
            route_monthly.groupby("Route")["AvgFuelPerNM"]
            .median()
            .rename("MedianRouteFuelPerNM")
            .reset_index()
        )
        baseline = baseline.merge(fuel_baseline, on="Route", how="left")

    route_monthly = route_monthly.merge(baseline, on="Route", how="left")

    route_monthly["FlightsAboveMedian"] = route_monthly["Flights"] - route_monthly["MedianMonthlyFlights"]
    route_monthly["FlightsPctVsMedian"] = route_monthly.apply(
        lambda r: pct_change_safe(r["Flights"], r["MedianMonthlyFlights"]), axis=1
    )

    route_monthly["AvgTOWAboveMedian_kg"] = route_monthly["AvgTOW_kg"] - route_monthly["MedianRouteAvgTOW_kg"]
    route_monthly["AvgTOWPctVsMedian"] = route_monthly.apply(
        lambda r: pct_change_safe(r["AvgTOW_kg"], r["MedianRouteAvgTOW_kg"]), axis=1
    )

    route_monthly["RegimeRankShiftVsMedian"] = route_monthly["TowRegimeRank"] - route_monthly["MedianTowRegimeRank"]

    if "AvgFuelPerNM" in route_monthly.columns and "MedianRouteFuelPerNM" in route_monthly.columns:
        route_monthly["FuelPerNMPctVsRouteMedian"] = route_monthly.apply(
            lambda r: pct_change_safe(r["AvgFuelPerNM"], r["MedianRouteFuelPerNM"]), axis=1
        )

    return route_monthly


def build_route_summary(route_monthly: pd.DataFrame):
    rows = []

    for route, g in route_monthly.groupby("Route"):
        g = g.sort_values("MonthStart")
        flights = g["Flights"]

        peak_idx = flights.idxmax()
        trough_idx = flights.idxmin()

        mean_f = flights.mean()
        std_f = flights.std(ddof=0)
        cv = std_f / mean_f if mean_f > 0 else np.nan

        rows.append({
            "Route": route,
            "ActiveMonths": len(g),
            "TotalFlights": flights.sum(),
            "AvgFlightsPerMonth": mean_f,
            "StdFlightsPerMonth": std_f,
            "CoeffVarFlights": cv,
            "PeakMonth": g.loc[peak_idx, "MonthStart"],
            "PeakFlights": g.loc[peak_idx, "Flights"],
            "TroughMonth": g.loc[trough_idx, "MonthStart"],
            "TroughFlights": g.loc[trough_idx, "Flights"],
            "PeakMinusTrough": g.loc[peak_idx, "Flights"] - g.loc[trough_idx, "Flights"],
            "PeakToTroughRatio": (
                g.loc[peak_idx, "Flights"] / g.loc[trough_idx, "Flights"]
                if g.loc[trough_idx, "Flights"] > 0 else np.nan
            ),
            "SeasonalityIndex": seasonality_index(flights),
            "PeakMonthAvgTOW_kg": g.loc[peak_idx, "AvgTOW_kg"],
            "TroughMonthAvgTOW_kg": g.loc[trough_idx, "AvgTOW_kg"],
            "PeakMonthTowRegime": g.loc[peak_idx, "TowRegime"],
            "TroughMonthTowRegime": g.loc[trough_idx, "TowRegime"],
            "PeakMonthRegimeRank": g.loc[peak_idx, "TowRegimeRank"],
            "TroughMonthRegimeRank": g.loc[trough_idx, "TowRegimeRank"],
            "MaxRegimeRankShift": g["RegimeRankShiftVsMedian"].max(),
            "MinRegimeRankShift": g["RegimeRankShiftVsMedian"].min(),
            "MaxAvgTOW_kg": g["AvgTOW_kg"].max(),
            "MinAvgTOW_kg": g["AvgTOW_kg"].min(),
            "DeltaAvgTOW_kg": g["AvgTOW_kg"].max() - g["AvgTOW_kg"].min(),
        })

    summary = pd.DataFrame(rows)
    summary["VolatilityScore"] = (
        summary["CoeffVarFlights"].fillna(0)
        * np.log1p(summary["TotalFlights"])
        * np.sqrt(summary["PeakMinusTrough"].clip(lower=0))
    )

    return summary.sort_values(["VolatilityScore", "TotalFlights"], ascending=[False, False])


def make_recommendations(route_monthly: pd.DataFrame, min_route_flights: int = 20):
    recs = route_monthly.copy()

    high_frequency_spike = recs["FlightsPctVsMedian"] >= 0.25
    very_high_frequency_spike = recs["FlightsPctVsMedian"] >= 0.50
    low_frequency_month = recs["FlightsPctVsMedian"] <= -0.25
    very_low_frequency_month = recs["FlightsPctVsMedian"] <= -0.50

    tow_up = recs["AvgTOWPctVsMedian"] >= 0.05
    tow_down = recs["AvgTOWPctVsMedian"] <= -0.05

    regime_up = recs["RegimeRankShiftVsMedian"] >= 1
    regime_down = recs["RegimeRankShiftVsMedian"] <= -1

    if "FuelPerNMPctVsRouteMedian" in recs.columns:
        fuel_worse = recs["FuelPerNMPctVsRouteMedian"] >= 0.05
        fuel_better = recs["FuelPerNMPctVsRouteMedian"] <= -0.05
    else:
        fuel_worse = pd.Series(False, index=recs.index)
        fuel_better = pd.Series(False, index=recs.index)

    enough_route_volume = recs.groupby("Route")["Flights"].transform("sum") >= min_route_flights

    recs["Recommendation"] = "Maintain current assignment"
    recs["RecommendationReason"] = "No strong frequency, TOW, regime, or fuel-efficiency signal."

    mask = enough_route_volume & high_frequency_spike & ~tow_up & ~regime_up
    recs.loc[mask, "Recommendation"] = "Review larger aircraft"
    recs.loc[mask, "RecommendationReason"] = (
        "Flights are above route median, but average TOW and TOW regime do not increase."
    )

    mask = enough_route_volume & very_high_frequency_spike & ~regime_up
    recs.loc[mask, "Recommendation"] = "Strong review for larger aircraft"
    recs.loc[mask, "RecommendationReason"] = (
        "Flight count is far above route median without a clear move into a heavier aircraft regime."
    )

    mask = enough_route_volume & high_frequency_spike & fuel_worse & ~regime_up
    recs.loc[mask, "Recommendation"] = "High-priority upgauge review"
    recs.loc[mask, "RecommendationReason"] = (
        "Peak frequency increases while fuel burn per NM is worse than the route median."
    )

    mask = enough_route_volume & high_frequency_spike & (tow_up | regime_up)
    recs.loc[mask, "Recommendation"] = "Current peak upgauge appears justified"
    recs.loc[mask, "RecommendationReason"] = (
        "Flights increase and average TOW or TOW regime also increases."
    )

    mask = enough_route_volume & low_frequency_month & (tow_up | regime_up)
    recs.loc[mask, "Recommendation"] = "Review smaller aircraft"
    recs.loc[mask, "RecommendationReason"] = (
        "Flight count is below route median but average TOW or aircraft regime is above median."
    )

    mask = enough_route_volume & very_low_frequency_month & (tow_up | regime_up)
    recs.loc[mask, "Recommendation"] = "Strong review for smaller aircraft"
    recs.loc[mask, "RecommendationReason"] = (
        "Very low frequency month still uses heavier aircraft behavior."
    )

    mask = enough_route_volume & low_frequency_month & (tow_down | regime_down)
    recs.loc[mask, "Recommendation"] = "Current seasonal downgauge appears justified"
    recs.loc[mask, "RecommendationReason"] = (
        "Flights decrease and average TOW or TOW regime also decreases."
    )

    mask = enough_route_volume & high_frequency_spike & (tow_up | regime_up) & fuel_better
    recs.loc[mask, "Recommendation"] = "Efficient peak upgauge candidate"
    recs.loc[mask, "RecommendationReason"] = (
        "Peak demand is associated with heavier aircraft behavior and better fuel burn per NM."
    )

    recs["PriorityScore"] = 0.0
    recs["PriorityScore"] += recs["FlightsPctVsMedian"].fillna(0).clip(lower=0) * 50
    if "FuelPerNMPctVsRouteMedian" in recs.columns:
        recs["PriorityScore"] += recs["FuelPerNMPctVsRouteMedian"].fillna(0).clip(lower=0) * 40
    recs["PriorityScore"] += (-recs["RegimeRankShiftVsMedian"].fillna(0).clip(upper=0)) * 10
    recs["PriorityScore"] += recs["FlightsAboveMedian"].fillna(0).clip(lower=0)

    return recs.sort_values(["PriorityScore", "Flights"], ascending=[False, False])


def summarize_recommendations(recs: pd.DataFrame):
    return (
        recs.groupby(["Route", "Recommendation"], as_index=False)
        .agg(
            Months=("MonthStart", "count"),
            TotalFlights=("Flights", "sum"),
            AvgFlightsPctVsMedian=("FlightsPctVsMedian", "mean"),
            AvgTOWPctVsMedian=("AvgTOWPctVsMedian", "mean"),
            AvgPriorityScore=("PriorityScore", "mean"),
        )
        .sort_values(["AvgPriorityScore", "TotalFlights"], ascending=[False, False])
    )


def print_simple_recommendations(recs: pd.DataFrame, top_n: int = 20):
    focus = recs[
        recs["Recommendation"].isin([
            "High-priority upgauge review",
            "Strong review for larger aircraft",
            "Review larger aircraft",
            "Strong review for smaller aircraft",
            "Review smaller aircraft",
            "Efficient peak upgauge candidate",
        ])
    ].copy()

    if focus.empty:
        print("\nNo strong aircraft-size recommendations found under the current thresholds.")
        return

    focus = focus.sort_values(["PriorityScore", "Flights"], ascending=[False, False]).head(top_n)

    print("\nSimple route-month recommendations")
    print("=" * 80)

    for _, r in focus.iterrows():
        month = pd.Timestamp(r["MonthStart"]).strftime("%Y-%m")
        print(f"\n{r['Route']} | {month}")
        print(f"Recommendation: {r['Recommendation']}")
        print(f"Flights: {r['Flights']:.0f} vs route median {r['MedianMonthlyFlights']:.1f}")
        print(f"Avg TOW: {r['AvgTOW_kg']:,.0f} kg vs route median {r['MedianRouteAvgTOW_kg']:,.0f} kg")
        print(f"TOW regime: {r['TowRegime']}")
        print(f"Reason: {r['RecommendationReason']}")


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


def plot_aircraft_regimes(regimes, output_dir):
    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(regimes))))
    for _, row in regimes.iterrows():
        ax.plot([row["OperationalLowTOW"], row["OperationalHighTOW"]],
                [row["Aircraft"], row["Aircraft"]],
                linewidth=8, solid_capstyle="round")
        ax.scatter(row["MedianTOW"], row["Aircraft"], s=60, zorder=3)
    ax.set_title("Aircraft TOW Operating Regimes")
    ax.set_xlabel("Takeoff Weight (kg)")
    ax.set_ylabel("Aircraft")
    fig.tight_layout()
    fig.savefig(output_dir / "aircraft_tow_regimes.png", dpi=200)
    plt.close(fig)


def plot_top_volatility(route_summary, output_dir, top_n=15):
    top = route_summary.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top["Route"], top["VolatilityScore"])
    ax.set_title(f"Top {top_n} Routes by Seasonal Volatility")
    ax.set_xlabel("Volatility Score")
    ax.set_ylabel("Route")
    fig.tight_layout()
    fig.savefig(output_dir / "top_route_volatility.png", dpi=200)
    plt.close(fig)


def plot_recommendation_counts(recs, output_dir):
    counts = recs["Recommendation"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(counts.index, counts.values)
    ax.set_title("Route-Month Recommendation Counts")
    ax.set_xlabel("Route-months")
    ax.set_ylabel("Recommendation")
    fig.tight_layout()
    fig.savefig(output_dir / "recommendation_counts.png", dpi=200)
    plt.close(fig)


def plot_candidate_bars(recs, output_dir, recommendation_terms, filename, title, top_n=15):
    plot_df = recs[recs["Recommendation"].isin(recommendation_terms)]
    plot_df = plot_df.sort_values(["PriorityScore", "Flights"], ascending=[False, False]).head(top_n)
    if plot_df.empty:
        return
    labels = plot_df["Route"] + " | " + plot_df["MonthStart"].dt.strftime("%Y-%m")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels.iloc[::-1], plot_df["PriorityScore"].iloc[::-1])
    ax.set_title(title)
    ax.set_xlabel("Priority Score")
    ax.set_ylabel("Route-month")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_top_route_profiles(route_monthly, route_summary, output_dir, top_n=5):
    routes = route_summary.head(top_n)["Route"].tolist()
    for route in routes:
        g = route_monthly[route_monthly["Route"] == route].sort_values("MonthStart")
        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(g["MonthStart"], g["Flights"], marker="o", label="Flights")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Flights")
        ax2 = ax1.twinx()
        ax2.plot(g["MonthStart"], g["AvgTOW_kg"], marker="s", linestyle="--", label="Avg TOW")
        ax2.set_ylabel("Average TOW (kg)")
        ax1.set_title(f"Seasonal Frequency and TOW: {route}")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / f"route_profile_{safe_filename(route)}.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Excel or CSV file")
    parser.add_argument("--output_dir", default="azul_decision_outputs")
    parser.add_argument("--strict_cleaning", action="store_true")
    parser.add_argument("--min_aircraft_flights", type=int, default=30)
    parser.add_argument("--min_route_flights", type=int, default=20)
    parser.add_argument("--print_top", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()

    df_raw = load_data(input_path)
    df, cols, cleaning_report = standardize_and_clean(df_raw, strict=args.strict_cleaning)

    if cols["tow"] is None:
        raise ValueError("This advanced version requires a takeoff weight / TOW column.")

    regimes = build_aircraft_regimes(df, cols, min_aircraft_flights=args.min_aircraft_flights)
    route_monthly = build_route_monthly(df, cols, regimes)
    route_summary = build_route_summary(route_monthly)
    recs = make_recommendations(route_monthly, min_route_flights=args.min_route_flights)
    rec_summary = summarize_recommendations(recs)

    cleaning_report.to_csv(output_dir / "cleaning_report.csv", index=False)
    df.to_csv(output_dir / "cleaned_flight_data.csv", index=False)
    regimes.to_csv(output_dir / "aircraft_tow_regimes.csv", index=False)
    route_monthly.to_csv(output_dir / "route_monthly_metrics.csv", index=False)
    route_summary.to_csv(output_dir / "route_seasonality_summary.csv", index=False)
    recs.to_csv(output_dir / "route_month_recommendations.csv", index=False)
    rec_summary.to_csv(output_dir / "route_recommendation_summary.csv", index=False)

    plot_aircraft_regimes(regimes, output_dir)
    plot_top_volatility(route_summary, output_dir)
    plot_recommendation_counts(recs, output_dir)

    plot_candidate_bars(
        recs, output_dir,
        ["High-priority upgauge review", "Strong review for larger aircraft", "Review larger aircraft", "Efficient peak upgauge candidate"],
        "top_upgauge_candidates.png",
        "Top Upgauge Review Candidates",
    )

    plot_candidate_bars(
        recs, output_dir,
        ["Strong review for smaller aircraft", "Review smaller aircraft"],
        "top_downgauge_candidates.png",
        "Top Downgauge Review Candidates",
    )

    plot_top_route_profiles(route_monthly, route_summary, output_dir)

    print("\nCleaning report")
    print("=" * 80)
    print(cleaning_report.to_string(index=False))

    print("\nAircraft TOW regimes")
    print("=" * 80)
    print(regimes[["Aircraft", "Flights", "OperationalLowTOW", "MedianTOW", "OperationalHighTOW"]].to_string(index=False))

    print("\nTop seasonal routes")
    print("=" * 80)
    print(route_summary[[
        "Route", "TotalFlights", "PeakMonth", "PeakFlights",
        "TroughMonth", "TroughFlights", "PeakMinusTrough",
        "PeakMonthTowRegime", "VolatilityScore"
    ]].head(15).to_string(index=False))

    print_simple_recommendations(recs, top_n=args.print_top)

    print("\nSaved outputs to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
