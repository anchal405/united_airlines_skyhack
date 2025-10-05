#!/usr/bin/env python3
"""
Post-analysis for SkyHack:
- Destination difficulty summaries
- Driver coefficients (overall & per-destination)
- Plots & an insights markdown report

Run:
  python src/post_analysis.py --merged outputs/master_merged.csv --config src/config.json --out_dir outputs
"""

import argparse, os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def _ensure_dir(p): 
    """Create directory if it doesn't exist"""
    os.makedirs(p, exist_ok=True)

def load(merged_path, config_path):
    """
    Load the merged dataset and configuration file.
    
    Args:
        merged_path: Path to master_merged.csv containing flight data
        config_path: Path to config.json with difficulty weights
    
    Returns:
        df: DataFrame with all flight data
        features: List of feature names from config
        cfg: Full configuration dictionary
    """
    df = pd.read_csv(merged_path)
    with open(config_path, "r") as f:
        cfg = json.load(f)
    features = list(cfg.get("difficulty_weights", {}).keys())
    return df, features, cfg

def station_summaries(df, out_dir):
    """
    Generate aggregate and daily summaries for each destination station.
    
    Creates two CSV files:
    - station_summary.csv: Overall stats per destination
    - station_daily_summary.csv: Daily stats per destination
    
    Args:
        df: Flight data DataFrame
        out_dir: Output directory for CSV files
    
    Returns:
        summ: Overall summary DataFrame
        daily: Daily summary DataFrame
    """
    # Use scheduled arrival station as the destination identifier
    dest_col = "scheduled_arrival_station_code"
    if dest_col not in df.columns:
        raise KeyError(f"Expected column '{dest_col}' not found in master_merged.csv")

    # Calculate per-destination metrics across entire time period
    grp = df.groupby(dest_col, dropna=False)
    summ = pd.DataFrame({
        "flights": grp["difficulty_score"].size(),  # Total number of flights
        "avg_difficulty": grp["difficulty_score"].mean(),  # Average difficulty score
        "pct_difficult": grp["difficulty_class"].apply(lambda s: (s=="Difficult").mean()*100),  # % classified as Difficult
        "avg_neg_slack": grp.get_group if False else grp["neg_slack_minutes"].mean(numeric_only=True),  # Average negative slack time
        "avg_transfer_bag_ratio": grp["transfer_bag_ratio"].mean(),  # Average transfer bag ratio
        "avg_special_needs_ratio": grp["special_needs_ratio"].mean(),  # Average special needs ratio
        "avg_pax_load_factor": grp["pax_load_factor"].mean(),  # Average passenger load factor
    }).reset_index().sort_values("avg_difficulty", ascending=False)

    # Save overall station summary
    summ.to_csv(os.path.join(out_dir, "station_summary.csv"), index=False)

    # Calculate per-day, per-destination metrics
    if "day" not in df.columns:
        # Extract date from scheduled departure timestamp
        df["day"] = pd.to_datetime(df["scheduled_departure_date_local"]).dt.date
    g2 = df.groupby(["day", dest_col], dropna=False)
    daily = pd.DataFrame({
        "flights": g2["difficulty_score"].size(),
        "avg_difficulty": g2["difficulty_score"].mean(),
        "pct_difficult": g2["difficulty_class"].apply(lambda s: (s=="Difficult").mean()*100),
    }).reset_index().sort_values(["day","avg_difficulty"], ascending=[True, False])
    
    # Save daily station summary
    daily.to_csv(os.path.join(out_dir, "station_daily_summary.csv"), index=False)

    return summ, daily

def geographical_eda(df, out_dir):
    """
    Add geographical context by merging with airport location data.
    Produces country-level summary of average difficulty.
    
    Args:
        df: Flight data DataFrame
        out_dir: Output directory for results
    
    Returns:
        geo_summary: DataFrame with country-level statistics
    """
    # Check if airport reference data is available
    airports_path = os.path.join("data", "raw", "Airports Data.csv")
    if not os.path.exists(airports_path):
        print("⚠️ Airports Data.csv not found — skipping geographical EDA.")
        return pd.DataFrame()

    # Load airport reference data (IATA codes, countries, etc.)
    airports = pd.read_csv(airports_path)
    airports = airports.rename(columns=str.strip)

    # Validate required column exists in flight data
    if "scheduled_arrival_station_code" not in df.columns:
        print("⚠️ scheduled_arrival_station_code column missing — cannot join airports.")
        return pd.DataFrame()

    # Merge flight data with airport locations to get country info
    df_geo = df.merge(
        airports,
        left_on="scheduled_arrival_station_code",
        right_on="airport_iata_code",
        how="left"
    )

    # Compute average difficulty metrics per country
    country_grp = df_geo.groupby("iso_country_code", dropna=False)
    geo_summary = pd.DataFrame({
        "flights": country_grp["difficulty_score"].size(),
        "avg_difficulty": country_grp["difficulty_score"].mean(),
        "pct_difficult": country_grp["difficulty_class"].apply(lambda s: (s=="Difficult").mean()*100)
    }).reset_index().sort_values("avg_difficulty", ascending=False)

    # Save country-level summary
    geo_path = os.path.join(out_dir, "country_summary.csv")
    geo_summary.to_csv(geo_path, index=False)

    # Create bar chart of top 10 countries by difficulty
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    top10 = geo_summary.head(10)
    plt.bar(top10["iso_country_code"].astype(str), top10["avg_difficulty"], color="steelblue")
    plt.title("Average Flight Difficulty by Country")
    plt.xlabel("Country Code")
    plt.ylabel("Average Difficulty Score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "avg_difficulty_by_country.png"))
    plt.close()

    print(f"🌍 Geographical EDA done → {geo_path}")
    return geo_summary

def overall_driver_coeffs(df, features, out_dir):
    """
    Fit a linear regression model to identify which features most strongly
    predict flight difficulty across all destinations.
    
    Uses standardized features for fair coefficient comparison.
    
    Args:
        df: Flight data DataFrame
        features: List of feature column names
        out_dir: Output directory
    
    Returns:
        coefs: DataFrame with feature coefficients sorted by impact
    """
    # Filter to features that exist in the dataset
    avail = [f for f in features if f in df.columns]
    X = df[avail].copy()
    y = df["difficulty_score"].copy()
    
    # Remove rows with missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]; y = y[mask]
    
    # Check if we have enough data for regression
    if len(X) < 20:
        coefs = pd.DataFrame({"feature": avail, "coef": np.nan})
    else:
        # Standardize features (mean=0, std=1) for comparable coefficients
        Xz = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
        Xz = Xz.fillna(0.0)
        
        # Fit linear regression model
        lr = LinearRegression().fit(Xz.values, y.values)
        
        # Extract coefficients and sort by magnitude
        coefs = pd.DataFrame({"feature": avail, "coef": lr.coef_}).sort_values("coef", ascending=False)
    
    # Save coefficients to CSV
    coefs.to_csv(os.path.join(out_dir, "driver_coefficients.csv"), index=False)
    return coefs

def per_destination_driver_coeffs(df, features, out_dir, top_n=5, min_flights=20):
    """
    Fit separate linear regression models for each of the top-N most difficult
    destinations to identify destination-specific difficulty drivers.
    
    Args:
        df: Flight data DataFrame
        features: List of feature column names
        out_dir: Output directory
        top_n: Number of top destinations to analyze
        min_flights: Minimum flights required for a destination to be included
    
    Returns:
        out: DataFrame with per-destination feature coefficients
    """
    dest_col = "scheduled_arrival_station_code"
    
    # Identify top N destinations by average difficulty
    top_dest = (
        df.groupby(dest_col)["difficulty_score"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    rows = []
    # Fit separate model for each top destination
    for dest in top_dest:
        # Filter to flights for this destination only
        sub = df[df[dest_col] == dest]
        avail = [f for f in features if f in sub.columns]
        X = sub[avail].copy()
        y = sub["difficulty_score"].copy()
        
        # Remove rows with missing values
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        # Skip if insufficient data for this destination
        if len(X) < min_flights:
            continue

        # Standardize features for this destination
        Xz = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
        Xz = Xz.fillna(0.0)

        # Fit regression model for this destination
        lr = LinearRegression().fit(Xz.values, y.values)

        # Store coefficients with destination identifier
        for f, c in zip(avail, lr.coef_):
            rows.append({
                "destination": dest,
                "feature": f,
                "coef": float(c),
                "n": int(len(X))
            })

    # Handle case where no destinations have enough data
    if not rows:
        print("⚠️  No destinations had enough flights for per-destination regression.")
        out = pd.DataFrame(columns=["destination","feature","coef","n"])
    else:
        out = pd.DataFrame(rows)
        # Sort by destination and coefficient magnitude
        if "destination" in out.columns:
            out = out.sort_values(["destination","coef"], ascending=[True, False])
        else:
            print("⚠️  'destination' column missing in per-destination results.")

    # Save per-destination coefficients
    out.to_csv(os.path.join(out_dir, "destination_driver_coeffs.csv"), index=False)
    return out

def make_plots(summ, out_dir):
    """
    Create visualization of top 10 destinations by average difficulty.
    
    Args:
        summ: Station summary DataFrame
        out_dir: Output directory for plot
    """
    # Ensure figures directory exists
    _ensure_dir(os.path.join(out_dir, "figures"))
    
    # Create bar chart of top 10 most difficult destinations
    top10 = summ.head(10)
    plt.figure()
    plt.bar(top10["scheduled_arrival_station_code"].astype(str), top10["avg_difficulty"])
    plt.title("Top 10 Destinations by Avg Difficulty")
    plt.xlabel("Destination")
    plt.ylabel("Avg Difficulty Score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "bar_top10_destinations.png"))
    plt.close()

def write_markdown(summ, coefs, per_dest, out_dir):
    """
    Generate a comprehensive markdown report with operational insights.
    
    Includes:
    - Most difficult destinations
    - Overall difficulty drivers
    - Destination-specific drivers
    - Geographical insights
    - Operational recommendations
    
    Args:
        summ: Station summary DataFrame
        coefs: Overall driver coefficients DataFrame
        per_dest: Per-destination coefficients DataFrame
        out_dir: Output directory
    
    Returns:
        md_path: Path to generated markdown report
    """
    lines = []
    lines.append("# Operational Insights\n")

    # Section 1: Most difficult destinations
    lines.append("## Destinations that trend difficult\n")
    top5 = summ.head(5)[["scheduled_arrival_station_code","flights","avg_difficulty","pct_difficult"]]
    for _, r in top5.iterrows():
        lines.append(
            f"- {r['scheduled_arrival_station_code']}: avg difficulty {r['avg_difficulty']:.2f}, "
            f"{r['pct_difficult']:.1f}% flights classified as Difficult (n={int(r['flights'])})."
        )

    # Section 2: Overall difficulty drivers across all destinations
    lines.append("\n## Common drivers (overall model)\n")
    if coefs.dropna().empty:
        lines.append("- Not enough data for regression. Showing correlations would be an alternative.")
    else:
        for _, r in coefs.sort_values('coef', ascending=False).iterrows():
            lines.append(f"- {r['feature']}: coefficient {r['coef']:.3f} (standardized)")

    # Section 3: Destination-specific drivers for hardest destinations
    if per_dest is not None and not per_dest.empty:
        lines.append("\n## Drivers for the hardest destinations\n")
        for dest in per_dest["destination"].unique():
            # Get top 3 drivers for this destination
            sub = per_dest[per_dest["destination"] == dest].sort_values("coef", ascending=False).head(3)
            desc = "; ".join([f"{row['feature']} ({row['coef']:.3f})" for _, row in sub.iterrows()])
            n = int(sub["n"].iloc[0]) if len(sub) else 0
            lines.append(f"- {dest}: top drivers → {desc} [n={n}]")

    # Section 4: Geographical insights (country-level patterns)
    geo_path = os.path.join(out_dir, "country_summary.csv")
    if os.path.exists(geo_path):
        try:
            geo = pd.read_csv(geo_path)
            if not geo.empty:
                lines.append("\n## Geographical Insights\n")
                top5_geo = geo.head(5)[["iso_country_code", "avg_difficulty", "pct_difficult"]]
                for _, r in top5_geo.iterrows():
                    lines.append(
                        f"- {r['iso_country_code']}: avg difficulty {r['avg_difficulty']:.2f}, "
                        f"{r['pct_difficult']:.1f}% flights Difficult."
                    )
        except Exception as e:
            lines.append(f"\n*(Geographical insights unavailable: {e})*\n")

    # Section 5: Actionable operational recommendations
    lines.append("\n## Operational recommendations\n")
    lines.append("- **Tight turns / negative slack** → Add buffer minutes to schedule on specific turns; pre-stage turn teams; coordinate fueling/catering earlier.")
    lines.append("- **High transfer bag ratio** → Pre-position bag transfer staff; tighten minimum connect times; priority belt allocation for connections.")
    lines.append("- **High passenger load factor** → Add an extra gate agent at boarding; enforce boarding groups; open a second document check line.")
    lines.append("- **High special needs ratio** → Pre-assign wheelchair handlers and aisle chairs; brief crew; allow earlier boarding for SSR passengers.")
    lines.append("- **High basic-economy mix** → Prepare for boarding exceptions (seating/baggage); proactive signage; pre-brief gate agents on exception handling.")

    # Write all sections to markdown file
    md_path = os.path.join(out_dir, "insights_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    return md_path

def main():
    """
    Main execution function.
    
    Orchestrates the complete post-analysis pipeline:
    1. Load data and configuration
    2. Generate station summaries (overall and daily)
    3. Calculate overall difficulty drivers
    4. Calculate per-destination drivers
    5. Create visualizations
    6. Perform geographical analysis
    7. Generate insights report
    """
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="outputs/master_merged.csv")
    ap.add_argument("--config", required=True, help="src/config.json")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    # Ensure output directory exists
    _ensure_dir(args.out_dir)
    
    # Load input data and configuration
    df, features, cfg = load(args.merged, args.config)

    # Run all analysis components
    summ, daily = station_summaries(df, args.out_dir)
    coefs = overall_driver_coeffs(df, features, args.out_dir)
    per_dest = per_destination_driver_coeffs(df, features, args.out_dir, top_n=5, min_flights=3)
    make_plots(summ, args.out_dir)
    geo_summary = geographical_eda(df, args.out_dir)
    md_path = write_markdown(summ, coefs, per_dest, args.out_dir)

    # Print summary of generated outputs
    print("✅ Post-analysis complete.")
    print("  - station_summary.csv")
    print("  - station_daily_summary.csv")
    print("  - driver_coefficients.csv")
    print("  - destination_driver_coeffs.csv")
    print("  - figures/bar_top10_destinations.png")
    print(f"  - insights_report.md  → {md_path}")

if __name__ == "__main__":
    main()
