#!/usr/bin/env python3
"""
analyze_results.py
Quick insight summary for United Airlines SkyHack Flight Difficulty System.
Reads outputs/master_merged.csv and prints key operational insights.
"""

import pandas as pd
import numpy as np

# ---------- load data ----------
df = pd.read_csv("outputs/master_merged.csv")

print("\n-------------------------------")
print("   SKYHACK RESULTS SUMMARY")
print("-------------------------------")

# 1 Tight turn flights (<10 min slack)
if {"scheduled_ground_time_minutes", "minimum_turn_minutes"}.issubset(df.columns):
    tight_turns = (
        (df["scheduled_ground_time_minutes"] <= df["minimum_turn_minutes"] + 10).mean()
        * 100
    )
    print(f" Tight-turn flights (<10min slack): {tight_turns:.2f}%")
else:
    print(" Ground time data missing")

# 2 Transfer bag effect
if {"transfer_bag_ratio", "departure_delay_minutes"}.issubset(df.columns):
    df["is_high_transfer"] = df["transfer_bag_ratio"] > 0.3
    mean_delay_high = df.loc[df["is_high_transfer"], "departure_delay_minutes"].mean()
    mean_delay_low = df.loc[~df["is_high_transfer"], "departure_delay_minutes"].mean()
    if pd.notna(mean_delay_high) and pd.notna(mean_delay_low):
        diff = mean_delay_high - mean_delay_low
        print(f"🧳 Delay impact (transfer>30%): +{diff:.1f} min avg")
else:
    print(" Bag data missing")

# 3 Special service request impact
if {"ssr_total", "departure_delay_minutes"}.issubset(df.columns):
    df["has_high_ssr"] = df["ssr_total"] > 10
    mean_delay_ssr = df.loc[df["has_high_ssr"], "departure_delay_minutes"].mean()
    mean_delay_non = df.loc[~df["has_high_ssr"], "departure_delay_minutes"].mean()
    if pd.notna(mean_delay_ssr) and pd.notna(mean_delay_non):
        diff = mean_delay_ssr - mean_delay_non
        print(f" Delay impact (SSR>10): +{diff:.1f} min avg")
else:
    print(" SSR remark data missing")

# 4 Load factor
if "pax_load_factor" in df.columns:
    print(f"👥 Average load factor: {df['pax_load_factor'].mean():.2f}")
else:
    print(" Passenger load factor missing")

# 5 Top difficult destinations
if {"scheduled_arrival_station_code", "difficulty_score"}.issubset(df.columns):
    top_dest = (
        df.groupby("scheduled_arrival_station_code")["difficulty_score"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    print("\n Top 5 Difficult Destinations:")
    print(top_dest)
else:
    print(" Destination data missing")

print("-------------------------------")
print(" Analysis complete.\n")
