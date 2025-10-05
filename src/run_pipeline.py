#!/usr/bin/env python3
"""
SkyHack: end-to-end pipeline to compute a Flight Difficulty Score and answer EDA questions.

Run:
  python src/run_pipeline.py --data_dir data/raw --out_dir outputs --cache_dir data/interim

Expected files in data_dir:
  - flight_data.csv
  - pnr_data.csv
  - bag_data.csv
  - (optional) PNR Remark Level Data.csv
  - (optional) Airports Data.csv   [enrichment used by post_analysis.py]
"""
import argparse, os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------- helpers ----------
def _read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def _percent_rank(s: pd.Series) -> pd.Series:
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return s.rank(pct=True, method="average")

def _safe_num(s, default=0):
    return pd.to_numeric(s, errors="coerce").fillna(default)

# ---------- loading ----------
def load_data(data_dir: str):
    flights = _read_csv(os.path.join(data_dir, "flight_data.csv"))
    pnrs    = _read_csv(os.path.join(data_dir, "pnr_data.csv"))
    bags    = _read_csv(os.path.join(data_dir, "bag_data.csv"))
    # optional remarks
    rem_path = os.path.join(data_dir, "PNR Remark Level Data.csv")
    remarks = _read_csv(rem_path) if os.path.exists(rem_path) else None
    return flights, pnrs, bags, remarks

# ---------- cleaning ----------
def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    dt_cols = [
        "scheduled_departure_datetime_local",
        "scheduled_arrival_datetime_local",
        "actual_departure_datetime_local",
        "actual_arrival_datetime_local",
    ]
    for c in dt_cols:
        if c in df.columns:
            df[c] = _to_dt(df[c])

    if "scheduled_departure_date_local" in df.columns:
        df["scheduled_departure_date_local"] = _to_dt(df["scheduled_departure_date_local"]).dt.date

    for c in ["total_seats","scheduled_ground_time_minutes","actual_ground_time_minutes","minimum_turn_minutes"]:
        if c in df.columns:
            df[c] = _safe_num(df[c])

    if "scheduled_departure_datetime_local" in df.columns:
        df["sched_dep_hour"] = df["scheduled_departure_datetime_local"].dt.hour

    return df

# ---------- robust PNR aggregation ----------
def aggregate_pnr(pnr: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PNR file to flight-day. Tolerates different column names and Y/N flags.
    """
    def pick(*cands):
        for c in cands:
            if c in pnr.columns:
                return c
        return None

    # standardize date key
    if "scheduled_departure_date_local" in pnr.columns:
        pnr["scheduled_departure_date_local"] = pd.to_datetime(
            pnr["scheduled_departure_date_local"], errors="coerce"
        ).dt.date

    # join keys (min set)
    keys = ["company_id","flight_number","scheduled_departure_date_local",
            "scheduled_departure_station_code","scheduled_arrival_station_code"]
    missing = [k for k in keys if k not in pnr.columns]
    if missing:
        raise KeyError(f"PNR file missing join columns: {missing}")

    # aliases
    col_total_pax   = pick("total_pax","pax","passenger_count","pnr_pax_count","pax_count")
    col_lap_child   = pick("lap_child_count","lap_infant_count","lap_infants","lap_infant")
    col_basic_cnt   = pick("basic_economy_pax","basic_economy_count","basic_econ_pax")
    col_basic_flag  = pick("is_basic_economy","basic_economy_flag","is_basic_econ")
    col_child_flag  = pick("is_child","child_flag","is_child_flag","child_indicator")
    col_strol_flag  = pick("is_stroller_user","stroller","stroller_flag","stroller_indicator")
    col_record_loc  = pick("record_locator","pnr_id","pnr_number","pnr")

    # helpers
    def as_num(col):
        if col and col in pnr.columns:
            return _safe_num(pnr[col], default=0)
        return pd.Series(0, index=pnr.index, dtype="float64")

    def as_flag01(col):
        if col and col in pnr.columns:
            s = pnr[col].astype(str).str.strip().str.upper()
            return s.isin(["Y","YES","T","TRUE","1"]).astype(int)
        return pd.Series(0, index=pnr.index, dtype="int64")

    total = as_num(col_total_pax)
    if total.sum() == 0:
        total = pd.Series(1, index=pnr.index, dtype="int64")  # safe lower bound: 1 per PNR if unknown
    pnr["_total_pax"] = total

    pnr["_lap_child"]  = as_num(col_lap_child)

    basic = as_num(col_basic_cnt)
    if basic.sum() == 0:
        basic = as_flag01(col_basic_flag)
    pnr["_basic_econ"] = basic

    child = as_num(col_child_flag)
    if child.sum() == 0:
        child = as_flag01(col_child_flag)
    pnr["_child"] = child

    stroller = as_num(col_strol_flag)
    if stroller.sum() == 0:
        stroller = as_flag01(col_strol_flag)
    pnr["_stroller"] = stroller

    agg = pnr.groupby(keys, dropna=False).agg(
        total_pax=("._total_pax".lstrip("."), "sum"),
        lap_child_count=("._lap_child".lstrip("."), "sum"),
        basic_economy_pax=("._basic_econ".lstrip("."), "sum"),
        child_count=("._child".lstrip("."), "sum"),
        stroller_users=("._stroller".lstrip("."), "sum"),
        pnr_count=(col_record_loc, "nunique") if col_record_loc else ("scheduled_departure_date_local","count"),
    ).reset_index()

    return agg

# ---------- PNR Remark aggregation (SSR) ----------
def aggregate_remarks(pnrs: pd.DataFrame, remarks: pd.DataFrame) -> pd.DataFrame:
    """
    Map SSR codes from PNR Remark Level Data into per-flight-day counts.
    Returns columns: ssr_total, ssr_wheelchair, ssr_umnr, ssr_medical, ssr_blind, ssr_deaf, ssr_pet
    """
    expected = [
        "company_id","flight_number","scheduled_departure_date_local",
        "scheduled_departure_station_code","scheduled_arrival_station_code",
        "ssr_total","ssr_wheelchair","ssr_umnr","ssr_medical","ssr_blind","ssr_deaf","ssr_pet"
    ]
    if remarks is None or remarks.empty:
        return pd.DataFrame(columns=expected)

    # Build PNR base to inherit full keys, requires record_locator
    if "record_locator" not in pnrs.columns:
        return pd.DataFrame(columns=expected)

    base_keys = [
        "company_id","flight_number","scheduled_departure_date_local",
        "scheduled_departure_station_code","scheduled_arrival_station_code","record_locator"
    ]
    have = [k for k in base_keys if k in pnrs.columns]
    base = pnrs[have].drop_duplicates().copy()

    # Normalize dates for joining
    if "scheduled_departure_date_local" in remarks.columns:
        remarks["scheduled_departure_date_local"] = pd.to_datetime(
            remarks["scheduled_departure_date_local"], errors="coerce"
        ).dt.date
    if "scheduled_departure_date_local" in base.columns:
        base["scheduled_departure_date_local"] = pd.to_datetime(
            base["scheduled_departure_date_local"], errors="coerce"
        ).dt.date

    # Remarks text → flags
    col = "special_service_request"
    if col not in remarks.columns:
        return pd.DataFrame(columns=expected)
    rtxt = remarks[col].astype(str).str.upper()

    remarks = remarks.copy()
    remarks["_ssr_any"]   = (rtxt.str.len() > 0) & ~rtxt.isin(["", "NONE", "NAN"])
    remarks["_wchr"]      = rtxt.str.contains(r"\bWCH[CRS]\b", na=False)
    remarks["_umnr"]      = rtxt.str.contains(r"\bUMNR\b", na=False)
    remarks["_meda"]      = rtxt.str.contains(r"\b(?:MEDA|STCR|OXYG)\b", na=False)
    remarks["_blind"]     = rtxt.str.contains(r"\bBLND\b", na=False)
    remarks["_deaf"]      = rtxt.str.contains(r"\bDEAF\b", na=False)
    remarks["_pet"]       = rtxt.str.contains(r"\b(?:PETC|SVAN)\b", na=False)
    # Join on record_locator (+ flight/date if present both sides)
    join_cols = ["record_locator"]
    if "flight_number" in remarks.columns and "flight_number" in base.columns:
        join_cols.append("flight_number")
    if "scheduled_departure_date_local" in remarks.columns and "scheduled_departure_date_local" in base.columns:
        join_cols.append("scheduled_departure_date_local")

    rem_small = remarks[join_cols + ["_ssr_any","_wchr","_umnr","_meda","_blind","_deaf","_pet"]].copy()
    aligned = base.merge(rem_small, on=join_cols, how="left")

    # Fill zeros
    for c in ["_ssr_any","_wchr","_umnr","_meda","_blind","_deaf","_pet"]:
        if c in aligned.columns:
            # cast -> nullable boolean -> fill -> small int; avoids FutureWarning
            aligned[c] = aligned[c].astype("boolean").fillna(False).astype("int8")
        else:
            aligned[c] = np.int8(0)


    grp_keys = [
        "company_id","flight_number","scheduled_departure_date_local",
        "scheduled_departure_station_code","scheduled_arrival_station_code"
    ]
    grp_keys = [k for k in grp_keys if k in aligned.columns]

    agg = (aligned.groupby(grp_keys, dropna=False)
                  .agg(
                      ssr_total=("._ssr_any".lstrip("."), "sum"),
                      ssr_wheelchair=("_wchr","sum"),
                      ssr_umnr=("_umnr","sum"),
                      ssr_medical=("_meda","sum"),
                      ssr_blind=("_blind","sum"),
                      ssr_deaf=("_deaf","sum"),
                      ssr_pet=("_pet","sum"),
                  )
                  .reset_index())
    return agg

# ---------- robust bag aggregation ----------
def aggregate_bags(bags: pd.DataFrame) -> pd.DataFrame:
    if "scheduled_departure_date_local" in bags.columns:
        bags["scheduled_departure_date_local"] = pd.to_datetime(
            bags["scheduled_departure_date_local"], errors="coerce"
        ).dt.date

    keys = ["company_id","flight_number","scheduled_departure_date_local",
            "scheduled_departure_station_code","scheduled_arrival_station_code"]
    miss = [k for k in keys if k not in bags.columns]
    if miss:
        raise KeyError(f"Bags file missing join columns: {miss}")

    if "bag_type" not in bags.columns:
        raise KeyError("Bags file missing 'bag_type' column")

    bags = bags.copy()
    bags["_bag_type"] = bags["bag_type"].astype(str).str.strip().str.lower()

    # robust: match common variants
    is_transfer = bags["_bag_type"].str.contains(
        r"(?:transfer|xfer|xfr|conn|connection)", regex=True, na=False
    )
    bags["is_transfer"] = is_transfer.astype(int)

    tag_col = "bag_tag_unique_number" if "bag_tag_unique_number" in bags.columns else None

    agg = bags.groupby(keys, dropna=False).agg(
        total_bags=(tag_col, "nunique") if tag_col else ("_bag_type","count"),
        transfer_bags=("is_transfer","sum"),
    ).reset_index()
    return agg

# ---------- merge & features ----------
def merge_all(flights, pnr_agg, bag_agg):
    keys = ["company_id","flight_number","scheduled_departure_date_local",
            "scheduled_departure_station_code","scheduled_arrival_station_code"]
    df = flights.merge(pnr_agg, on=keys, how="left").merge(bag_agg, on=keys, how="left")
    for c in ["total_pax","lap_child_count","basic_economy_pax","child_count","stroller_users","pnr_count","total_bags","transfer_bags",
              "ssr_total","ssr_wheelchair","ssr_umnr","ssr_medical","ssr_blind","ssr_deaf","ssr_pet"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # turnaround slack (negated so higher => more difficult)
    if {"scheduled_ground_time_minutes","minimum_turn_minutes"}.issubset(df.columns):
        df["slack_minutes"] = df["scheduled_ground_time_minutes"] - df["minimum_turn_minutes"]
        df["neg_slack_minutes"] = -df["slack_minutes"]
    else:
        df["neg_slack_minutes"] = np.nan

    # load factor (pure pandas: where + fillna)
    if "total_seats" in df.columns:
        denom = df["total_seats"].replace({0: np.nan})
        lf = (df.get("total_pax", 0) / denom).where(denom.gt(0), 0.0)
        df["pax_load_factor"] = pd.to_numeric(lf, errors="coerce").fillna(0.0).clip(lower=0, upper=1)
    else:
        df["pax_load_factor"] = 0.0

    # transfer bag ratio (pure pandas; no np.where)
    if {"transfer_bags","total_bags"}.issubset(df.columns):
        denom_b = df["total_bags"].replace({0: np.nan})
        tbr = (df["transfer_bags"] / denom_b).where(denom_b.gt(0), 0.0)
        df["transfer_bag_ratio"] = pd.to_numeric(tbr, errors="coerce").fillna(0.0).clip(lower=0, upper=1)
    else:
        df["transfer_bag_ratio"] = 0.0

    # special needs ratio (now includes SSR counts if present)
    df["special_needs_numer"] = (
        df.get("child_count", 0)
        + df.get("lap_child_count", 0)
        + df.get("stroller_users", 0)
        + df.get("ssr_total", 0)
    )
    pax = df.get("total_pax", 0).replace({0: np.nan})
    snr = (df["special_needs_numer"] / pax).where(pax.gt(0), 0.0)
    df["special_needs_ratio"] = pd.to_numeric(snr, errors="coerce").fillna(0.0).clip(lower=0, upper=1)

    # Optional granular SSR ratios for EDA/insights
    for raw_col, ratio_col in [
        ("ssr_wheelchair","wheelchair_ratio"),
        ("ssr_umnr","umnr_ratio"),
        ("ssr_medical","medical_ratio"),
        ("ssr_blind","blind_ratio"),
        ("ssr_deaf","deaf_ratio"),
        ("ssr_pet","pet_ratio"),
    ]:
        if raw_col in df.columns:
            r = (df[raw_col] / pax).where(pax.gt(0), 0.0)
            df[ratio_col] = pd.to_numeric(r, errors="coerce").fillna(0.0).clip(lower=0, upper=1)
        else:
            df[ratio_col] = 0.0

    # realized delays (for EDA/validation only)
    if {"scheduled_departure_datetime_local","actual_departure_datetime_local"}.issubset(df.columns):
        df["departure_delay_minutes"] = (
            df["actual_departure_datetime_local"] - df["scheduled_departure_datetime_local"]
        ).dt.total_seconds()/60.0
    if {"scheduled_arrival_datetime_local","actual_arrival_datetime_local"}.issubset(df.columns):
        df["arrival_delay_minutes"] = (
            df["actual_arrival_datetime_local"] - df["scheduled_arrival_datetime_local"]
        ).dt.total_seconds()/60.0

    # day key
    if "scheduled_departure_date_local" in df.columns:
        df["day"] = pd.to_datetime(df["scheduled_departure_date_local"]).dt.date
    else:
        df["day"] = pd.to_datetime(df["scheduled_departure_datetime_local"]).dt.date

    return df

# ---------- scoring ----------
def daily_difficulty(df: pd.DataFrame, weights: dict, qtiles=(0.25,0.75)) -> pd.DataFrame:
    feats = list(weights.keys())
    for f in feats:
        if f in df.columns:
            df[f+"_pr"] = df.groupby("day")[f].transform(_percent_rank)
        else:
            df[f+"_pr"] = 0.0

    df["difficulty_score"] = 0.0
    for f,w in weights.items():
        df["difficulty_score"] += w * df[f+"_pr"]

    df["daily_rank"] = df.groupby("day")["difficulty_score"].rank(ascending=False, method="first")

    lo, hi = qtiles
    def classify(s):
        if s.notna().sum() == 0:
            return pd.Series(["Medium"]*len(s), index=s.index)
        q1, q3 = s.quantile(lo), s.quantile(hi)
        out = pd.Series(index=s.index, dtype="object")
        out[s <= q1] = "Easy"
        out[(s > q1) & (s < q3)] = "Medium"
        out[s >= q3] = "Difficult"
        return out

    df["difficulty_class"] = df.groupby("day")["difficulty_score"].transform(classify)
    return df

# ---------- EDA ----------
def eda_questions(df: pd.DataFrame, cfg: dict) -> dict:
    out = {}

    if "departure_delay_minutes" in df.columns:
        dep = pd.to_numeric(df["departure_delay_minutes"], errors="coerce").dropna()
        out["avg_departure_delay_minutes"] = float(dep.mean()) if len(dep) else None
        thr = cfg.get("late_departure_threshold_minutes", 0)
        out["pct_departures_late"] = float((dep > thr).mean()*100) if len(dep) else None

    if {"scheduled_ground_time_minutes","minimum_turn_minutes"}.issubset(df.columns):
        margin = cfg.get("tight_turn_margin_minutes", 10)
        cond = df["scheduled_ground_time_minutes"] <= (df["minimum_turn_minutes"] + margin)
        out["n_flights_tight_turn"] = int(cond.sum())
        out["pct_flights_tight_turn"] = float(cond.mean()*100)

    if {"transfer_bags","total_bags"}.issubset(df.columns):
        ratio = np.where(df["total_bags"]>0, df["transfer_bags"]/df["total_bags"], np.nan)
        out["avg_transfer_bag_ratio"] = float(pd.Series(ratio).mean(skipna=True))

    if "pax_load_factor" in df.columns:
        out["avg_load_factor"] = float(pd.to_numeric(df["pax_load_factor"], errors="coerce").mean(skipna=True))
        if "departure_delay_minutes" in df.columns:
            x = df["pax_load_factor"].values.reshape(-1,1)
            y = df["departure_delay_minutes"].values
            mask = ~np.isnan(x).ravel() & ~np.isnan(y)
            if mask.sum() > 5:
                lr = LinearRegression().fit(x[mask], y[mask])
                out["load_vs_delay_slope"] = float(lr.coef_[0])
            else:
                out["load_vs_delay_slope"] = None

    if {"departure_delay_minutes","special_needs_ratio","pax_load_factor"}.issubset(df.columns):
        X = df[["special_needs_ratio","pax_load_factor"]].values
        y = df["departure_delay_minutes"].values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        if mask.sum() > 5:
            lr = LinearRegression().fit(X[mask], y[mask])
            out["delay_coeff_special_needs"] = float(lr.coef_[0])
            out["delay_coeff_load_factor"] = float(lr.coef_[1])
        else:
            out["delay_coeff_special_needs"] = None
            out["delay_coeff_load_factor"] = None

    # Optional: summarize SSR ratios if present
    for col in ["wheelchair_ratio","umnr_ratio","medical_ratio","blind_ratio","deaf_ratio","pet_ratio"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            out[col] = {
                "count": int(s.notna().sum()),
                "mean": float(s.mean(skipna=True)),
                "std": float(s.std(skipna=True)),
                "min": float(s.min(skipna=True)),
                "max": float(s.max(skipna=True)),
                "p50": float(s.quantile(0.5)),
                "p90": float(s.quantile(0.9)),
            }

    return out

# ---------- plots ----------
def plot_hist(series, title, path):
    plt.figure()
    s = pd.Series(series).dropna()
    if len(s)==0:
        plt.close(); return
    plt.hist(s, bins=30)
    plt.title(title); plt.xlabel(title); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_scatter(x, y, title, xlabel, ylabel, path):
    plt.figure()
    xs, ys = pd.Series(x), pd.Series(y)
    mask = ~xs.isna() & ~ys.isna()
    if mask.sum()==0:
        plt.close(); return
    plt.scatter(xs[mask], ys[mask], s=10)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(path); plt.close()

# ---------- main ----------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))

    flights, pnrs, bags, remarks = load_data(args.data_dir)
    flights = clean_flights(flights)
    pnr_agg = aggregate_pnr(pnrs)
    bag_agg = aggregate_bags(bags)

    # Aggregate remarks to flight-day and merge into PNR aggregate
    if remarks is not None and not remarks.empty:
        remarks_agg = aggregate_remarks(pnrs, remarks)
        if remarks_agg is not None and not remarks_agg.empty:
            keys_full = [
                "company_id","flight_number","scheduled_departure_date_local",
                "scheduled_departure_station_code","scheduled_arrival_station_code"
            ]
            join_keys = [k for k in keys_full if k in pnr_agg.columns and k in remarks_agg.columns]
            pnr_agg = pnr_agg.merge(remarks_agg, on=join_keys, how="left")

    merged = merge_all(flights, pnr_agg, bag_agg)
    merged = engineer_features(merged)

    # cache (pre-scoring)
    merged.to_csv(os.path.join(args.cache_dir, "master_merged.csv"), index=False)

    # score (daily reset via percent ranks)
    merged = daily_difficulty(
        merged.copy(),
        weights=cfg["difficulty_weights"],
        qtiles=tuple(cfg["classification_quantiles"])
    )

    # save main outputs
    merged.to_csv(os.path.join(args.out_dir, "difficulty_scores.csv"), index=False)
    merged.to_csv(os.path.join(args.out_dir, "master_merged.csv"), index=False)

    # EDA answers
    eda = eda_questions(merged, cfg)
    with open(os.path.join(args.out_dir, "eda_summary.json"), "w") as f:
        json.dump(eda, f, indent=2)

    # figures
    if "departure_delay_minutes" in merged.columns:
        plot_hist(merged["departure_delay_minutes"], "Departure Delay (min)",
                  os.path.join(args.out_dir,"figures","hist_departure_delay.png"))
    if {"scheduled_ground_time_minutes","minimum_turn_minutes"}.issubset(merged.columns):
        plot_hist(merged["scheduled_ground_time_minutes"] - merged["minimum_turn_minutes"], "Ground Slack (min)",
                  os.path.join(args.out_dir,"figures","hist_slack.png"))
    if {"pax_load_factor","departure_delay_minutes"}.issubset(merged.columns):
        plot_scatter(merged["pax_load_factor"], merged["departure_delay_minutes"],
                     "Load vs Departure Delay", "Load Factor","Departure Delay (min)",
                     os.path.join(args.out_dir,"figures","scatter_load_vs_delay.png"))
    if {"transfer_bag_ratio","difficulty_score"}.issubset(merged.columns):
        plot_scatter(merged["transfer_bag_ratio"], merged["difficulty_score"],
                     "Transfer Bag Ratio vs Difficulty", "Transfer Bag Ratio","Difficulty Score",
                     os.path.join(args.out_dir,"figures","scatter_transfer_vs_difficulty.png"))

    # top difficult per day
    top = (merged.sort_values(["day","difficulty_score"], ascending=[True,False])
                 .groupby("day").head(15))
    top.to_csv(os.path.join(args.out_dir, "daily_top_difficult.csv"), index=False)

    print("✅ Done. Outputs written to:", args.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    args = ap.parse_args()
    main(args)