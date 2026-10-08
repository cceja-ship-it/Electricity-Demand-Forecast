import sys
sys.path.append("src")
 
import numpy as np
import pandas as pd
import xgboost as xgb
 
from features import build_long_format_frame
 
 
def train_xgboost(train_long: pd.DataFrame, feature_cols: list[str]) -> xgb.XGBRegressor:
    """Trains one XGBoost model on the long-format (origin, horizon) frame."""
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(train_long[feature_cols], train_long["target"])
    return model
 
 
def walk_forward_forecast_xgb(
    full_series: pd.Series,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    refit_every_days: int = 30,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """
    Same walk-forward structure as the SARIMA script, for a fair comparison:
    forecasts 24 hours ahead once per day, refitting periodically. XGBoost
    trains much faster than SARIMA, so refitting more often is cheap here —
    default is every 30 days, but feel free to tighten this.
    """
    long_df_full = build_long_format_frame(full_series)
    feature_cols = [
        c for c in long_df_full.columns
        if c not in ("target", "origin_timestamp")
    ]
 
    results = []
    current_day = val_start
    model = None
    days_since_refit = 0
 
    while current_day + pd.Timedelta(hours=horizon_hours) <= val_end:
        needs_refit = model is None or days_since_refit >= refit_every_days
        if needs_refit:
            print(f"Refitting XGBoost as of {current_day}...")
            # Filter on the *target* time (origin + horizon), not just the
            # origin: a row with origin 23:00 yesterday and horizon 24 has a
            # target inside the day we're about to forecast.
            target_ts = long_df_full["origin_timestamp"] + pd.to_timedelta(
                long_df_full["horizon"], unit="h"
            )
            train_slice = long_df_full[target_ts < current_day]
            model = train_xgboost(train_slice, feature_cols)
            days_since_refit = 0
 
        # Build the 24 prediction rows for this origin day: one per horizon,
        # all using features as of `current_day - 1 hour` (last known point).
        origin_ts = current_day - pd.Timedelta(hours=1)
        origin_row = long_df_full[long_df_full["origin_timestamp"] == origin_ts]
 
        if origin_row.empty:
            # origin timestamp fell out of the feature frame (e.g. NaN warmup) — skip
            current_day += pd.Timedelta(days=1)
            days_since_refit += 1
            continue
 
        # origin_row currently has 24 rows (one per horizon, all with same
        # origin_timestamp) — that's exactly what we want to predict at once.
        preds = model.predict(origin_row[feature_cols])
        actual = full_series.loc[
            current_day : current_day + pd.Timedelta(hours=horizon_hours - 1)
        ]
 
        for h, pred, ts, act in zip(
            origin_row["horizon"].values, preds, actual.index, actual.values
        ):
            results.append({
                "timestamp": ts,
                "forecast_origin": current_day,
                "horizon": h,
                "actual": act,
                "forecast": pred,
            })
 
        current_day += pd.Timedelta(days=1)
        days_since_refit += 1
 
    return pd.DataFrame(results).set_index("timestamp")
 
 
def compute_metrics(results: pd.DataFrame) -> tuple[float, float]:
    rmse = np.sqrt(((results["actual"] - results["forecast"]) ** 2).mean())
    mae = (results["actual"] - results["forecast"]).abs().mean()
    return rmse, mae
 
 
if __name__ == "__main__":
    from data_pipe import load_region
    from data_preprocessing import build_clean_hourly_index
 
    df = load_region("DAYTON")
    df = build_clean_hourly_index(df)
    series = df["load_mw"]
 
    VAL_START = pd.Timestamp("2017-01-01")
    VAL_END = pd.Timestamp("2018-01-01")
 
    results = walk_forward_forecast_xgb(series, VAL_START, VAL_END)
    rmse, mae = compute_metrics(results)
 
    print(f"\n--- XGBoost Results (full 2017 validation year) ---")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE:  {mae:.2f} MW")

    # --- Per-horizon breakdown ---
    # Reveals whether error is uniform across the 24-hour window or grows
    # with horizon length — a key diagnostic for multi-step forecasting.
    per_horizon = results.groupby("horizon").apply(
        lambda g: pd.Series({
            "rmse": np.sqrt(((g["actual"] - g["forecast"]) ** 2).mean()),
            "mae": (g["actual"] - g["forecast"]).abs().mean(),
        })
    )
    print("\n--- MAE/RMSE by forecast horizon ---")
    print(per_horizon.to_string())
 
    per_horizon.to_csv("results/xgboost_per_horizon.csv")
    results.to_csv("results/xgboost_val_2017_full.csv")
