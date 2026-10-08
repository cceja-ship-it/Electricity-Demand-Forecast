import numpy as np
import pandas as pd
 
# US federal holidays roughly correlate with atypical demand (offices/schools
# closed). Using a small hardcoded list keeps this dependency-free; swap for
# the `holidays` package if you want full accuracy across all years.
try:
    import holidays as holidays_lib
    US_HOLIDAYS = holidays_lib.US()
except ImportError:
    US_HOLIDAYS = set()  # falls back to no holiday feature if package missing
 
 
def build_features(series: pd.Series) -> pd.DataFrame:
    """
    Builds a feature DataFrame indexed by forecast-origin timestamp `t`.
    Every column here is knowable at time `t` — no lookahead.
    """
    df = pd.DataFrame(index=series.index)
    df["load_mw"] = series  # kept for reference; not used as a feature itself
 
    # --- Lag features: past values relative to origin `t` ---
    for lag in [1, 2, 3, 24, 48, 168]:  # 1-3h, 1-2 days, 1 week
        df[f"lag_{lag}h"] = series.shift(lag)
 
    # --- Rolling statistics, computed using only data up to and including `t` ---
    # shift(1) before rolling ensures the window doesn't include `t` itself
    # when you want a "prior N hours" stat — here we allow t itself since it's
    # available at forecast time, but shift the window start to avoid any
    # accidental future leakage from how rolling() aligns windows.
    df["roll_mean_24h"] = series.rolling(24).mean()
    df["roll_std_24h"] = series.rolling(24).std()
    df["roll_mean_168h"] = series.rolling(168).mean()
 
    # --- Calendar features (always safe — known in advance, no leakage risk) ---
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["is_holiday"] = pd.Series(df.index, index=df.index).apply(
        lambda ts: 1 if ts in US_HOLIDAYS else 0
    )
 
    # Cyclical encoding for hour/month so the model understands 23:00 is
    # close to 00:00 (a raw integer would treat them as far apart)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
 
    return df
 
 
def build_long_format_frame(
    series: pd.Series, horizons: list[int] = list(range(1, 25))
) -> pd.DataFrame:
    """
    Builds a single 'long format' training frame: one row per
    (origin_timestamp, horizon) pair, with a 'horizon' feature column and a
    single 'target' column. This lets one XGBoost model handle all 24
    horizons, rather than training 24 separate models.
    """
    features = build_features(series)
    feature_cols = [c for c in features.columns if c != "load_mw"]
 
    long_rows = []
    for h in horizons:
        chunk = features[feature_cols].copy()
        chunk["horizon"] = h
        chunk["target"] = series.shift(-h)
        chunk["origin_timestamp"] = chunk.index
        long_rows.append(chunk)
 
    long_df = pd.concat(long_rows, axis=0).dropna()
    return long_df.reset_index(drop=True)
 
 
if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from data_pipe import load_region
    from data_preprocessing import build_clean_hourly_index
 
    df = load_region("DAYTON")
    df = build_clean_hourly_index(df)
 
    long_df = build_long_format_frame(df["load_mw"])
    print(long_df.head())
    print(f"\nShape: {long_df.shape}")
    print(f"Columns: {list(long_df.columns)}")