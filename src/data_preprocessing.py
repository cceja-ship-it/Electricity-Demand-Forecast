from dataclasses import dataclass
import numpy as np
import pandas as pd


def build_clean_hourly_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindexes the series onto a complete, gap-free hourly DatetimeIndex.

    This fixes two DST-related issues in the raw data:
      - "Spring forward": I found that one hour per year is skipped entirely in the raw
        timestamps (it never occurred locally) due to time zone changes, which shows up as a gap.
      - "Fall back": I also found that one hour per year is duplicated (occurred twice
        locally). data_pipe.load_region() already drops exact duplicate index
        entries, so this function mainly needs to handle the spring-forward gaps.

    Missing hours are filled via linear interpolation, which works well for a smooth, cyclical series like electricity usage.
    """
    full_index = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq="h"
    )
    df = df.reindex(full_index)
    df.index.name = "datetime"

    n_missing = df["load_mw"].isna().sum()
    if n_missing > 0:
        print(f"Interpolating {n_missing} missing hourly readings "
              f"({n_missing / len(df):.3%} of series)")
        df["load_mw"] = df["load_mw"].interpolate(method="time")

    return df


def flag_outliers(df: pd.DataFrame, z_thresh: float = 5.0) -> pd.DataFrame:
    """
    Flags data points where usage deviates more than
    `z_thresh` standard deviations from a rolling local mean. This function
    manually inspects candidates before deciding whether to clip or keep
    them. Certain conditions like heat spikes or cold snaps can cause changes 
    in usage so they aren't just noise that should be blindly dropped.
    """
    rolling_mean = df["load_mw"].rolling(24 * 7, center=True, min_periods=24).mean()
    rolling_std = df["load_mw"].rolling(24 * 7, center=True, min_periods=24).std()
    z_score = (df["load_mw"] - rolling_mean) / rolling_std

    df = df.copy()
    df["outlier_flag"] = z_score.abs() > z_thresh
    return df


@dataclass
class SplitConfig:
    train_end: str   
    val_end: str     


def chronological_split(df: pd.DataFrame, config: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits strictly by time rather than shuffling or using random state. 
    This is purposeful because any random split would let the model train 
    on data that comes chronologically after points in the "test" set, 
    creating an unfair prediciton. 
    """
    train_end_ts = pd.Timestamp(config.train_end) + pd.Timedelta(hours=23)
    val_end_ts = pd.Timestamp(config.val_end) + pd.Timedelta(hours=23)

    train = df.loc[df.index <= train_end_ts]
    val = df.loc[(df.index > train_end_ts) & (df.index <= val_end_ts)]
    test = df.loc[df.index > val_end_ts]

    print(f"Train: {train.index.min()} -> {train.index.max()} ({len(train)} rows)")
    print(f"Val:   {val.index.min()} -> {val.index.max()} ({len(val)} rows)")
    print(f"Test:  {test.index.min()} -> {test.index.max()} ({len(test)} rows)")

    return train, val, test

if __name__ == "__main__":
    from data_pipe import load_region

    df = load_region("DAYTON")
    df = build_clean_hourly_index(df)
    df = flag_outliers(df)

    print(f"\n{df['outlier_flag'].sum()} points flagged as outliers "
          f"({df['outlier_flag'].mean():.3%})")

    config = SplitConfig(train_end="2016-12-31", val_end="2017-12-31")
    train, val, test = chronological_split(df, config)