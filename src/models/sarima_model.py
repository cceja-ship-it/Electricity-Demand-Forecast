from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
 
 
@dataclass
class SarimaConfig:
    order: tuple = (1, 0, 1)
    seasonal_order: tuple = (1, 1, 1, 24)
    train_window_days: int = 730
    refit_every_days: int = 14
    horizon_hours: int = 24
 
 
def fit_sarima(train_series: pd.Series, config: SarimaConfig) -> SARIMAX:
    """Fits a fresh SARIMAX model on the given series."""
    model = SARIMAX(
        train_series,
        order=config.order,
        seasonal_order=config.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)
 
 
def walk_forward_forecast(full_series: pd.Series, val_start: pd.Timestamp, val_end: pd.Timestamp, config: SarimaConfig) -> pd.DataFrame:
    """
    Walks-forward evaluation one day at a time during the validation section, forecasting
    24 hours ahead each time. Refits the model every `refit_every_days` days. On days in between, .append() is used to fold in new observations without a
    full refit to save time.
 
    Returns a DataFrame with columns: actual, forecast, forecast_origin.
    """
    results = []
    current_day = val_start
    fitted_model = None
    days_since_refit = 0
 
    while current_day + pd.Timedelta(hours=config.horizon_hours) <= val_end:
        train_cutoff = current_day - pd.Timedelta(hours=1)
        window_start = train_cutoff - pd.Timedelta(days=config.train_window_days)
        history = full_series.loc[window_start:train_cutoff]
 
        needs_refit = fitted_model is None or days_since_refit >= config.refit_every_days
        if needs_refit:
            print(f"Refitting SARIMA as of {train_cutoff}...")
            fitted_model = fit_sarima(history, config)
            days_since_refit = 0
        else:
            # Fold in the new day's observations without a full refit
            new_obs = full_series.loc[
                current_day - pd.Timedelta(days=1) : train_cutoff
            ]
            fitted_model = fitted_model.append(new_obs, refit=False)
 
        forecast = fitted_model.forecast(steps=config.horizon_hours)
        actual = full_series.loc[
            current_day : current_day + pd.Timedelta(hours=config.horizon_hours - 1)
        ]
 
        for ts, pred, act in zip(forecast.index, forecast.values, actual.values):
            results.append({
                "timestamp": ts,
                "forecast_origin": current_day,
                "actual": act,
                "forecast": pred,
            })
 
        current_day += pd.Timedelta(days=1)
        days_since_refit += 1
 
    return pd.DataFrame(results).set_index("timestamp")
 
def naive_seasonal_baseline(full_series: pd.Series, val_start: pd.Timestamp, val_end: pd.Timestamp, horizon_hours: int = 24, lag_hours: int = 168,) -> pd.DataFrame:
    """
    Naive baseline: predict each hour using the value from `lag_hours` ago
    (default 168 = same hour, same day, one week prior). This is a much
    fairer baseline than "yesterday" for a series with weekly seasonality —
    a model has to beat this, not just beat a coin flip, to be worth using.
    """
    results = []
    current_day = val_start
 
    while current_day + pd.Timedelta(hours=horizon_hours) <= val_end:
        actual = full_series.loc[
            current_day : current_day + pd.Timedelta(hours=horizon_hours - 1)
        ]
        forecast_index = actual.index
        forecast_values = full_series.loc[
            forecast_index - pd.Timedelta(hours=lag_hours)
        ].values
 
        for ts, pred, act in zip(forecast_index, forecast_values, actual.values):
            results.append({
                "timestamp": ts,
                "actual": act,
                "forecast": pred,
            })
 
        current_day += pd.Timedelta(days=1)
 
    return pd.DataFrame(results).set_index("timestamp")

def compute_metrics(results: pd.DataFrame) -> tuple[float, float]:
    rmse = np.sqrt(((results["actual"] - results["forecast"]) ** 2).mean())
    mae = (results["actual"] - results["forecast"]).abs().mean()
    return rmse, mae

if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from data_pipe import load_region
    from data_preprocessing import build_clean_hourly_index
 
    df = load_region("DAYTON")
    df = build_clean_hourly_index(df)
    series = df["load_mw"]
 
    VAL_START = pd.Timestamp("2017-01-01")
    VAL_END = pd.Timestamp("2018-01-01")
 
    config = SarimaConfig()
 
    sarima_results = walk_forward_forecast(series, VAL_START, VAL_END, config)
    sarima_rmse, sarima_mae = compute_metrics(sarima_results)
 
    baseline_results = naive_seasonal_baseline(series, VAL_START, VAL_END)
    baseline_rmse, baseline_mae = compute_metrics(baseline_results)
 
    print("\n--- Results (full 2017 validation year) ---")
    print(f"{'Model':<20}{'RMSE':>10}{'MAE':>10}")
    print(f"{'Naive (t-168h)':<20}{baseline_rmse:>10.2f}{baseline_mae:>10.2f}")
    print(f"{'SARIMA':<20}{sarima_rmse:>10.2f}{sarima_mae:>10.2f}")
 
    improvement = (1 - sarima_mae / baseline_mae) * 100
    print(f"\nSARIMA improves MAE over naive baseline by {improvement:.1f}%")
 
    #Actual vs. Both
    plot_slice = sarima_results.loc[:VAL_START + pd.Timedelta(days=30)]
    baseline_plot_slice = baseline_results.loc[:VAL_START + pd.Timedelta(days=30)]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_slice.index, plot_slice["actual"], label="Actual", color="black", linewidth=1.5)
    ax.plot(plot_slice.index, plot_slice["forecast"], label="SARIMA forecast", alpha=0.8)
    ax.plot(baseline_plot_slice.index, baseline_plot_slice["forecast"], label="Naive (t-168h) forecast", alpha=0.6, linestyle="--")
    ax.set_title("24h-ahead forecasts vs. actual — DAYTON load, Jan 2017")
    ax.set_ylabel("Load (MW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/sarima_vs_baseline.png", dpi=150)
    plt.show()
 
    sarima_results.to_csv("results/sarima_val_sample.csv")