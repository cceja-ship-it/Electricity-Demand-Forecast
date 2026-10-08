import pandas as pd
import matplotlib.pyplot as plt

# Two representative weeks from the 2017 validation year: one winter, one
# summer, so the plot shows both the heating and cooling demand regimes.
WINDOWS = [
    ("Winter week", "2017-01-09", "2017-01-15 23:00"),
    ("Summer week", "2017-07-17", "2017-07-23 23:00"),
]

MODEL_FILES = {
    "SARIMA": "results/sarima_val_sample.csv",
    "XGBoost": "results/xgboost_val_2017_full.csv",
    "LSTM": "results/lstm_val_sample.csv",
}


def load_forecasts() -> pd.DataFrame:
    frames = {}
    actual = None
    for name, path in MODEL_FILES.items():
        df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        frames[name] = df["forecast"]
        actual = df["actual"] if actual is None else actual
    out = pd.DataFrame(frames)
    out.insert(0, "Actual", actual)
    return out


if __name__ == "__main__":
    data = load_forecasts()

    fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(14, 8))
    for ax, (title, start, end) in zip(axes, WINDOWS):
        window = data.loc[start:end]
        ax.plot(window.index, window["Actual"], color="black", linewidth=2, label="Actual")
        for name in MODEL_FILES:
            ax.plot(window.index, window[name], linewidth=1.2, alpha=0.85, label=name)
        ax.set_title(f"{title} — 24h-ahead forecasts vs. actual (DAYTON, 2017)")
        ax.set_ylabel("Load (MW)")
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("results/figures/forecast_comparison.png", dpi=150)
    print("Saved results/figures/forecast_comparison.png")
