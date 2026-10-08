import sys
sys.path.append("src")
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
 
LOOKBACK_HOURS = 168
HORIZON_HOURS = 24
 
 
class SequenceDataset(Dataset):
    """
    Slides a window over the series: each sample is (past 168 hours,
    next 24 hours). Operates on already-normalized values (plain numpy
    array), so no leakage risk lives in this class — that's handled
    upstream by fitting the scaler on train data only.
    """
 
    def __init__(self, values: np.ndarray, lookback: int = LOOKBACK_HOURS,
                 horizon: int = HORIZON_HOURS):
        self.values = values
        self.lookback = lookback
        self.horizon = horizon
 
    def __len__(self):
        return len(self.values) - self.lookback - self.horizon + 1
 
    def __getitem__(self, idx):
        x = self.values[idx: idx + self.lookback]
        y = self.values[idx + self.lookback: idx + self.lookback + self.horizon]
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(-1),  # (lookback, 1)
            torch.tensor(y, dtype=torch.float32),                 # (horizon,)
        )
 
 
class LSTMForecaster(nn.Module):
    """
    A straightforward stacked LSTM: reads the 168-hour input sequence,
    takes the final hidden state, and maps it through a linear layer to
    24 output values (one per forecast hour) in a single shot.
    """
 
    def __init__(self, hidden_size: int = 64, num_layers: int = 2,
                 horizon: int = HORIZON_HOURS, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)
 
    def forward(self, x):
        # x: (batch, lookback, 1)
        _, (h_n, _) = self.lstm(x)
        last_layer_hidden = h_n[-1]  # (batch, hidden_size) — final layer's hidden state
        return self.fc(last_layer_hidden)  # (batch, horizon)
 
 
class Scaler:
    """Simple standardization, fit only on training data to avoid leakage."""
 
    def __init__(self):
        self.mean_ = None
        self.std_ = None
 
    def fit(self, values: np.ndarray):
        self.mean_ = values.mean()
        self.std_ = values.std()
        return self
 
    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.std_
 
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std_ + self.mean_
 
 
def train_lstm(
    train_values: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> LSTMForecaster:
    dataset = SequenceDataset(train_values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
    model = LSTMForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
 
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
 
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
 
            epoch_loss += loss.item() * x_batch.size(0)
 
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} — MSE loss (normalized scale): {epoch_loss:.4f}")
 
    return model
 
 
def walk_forward_forecast_lstm(
    full_series: pd.Series,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    model: LSTMForecaster,
    scaler: Scaler,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Evaluates the (already-trained) model across the validation window,
    forecasting 24 hours ahead once per day using the prior 168 hours of
    actual observed data as input. No refitting during this loop — see
    README ("Model-specific setup") for why that is a deliberate choice.
    """
    model.eval()
    results = []
    current_day = val_start
 
    with torch.no_grad():
        while current_day + pd.Timedelta(hours=HORIZON_HOURS) <= val_end:
            window_start = current_day - pd.Timedelta(hours=LOOKBACK_HOURS)
            window_end = current_day - pd.Timedelta(hours=1)
            history = full_series.loc[window_start:window_end].values
 
            if len(history) != LOOKBACK_HOURS:
                current_day += pd.Timedelta(days=1)
                continue
 
            history_norm = scaler.transform(history)
            x = torch.tensor(history_norm, dtype=torch.float32).view(1, LOOKBACK_HOURS, 1).to(device)
 
            pred_norm = model(x).cpu().numpy().flatten()
            pred = scaler.inverse_transform(pred_norm)
 
            actual = full_series.loc[
                current_day: current_day + pd.Timedelta(hours=HORIZON_HOURS - 1)
            ]
 
            for ts, p, a in zip(actual.index, pred, actual.values):
                results.append({"timestamp": ts, "actual": a, "forecast": p})
 
            current_day += pd.Timedelta(days=1)
 
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
 
    TRAIN_END = pd.Timestamp("2016-12-31 23:00:00")
    VAL_START = pd.Timestamp("2017-01-01")
    VAL_END = pd.Timestamp("2018-01-01")
 
    train_values = series.loc[:TRAIN_END].values
 
    scaler = Scaler().fit(train_values)
    train_values_norm = scaler.transform(train_values)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
 
    model = train_lstm(train_values_norm, epochs=15, device=device)
 
    results = walk_forward_forecast_lstm(
        series, TRAIN_END, VAL_START, VAL_END, model, scaler, device=device
    )
    rmse, mae = compute_metrics(results)
 
    print(f"\n--- LSTM Results (full 2017 validation year) ---")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE:  {mae:.2f} MW")
 
    results.to_csv("results/lstm_val_sample.csv")