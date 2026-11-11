"""
LSTM acceleration prediction (Stage 1), built to match the PatchTST repo so the
two data-driven methods are trained under the same conditions.

Task: rows τ−10 … τ−1 (v1, delta_x, delta_v, a1) → acceleration of row τ.
92 test targets per vehicle (test rows 11–102), the same for all three models
and for every other method.

Training data (as in the PatchTST repo, src/data_utils.py):
  User-specific : windows over that vehicle's 248 training rows (238 windows)
  CDT Full      : windows over the 30 vehicles' training rows joined in vehicle
                  order (7,430 windows; 290 of them span two vehicles)
  CDT Limited   : windows over the 270 sampled rows (9 per vehicle, seed 12)
                  joined in vehicle order (260 windows)

No validation set. Every model trains for a fixed number of epochs
(settings.LSTM["epochs"], 100) with a constant learning rate and keeps its
final weights. The test files are used only once, at the end, for the scores.

Training: MSE loss, Adam (lr 1e-3, constant), gradient clipping 1.0, batch 32.
Inputs are standardised with the mean and SD of each model's own training rows.

Reported for every model:
  training MAE / MSE : |predicted − actual| over the windows it was trained on
                       (CDT values are the same in every vehicle's file)
  test MAE / MSE     : over this vehicle's 92 test targets

Usage:
    python lstm.py            # full Stage 1 run
    python lstm.py --quick    # 2 epochs, only to check the code runs
"""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import common as C
import settings as S

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (batch, 10, 4)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class Standardiser:
    def __init__(self, rows):
        X = rows[S.FEATURES].to_numpy(np.float32)
        self.mean = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8

    def __call__(self, Xw):
        return torch.tensor((Xw - self.mean) / self.sd, dtype=torch.float32)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(Xtr, ytr, cfg, seed):
    """Fixed number of epochs, constant learning rate, final weights kept."""
    seed_everything(seed)
    model = LSTMRegressor(len(S.FEATURES), cfg["hidden"], cfg["layers"], cfg["dropout"]).to(DEVICE)
    g = torch.Generator().manual_seed(seed)
    bs = max(1, min(cfg["batch_size"], len(ytr)))
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True, generator=g)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    losses = []
    for _ in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            running += loss.item() * len(Xb)
        losses.append(running / len(ytr))          # training MSE of this epoch
    model.eval()
    return model, losses


def predict(model, X):
    with torch.no_grad():
        return model(X.to(DEVICE)).cpu().numpy().astype(float)


def main():
    p = argparse.ArgumentParser(description="LSTM acceleration prediction (Stage 1)")
    p.add_argument("--quick", action="store_true", help="2 epochs, only to check the code runs")
    p.add_argument("--vehicles", type=int, nargs="+", default=None,
                   help="Subset of vehicles, for testing only (the experiment uses all 30)")
    args = p.parse_args()
    cfg = dict(S.LSTM)
    if args.quick:
        cfg["epochs"] = 2

    data = C.load_all(args.vehicles)
    sets = C.nn_training_sets(data)
    tests = {v: C.test_windows(test) for v, (_, test) in data.items()}
    for v, (_, test) in data.items():
        C.check_same_targets(tests[v]["target_idx"], C.test_points(test)["target_idx"])

    out_dir = S.OUTPUT_DIR / "LSTM"
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows, loss_rows, metric_rows = [], [], []

    def fit(label, train_set, seed):
        scaler = Standardiser(train_set["rows"])
        Xtr = scaler(train_set["X"])
        ytr = torch.tensor(train_set["y"], dtype=torch.float32)
        t0 = time.time()
        model, losses = train_model(Xtr, ytr, cfg, seed)
        train_metrics = C.mae_mse(train_set["y"].astype(float), predict(model, Xtr))
        log_rows.append(dict(model=label, n_train_windows=len(ytr),
                             n_windows_spanning_two_vehicles=int(train_set["mixed"].sum()),
                             epochs=len(losses), train_MAE=train_metrics[0], train_MSE=train_metrics[1],
                             seconds=time.time() - t0))
        loss_rows.extend(dict(model=label, epoch=e + 1, train_MSE=l) for e, l in enumerate(losses))
        print(f"  LSTM {label:>18}: {len(ytr):5d} windows, {len(losses)} epochs, "
              f"training MAE {train_metrics[0]:.4f}, MSE {train_metrics[1]:.4f}", flush=True)
        return model, scaler, train_metrics

    print("LSTM: training CDT models", flush=True)
    cdt_full = fit("CDT Full", sets["full"], S.SEED)
    cdt_lim = fit("CDT Limited", sets["limited"], S.SEED)

    print("LSTM: training User-specific models", flush=True)
    for v in sorted(tests):
        user = fit(f"User-specific V{v}", sets["user"][v], S.SEED)
        tw = tests[v]
        y_true = C.test_points(data[v][1])["y"]           # same targets, full precision
        models = {"User-specific": user, "CDT Full": cdt_full, "CDT Limited": cdt_lim}
        preds = {name: predict(m, scaler(tw["X"])) for name, (m, scaler, _) in models.items()}
        C.save_predictions(pred_dir, "LSTM", v, tw["target_idx"], y_true, preds)
        metric_rows += C.metrics_table("LSTM", v, y_true, preds,
                                       train={name: tm for name, (_, _, tm) in models.items()})

    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(out_dir / "training_loss.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
    print(f"LSTM: wrote {pred_dir}, metrics.csv, training_log.csv, training_loss.csv")


if __name__ == "__main__":
    main()
