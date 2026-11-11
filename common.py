"""
Data loading, training-set construction and prediction-file helpers shared by
every Stage 1 script (Helly, IDM, LSTM, fusion, BCa, Excel export).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import settings as S

COLUMNS = ["idx", "id"] + S.FEATURES


# =============================================================================
# Loading
# =============================================================================
def load_vehicle(v):
    """Return (train, test) DataFrames for vehicle v, sorted by idx and checked."""
    train_path = S.TRAIN_DIR / S.TRAIN_FILE.format(v=v)
    test_path = S.TEST_DIR / S.TEST_FILE.format(v=v)
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing data file: {p}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    out = []
    for name, df, n_rows in (("train", train, S.TRAIN_ROWS), ("test", test, S.TEST_ROWS)):
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Vehicle {v} {name} file is missing columns {sorted(missing)}")
        df = df[COLUMNS].sort_values("idx").reset_index(drop=True)
        if len(df) != n_rows:
            raise ValueError(f"Vehicle {v} {name}: expected {n_rows} rows, found {len(df)}")
        if df[S.FEATURES].isna().any().any():
            raise ValueError(f"Vehicle {v} {name}: missing values in {S.FEATURES}")
        if not (np.diff(df["idx"].to_numpy()) == 1).all():
            raise ValueError(f"Vehicle {v} {name}: idx values are not consecutive")
        if not (df["id"] == v).all():
            raise ValueError(f"Vehicle {v} {name}: the id column does not equal {v}")
        out.append(df)
    train, test = out
    if test["idx"].iloc[0] != train["idx"].iloc[-1] + 1:
        raise ValueError(f"Vehicle {v}: the test rows do not follow the training rows")
    return train, test


def load_all(vehicles=None):
    """{vehicle: (train, test)} for every vehicle."""
    vehicles = S.VEHICLES if vehicles is None else vehicles
    return {v: load_vehicle(v) for v in vehicles}


# =============================================================================
# CDT with limited data: rows sampled exactly as in the PatchTST repo
# =============================================================================
def limited_positions(train):
    """
    Positions (0-based, within each vehicle's training rows) of the 9 sampled rows.
    Same code as the PatchTST repo's build_cdt_datasets: one default_rng(12),
    vehicles in ascending order, rng.choice(len(train), 9, replace=False), sorted.
    """
    rng = np.random.default_rng(S.LIMITED_SEED)
    positions = {}
    for v in sorted(train):
        n = len(train[v])
        positions[v] = np.sort(rng.choice(n, size=min(S.LIMITED_N, n), replace=False))
    return positions


# =============================================================================
# Helly / IDM samples: input row t  →  a at row t + 10 (same vehicle)
# =============================================================================
def point_pairs(df, starts=None, steps=S.REACTION_STEPS):
    X = df[S.FEATURES].to_numpy(float)
    a = df[S.TARGET].to_numpy(float)
    idx = df["idx"].to_numpy()
    if starts is None:
        starts = np.arange(len(df) - steps)
    starts = np.asarray(starts)
    return dict(X=X[starts], y=a[starts + steps],
                input_idx=idx[starts], target_idx=idx[starts + steps])


def _concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def physics_training_sets(data):
    """
    Training pairs for Helly and IDM, always inside one vehicle's trajectory:
      User-specific : that vehicle's 248 training rows        → 238 pairs
      CDT Full      : all 30 vehicles' pairs together          → 7,140 pairs
      CDT Limited   : one pair per sampled row                 → 270 pairs
        Each sampled row t is the input and a(t+10) the target. When t+10 would
        fall in the test period (sampled rows in the last second of training),
        the pair ending at the sampled row is used instead (input t−10, target t),
        so no test data is used and the set keeps 270 pairs.
    """
    train = {v: tr for v, (tr, _) in data.items()}
    steps = S.REACTION_STEPS
    user = {v: point_pairs(train[v]) for v in sorted(train)}
    full = _concat([user[v] for v in sorted(train)])

    positions = limited_positions(train)
    parts = []
    for v in sorted(train):
        n = len(train[v])
        p = positions[v]
        starts = np.where(p + steps <= n - 1, p, p - steps)
        parts.append(point_pairs(train[v], starts=starts))
    limited = _concat(parts)
    return dict(user=user, full=full, limited=limited)


# =============================================================================
# LSTM / PatchTST samples: rows i … i+9  →  a at row i+10
# =============================================================================
def windows(rows, L=S.WINDOW):
    """
    Sliding windows over the rows exactly in the order given (as in the PatchTST
    repo's AccelerationDataset). 'mixed' marks windows whose 11 rows come from
    more than one vehicle (only possible when vehicles' rows are joined together).
    """
    X = rows[S.FEATURES].to_numpy(np.float32)
    a = rows[S.TARGET].to_numpy(np.float32)
    idx = rows["idx"].to_numpy()
    vid = rows["id"].to_numpy()
    n = len(rows) - L
    if n <= 0:
        raise ValueError("Not enough rows for one window")
    Xw = np.lib.stride_tricks.sliding_window_view(X, (L, X.shape[1]))[:n, 0].copy()
    span = np.lib.stride_tricks.sliding_window_view(vid, L + 1)
    mixed = (span != span[:, :1]).any(axis=1)
    return dict(X=Xw, y=a[L:].copy(), target_idx=idx[L:], mixed=mixed, rows=rows)


def nn_training_sets(data):
    """
    Training windows for the LSTM, built exactly like the PatchTST repo:
      User-specific : windows over that vehicle's 248 training rows   → 238
      CDT Full      : windows over the 30 vehicles' training rows joined
                      in vehicle order (290 windows span two vehicles) → 7,430
      CDT Limited   : windows over the 270 sampled rows joined in vehicle
                      order (as designed in the repo)                  → 260
    """
    train = {v: tr for v, (tr, _) in data.items()}
    user = {v: windows(train[v]) for v in sorted(train)}
    full = windows(pd.concat([train[v] for v in sorted(train)], ignore_index=True))
    positions = limited_positions(train)
    limited_rows = pd.concat([train[v].iloc[positions[v]] for v in sorted(train)],
                             ignore_index=True)
    limited = windows(limited_rows)
    return dict(user=user, full=full, limited=limited)


# =============================================================================
# Test samples (92 per vehicle, identical targets for every method)
# =============================================================================
def test_points(test):
    return point_pairs(test)


def test_windows(test):
    return windows(test)


def check_same_targets(a, b):
    if not np.array_equal(a, b):
        raise AssertionError("Methods do not predict the same target rows")


# =============================================================================
# Prediction files (same layout as the BCa prediction files)
# =============================================================================
def pred_path(folder, method, v):
    return Path(folder) / f"V{v}_{method}_Predictions.csv"


def save_predictions(folder, method, v, target_idx, y_true, preds):
    """preds: {variant: array}. Writes idx, a1_true and one column per variant."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"idx": np.asarray(target_idx), "a1_true": np.asarray(y_true, float)})
    for variant, col in S.PRED_COLUMN.items():
        df[col] = np.asarray(preds[variant], float)
    df.to_csv(pred_path(folder, method, v), index=False)
    return df


def read_predictions(folder, method, v):
    path = pred_path(folder, method, v)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    needed = ["idx", "a1_true"] + list(S.PRED_COLUMN.values())
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if len(df) != S.N_TEST_TARGETS:
        raise ValueError(f"{path}: expected {S.N_TEST_TARGETS} rows, found {len(df)}")
    return df[needed]


def mae_mse(y_true, y_pred):
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.mean(np.abs(e))), float(np.mean(e ** 2))


def metrics_table(method, v, y_true, preds, train=None):
    """
    One row per variant: test MAE / MSE on this vehicle's 92 test targets and,
    when given, the training MAE / MSE.
    train: {variant: (train_MAE, train_MSE)}, measured on the data that model was
    trained on (User-specific: this vehicle; CDT Full: all 30 vehicles; CDT
    Limited: the 270 sampled rows), so the CDT values repeat for every vehicle.
    """
    rows = []
    for variant in S.VARIANTS:
        mae, mse = mae_mse(y_true, preds[variant])
        tr = train[variant] if train is not None else (np.nan, np.nan)
        rows.append(dict(method=method, vehicle=v, variant=variant,
                         test_MAE=mae, test_MSE=mse, train_MAE=tr[0], train_MSE=tr[1]))
    return rows
