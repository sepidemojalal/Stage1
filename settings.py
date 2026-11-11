"""
Stage 1 settings shared by every script.
Edit the paths and settings here; the scripts read everything from this file.
"""
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
TRAIN_DIR = Path("data/train")                 # folder with the training CSV files
TEST_DIR = Path("data/test")                   # folder with the test CSV files
TRAIN_FILE = "Copy of X_Train_V_id_{v}.csv"    # one file per vehicle (248 rows)
TEST_FILE = "X_Test_V_id_{v}.csv"              # one file per vehicle (102 rows)

OUTPUT_DIR = Path("outputs")                   # everything this code writes goes here

# Per-vehicle Transformer predictions written by the patched PatchTST repo
# (outputs/predictions inside the repo folder).
TRANSFORMER_PRED_DIR = Path("PatchTST-Vehicle-Acceleration/outputs/predictions")

# =============================================================================
# Data
# =============================================================================
VEHICLES = list(range(1, 31))
FEATURES = ["v1", "delta_x", "delta_v", "a1"]  # column names in your CSV files
TARGET = "a1"
TRAIN_ROWS = 248
TEST_ROWS = 102
DT = 0.1                                       # seconds per row

# delta_v in your data is (leader speed − follower speed): delta_x changes every
# 0.1 s by delta_v × 0.1. The data is never changed; each model formula applies it.

# =============================================================================
# Prediction set-up (matches the PatchTST repo and the Stage 1 figure)
# =============================================================================
REACTION_STEPS = 10   # Helly / IDM: T = 1 s → input row τ−10 predicts a(τ)
WINDOW = 10           # LSTM / PatchTST: rows τ−10 … τ−1 predict a(τ)
# Both give the same 92 test targets per vehicle: test rows 11–102.
N_TEST_TARGETS = TEST_ROWS - max(REACTION_STEPS, WINDOW)    # 92

# =============================================================================
# Model variants
# =============================================================================
VARIANTS = ["User-specific", "CDT Full", "CDT Limited"]
PRED_COLUMN = {                                 # column names in prediction files
    "CDT Full": "CDT_full_pred",
    "User-specific": "User_specific_pred",
    "CDT Limited": "CDT_limited_pred",
}

# CDT with limited data: the same sampling code as the PatchTST repo
# (src/data_utils.py → build_cdt_datasets): 9 random training rows per vehicle,
# numpy default_rng(12), vehicles in ascending order. All four methods use these rows.
LIMITED_N = 9
LIMITED_SEED = 12

SEED = 42                                       # master seed (MCMC, LSTM, bootstrap)

# =============================================================================
# MCMC (Metropolis–Hastings) for Helly and IDM
# =============================================================================
MCMC = dict(
    n_chains=4,          # independent chains, each started from a random prior draw
    n_epochs=3,          # each epoch uses the previous epoch's posterior as its prior
    n_burn=4000,         # burn-in iterations per chain per epoch (proposal tuning)
    n_samples=8000,      # kept iterations per chain per epoch
    target_accept=0.30,  # acceptance rate the random-walk proposal is tuned towards
    p_independence=0.2,  # share of moves drawn from the independence proposal
                         # (see helly_mcmc.py and idm_mcmc.py)
)

# Broad Gaussian priors: (mean, standard deviation). Edit freely.
# The likelihood is Gaussian with noise σ; log σ is sampled together with the
# model parameters and is not used for prediction.
# Helly SDs cover the range of the earlier calibrated values
# (C1 −11 … 1, C2 −0.6 … 2.3, α −5 … 138, β −9 … 4, γ −15 … 29).
HELLY_PRIORS = {
    "C1":    (0.5, 2.0),     # 1/s
    "C2":    (0.05, 1.0),    # 1/s²
    "alpha": (5.0, 50.0),    # m
    "beta":  (1.0, 10.0),    # s
    "gamma": (0.0, 20.0),    # s²
}
IDM_PRIORS = {               # truncated to positive values
    "a_max": (1.5, 1.0),     # m/s²
    "T_i":   (1.5, 1.0),     # s
    "b":     (2.0, 1.5),     # m/s²
}
LOG_SIGMA_PRIOR = (-0.7, 2.0)  # log of the noise SD (m/s²); exp(-0.7) ≈ 0.5

# How the calibrated models predict the test set:
#   "posterior_mean"       : the posterior means of the parameters (methods text)
#   "posterior_predictive" : the average prediction over posterior draws
MCMC_PREDICTION = "posterior_mean"

# IDM fixed values (from the methods text)
IDM_FIXED = dict(v0=32.0, s0=2.0, delta=4.0)

# The IDM formula's Δv_n is (follower − leader) = −delta_v in your data.
#   -1 : use −delta_v in the formula (IDM brakes when closing in)
#   +1 : use delta_v as it is
IDM_DELTA_V_SIGN = -1

# delta_x in the data looks like front-to-front spacing (≈ 6.9 m when stopped).
# IDM uses Δx − IDM_LEADER_LENGTH as the gap. 0 = use delta_x as it is (methods text).
IDM_LEADER_LENGTH = 0.0

# =============================================================================
# LSTM (same training settings as the PatchTST repo's config.py)
# No validation set: every model trains for a fixed number of epochs with a
# constant learning rate and keeps its final weights. Test data is used only
# for the final scores.
# =============================================================================
LSTM = dict(
    hidden=64,
    layers=1,
    dropout=0.0,           # only used when layers > 1
    batch_size=32,
    epochs=100,            # fixed number of training epochs
    lr=1e-3,               # Adam, constant
    grad_clip=1.0,
)

# =============================================================================
# BCa bootstrap (protocol: "BCa Bootstrap Confidence Intervals for the MAE")
# =============================================================================
BCA = dict(
    B=10_000,        # bootstrap trials per vehicle × method × variant
    alpha=0.025,     # 95% two-sided interval
)

# Methods read by fusion.py, bca.py and export_excel.py, with their folders.
BASE_METHODS = {
    "Helly": OUTPUT_DIR / "Helly" / "predictions",
    "IDM": OUTPUT_DIR / "IDM" / "predictions",
    "LSTM": OUTPUT_DIR / "LSTM" / "predictions",
    "Transformer": TRANSFORMER_PRED_DIR,
}
FUSION_METHODS = {
    "AverageFusion": OUTPUT_DIR / "AverageFusion" / "predictions",
    "MedianFusion": OUTPUT_DIR / "MedianFusion" / "predictions",
}

# Training and test MAE / MSE of the four base methods (read by export_excel.py).
# Columns: method, vehicle, variant, test_MAE, test_MSE, train_MAE, train_MSE.
METRICS_FILES = {
    "Helly": OUTPUT_DIR / "Helly" / "metrics.csv",
    "IDM": OUTPUT_DIR / "IDM" / "metrics.csv",
    "LSTM": OUTPUT_DIR / "LSTM" / "metrics.csv",
    "Transformer": Path("PatchTST-Vehicle-Acceleration/outputs/metrics.csv"),
}
