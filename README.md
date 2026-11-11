# Stage 1 — Helly, IDM, LSTM, Fusion and BCa

Code for the Stage 1 comparison of User-specific, CDT with full data and CDT
with limited data models. The Transformer (PatchTST) comes from the patched
`PatchTST-Vehicle-Acceleration` repository; this code reads its predictions.

## Files

| File | What it does |
|---|---|
| `settings.py` | All paths and settings (data folders, priors, MCMC, LSTM, BCa, IDM options) |
| `common.py` | Loads the data, builds the training sets and test targets, reads and writes prediction files |
| `mcmc.py` | Metropolis–Hastings calibration shared by Helly and IDM |
| `helly_mcmc.py` | Helly model, MCMC calibration, test predictions |
| `idm_mcmc.py` | IDM, MCMC calibration of (a_max, T_i, b), test predictions |
| `lstm.py` | LSTM, trained like the PatchTST repo, test predictions |
| `prepare_patchtst_data.py` | Writes the single CSV the PatchTST repo reads |
| `fusion.py` | Average and Median fusion of Helly, IDM, LSTM and Transformer |
| `bca.py` | BCa 95 % confidence interval of the MAE (your protocol), all methods |
| `export_excel.py` | One Excel workbook per vehicle: estimated accelerations, MAE, MSE, BCa CI |

## Folder layout

```
stage1_code/
  data/train/Copy of X_Train_V_id_1.csv … _30.csv   (vehicle 7's file included)
  data/test/X_Test_V_id_1.csv … _30.csv
  PatchTST-Vehicle-Acceleration/                     (the patched repo)
  *.py
```

Change `TRAIN_DIR`, `TEST_DIR` or the file names in `settings.py` if yours differ.

## Run order

```bash
pip install numpy pandas scipy torch openpyxl transformers matplotlib

python prepare_patchtst_data.py
cd PatchTST-Vehicle-Acceleration
python main.py --data data/ngsim_30_vehicles.csv
cd ..
python helly_mcmc.py
python idm_mcmc.py
python lstm.py
python fusion.py
python bca.py
python export_excel.py
```

`python bca.py --selftest` reproduces Example 1 of the BCa protocol.
`--quick` on `helly_mcmc.py`, `idm_mcmc.py` and `lstm.py` uses tiny settings,
only to check that the code runs.

## Outputs (in `outputs/`)

| Folder / file | Content |
|---|---|
| `<Method>/predictions/V{N}_<Method>_Predictions.csv` | idx, a1_true, CDT_full_pred, User_specific_pred, CDT_limited_pred (92 rows) |
| `<Method>/metrics.csv` | Test MAE and MSE per vehicle and variant, plus training MAE and MSE for Helly, IDM and LSTM (fusion: test only) |
| `Helly/`, `IDM/` `calibrated_parameters.csv` | Posterior mean, SD and R-hat of every parameter, acceptance rate, training MAE and MSE, for all 32 models |
| `Helly/`, `IDM/` `mcmc_epochs.csv` | The same for every epoch |
| `LSTM/training_log.csv` | Training windows, epochs, training MAE and MSE per model |
| `LSTM/training_loss.csv` | Training MSE of every epoch, per model |
| `BCa/<Method>/V{N}_<Method>_MAE_bootstrap.csv` | The 10,000 bootstrap MAEs per variant |
| `BCa/<Method>/<Method>_BCa_summary.xlsx` | MAE, bootstrap mean and SD, ẑ0, â, α1, α2, 95 % CI |
| `BCa/bca_all_methods.csv` | All of the above in one table |
| `Excel/Stage1_Vehicle_XX.xlsx` | Per vehicle: all six methods, three variants, test MAE and MSE, BCa CI, training MAE and MSE |

## Set-up

**Targets.** 92 per vehicle, test rows 11–102, the same for every method.

**Inputs.** Helly and IDM use row τ−10 to predict a(τ), so T = 1 s. LSTM and
PatchTST use rows τ−10 … τ−1 to predict a(τ), the next row.

**Training data.**
- User-specific: the vehicle's 248 training rows.
- CDT Full: all 30 vehicles' training rows.
- CDT Limited: 9 random training rows per vehicle (numpy `default_rng(12)`,
  the PatchTST repo's sampling code), shared by all four methods.
- Helly and IDM build input/target pairs inside one vehicle's trajectory. For
  CDT Limited, each sampled row is the input and the row 1 s later is the
  target. When that target would fall in the test period, the pair ending at
  the sampled row is used instead, so no test data is used and the set keeps
  270 pairs.
- LSTM and PatchTST build windows over the rows joined in vehicle order, as in
  the PatchTST repo. For CDT Full, 290 of 7,430 windows span two vehicles. For
  CDT Limited, the 260 windows run over the 270 sampled rows.

**No validation.** Every model learns from its training data only. The test
files are used once, at the end, to score each model; all three models are
scored on the same 92 test rows of each vehicle. LSTM and PatchTST train for
100 epochs with a constant learning rate (0.001) and keep their final weights.

**Training and test results.** Test MAE and MSE are the reported results.
Training MAE and MSE (|predicted − actual| over the data each model was trained
on) are reported alongside: User-specific on that vehicle's training data,
CDT Full on all 30 vehicles', CDT Limited on its 270 sampled rows, so the CDT
values are the same in every vehicle's file. A low training error with a much
higher test error points to overfitting.

**MCMC.**
- 4 chains from random prior draws, 3 epochs (each uses the previous posterior
  as its prior), 4,000 burn-in + 8,000 kept iterations per chain and epoch.
- The calibrated parameters are the posterior means of the last epoch
  (`MCMC_PREDICTION = "posterior_mean"`; `"posterior_predictive"` averages the
  predictions over posterior draws instead).
- Check `max_rhat` (below 1.05) and `acceptance` in `calibrated_parameters.csv`.
  `train_rmse_posterior_mean` and `train_rmse_posterior_predictive` should be
  close; a large gap means the posterior means do not summarise the posterior well.
- Using the posterior as the next prior with the same data counts the data more
  than once: posterior means are hardly affected, but posterior SDs shrink.

**Helly sampler.** α, β and γ only enter the equation multiplied by C2. In
(C1, C2, α, β, γ) the posterior is a curved ridge, and plain random-walk chains
do not converge (R-hat up to 24 in testing). The sampler therefore moves in the
equation's linear coefficients (C1, C2, −C2α, −C2β, −C2γ), with the Jacobian
included. 20 % of its moves are drawn from a heavy-tailed proposal centred on
the least-squares fit, so chains cannot get stuck near C2 = 0. Both choices
change only how the chains move; the target is still the posterior of
(C1, C2, α, β, γ) under the priors in `settings.py`, and everything is
reported in those parameters.

**IDM sampler.** b only enters through √(a_max·b), so the posterior can have
separate modes (one test vehicle reached R-hat 38). 20 % of the moves are drawn
from a heavy-tailed proposal centred on the prior-regularised least-squares fit
of IDM, so chains can move between modes. The target posterior is unchanged.

**Convergence in testing.** On synthetic data of the same size (30 vehicles,
248 + 102 rows), all 32 Helly and all 32 IDM calibrations reached R-hat ≤ 1.02.

**IDM.**
- δ = 4, v0 = 32 m/s, s0 = 2 m are fixed.
- The dynamic term of s* uses v_n(t−T).
- The formula's Δv_n (follower − leader) is −delta_v in the data
  (`IDM_DELTA_V_SIGN = -1`). Set it to +1 to use delta_v as it is.
- `IDM_LEADER_LENGTH` (default 0) is subtracted from delta_x if you want IDM to
  use a net gap.

**BCa.**
- Your protocol, steps 1–8.
- For each vehicle the same 10,000 resampled row sets are used for every method
  and variant, so the intervals are directly comparable.
- Rows 0.1 s apart have correlated errors, so resampling single rows treats
  them as independent; the intervals can be somewhat narrower than with a
  block bootstrap.

**Runtime** (measured on a 2-core CPU with synthetic data of the same size):
Helly about 5 minutes, IDM about 4.5 minutes, LSTM about 2 minutes (32 models ×
100 epochs), fusion under a second, BCa about 10 seconds, Excel export a few
seconds. PatchTST takes longer (also 100 epochs per model).
