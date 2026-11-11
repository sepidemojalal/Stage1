"""
BCa bootstrap confidence intervals for the MAE of predicted acceleration.

Implements the protocol "BCa Bootstrap Confidence Intervals for the Mean
Absolute Error of Predicted Vehicle Acceleration" step by step
(Efron & Tibshirani, 1993, Chapters 6, 11, 13, 14):

  Step 1  e_i = |ŷ_i − y_i|, i = 1 … n (n = 92)
  Step 2  θ̂ = mean(e)                                   observed MAE
  Step 3  B = 10,000 resamples of n rows with replacement → θ̂*(b)
  Step 4  ẑ0 = Φ⁻¹( #{θ̂*(b) < θ̂} / B )                bias correction
  Step 5  jackknife θ̂(i) = (Σe − e_i)/(n − 1), θ̂(·) = mean
  Step 6  â = Σ(θ̂(·) − θ̂(i))³ / (6 [Σ(θ̂(·) − θ̂(i))²]^(3/2))   acceleration
  Step 7  α1 = Φ(ẑ0 + (ẑ0 + z^(α)) / (1 − â(ẑ0 + z^(α))))
          α2 = Φ(ẑ0 + (ẑ0 + z^(1−α)) / (1 − â(ẑ0 + z^(1−α))))
  Step 8  CI = [θ̂*^(α1), θ̂*^(α2)], percentiles of the sorted θ̂* with linear
          interpolation at position p·(B − 1)

For each vehicle, the same 10,000 sets of resampled rows are used for every
method and variant, so all intervals for that vehicle are directly comparable.

Usage:
    python bca.py              # all methods found in settings.py
    python bca.py --selftest   # reproduce Example 1 of the protocol
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

import common as C
import settings as S


# =============================================================================
# The protocol
# =============================================================================
def resample_indices(n, B, rng):
    """Step 3.1: B × n row numbers drawn at random with replacement."""
    return rng.integers(0, n, size=(B, n))


def bias_correction(boot, theta_hat):
    """Step 4. Returns (ẑ0, proportion below θ̂)."""
    B = len(boot)
    prop = float(np.sum(boot < theta_hat)) / B
    # A proportion of exactly 0 or 1 would give ẑ0 = ±∞; keep it inside (0, 1).
    prop_used = min(max(prop, 0.5 / B), 1.0 - 0.5 / B)
    return float(norm.ppf(prop_used)), prop


def jackknife(e):
    """Step 5. Leave-one-out MAEs and their average."""
    n = len(e)
    theta_i = (e.sum() - e) / (n - 1)
    return theta_i, float(theta_i.mean())


def acceleration(theta_i, theta_dot):
    """Step 6."""
    d = theta_dot - theta_i
    den = 6.0 * float(np.sum(d ** 2)) ** 1.5
    return float(np.sum(d ** 3) / den) if den > 0 else 0.0


def adjusted_levels(z0, a, alpha):
    """Step 7. Returns (α1, α2)."""
    z_lo, z_hi = norm.ppf(alpha), norm.ppf(1.0 - alpha)
    a1 = norm.cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)))
    a2 = norm.cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)))
    return float(a1), float(a2)


def percentile(boot, p):
    """Step 8: value at position p·(B − 1) of the sorted values, linear interpolation."""
    return float(np.quantile(boot, p, method="linear"))


def bca_mae(y_true, y_pred, idx, alpha=S.BCA["alpha"]):
    """Full protocol for one vehicle × method × variant. idx: (B, n) resampled rows."""
    e = np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))   # Step 1
    theta_hat = float(e.mean())                                          # Step 2
    boot = e[idx].mean(axis=1)                                           # Step 3
    z0, prop = bias_correction(boot, theta_hat)                          # Step 4
    theta_i, theta_dot = jackknife(e)                                    # Step 5
    a_hat = acceleration(theta_i, theta_dot)                             # Step 6
    a1, a2 = adjusted_levels(z0, a_hat, alpha)                           # Step 7
    return dict(MAE=theta_hat, boot_mean=float(boot.mean()), boot_sd=float(boot.std(ddof=1)),
                prop_below=prop, z0=z0, a=a_hat, alpha1=a1, alpha2=a2,
                ci_low=percentile(boot, a1), ci_high=percentile(boot, a2),   # Step 8
                boot=boot)


def selftest():
    """Example 1 of the protocol (10 bootstrap values, θ̂ = 5.0, â = 0)."""
    boot = np.array([3.8, 4.1, 4.3, 4.5, 4.7, 4.8, 4.9, 5.2, 5.6, 6.3])
    z0, prop = bias_correction(boot, 5.0)
    a1, a2 = adjusted_levels(z0, 0.0, 0.025)
    plain = (percentile(boot, 0.025), percentile(boot, 0.975))
    bca = (percentile(boot, a1), percentile(boot, a2))
    print(f"proportion below θ̂ = {prop:.2f}, ẑ0 = {z0:.2f}")
    print(f"α1 = {a1:.3f}, α2 = {a2:.4f}")
    print(f"percentile interval [{plain[0]:.2f}, {plain[1]:.2f}]   (protocol: [3.87, 6.14])")
    print(f"BCa interval        [{bca[0]:.2f}, {bca[1]:.2f}]   (protocol: [4.23, 6.29])")
    assert abs(z0 - 0.52) < 0.01 and abs(a1 - 0.181) < 0.001
    assert np.allclose(plain, (3.87, 6.14), atol=0.005) and np.allclose(bca, (4.23, 6.29), atol=0.005)
    print("Self-test passed.")


# =============================================================================
# Run for all methods
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="BCa confidence intervals for the MAE")
    p.add_argument("--selftest", action="store_true", help="Reproduce Example 1 of the protocol")
    p.add_argument("--vehicles", type=int, nargs="+", default=None)
    p.add_argument("--B", type=int, default=S.BCA["B"], help="Bootstrap trials (protocol: 10,000)")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return

    methods = {**S.BASE_METHODS, **S.FUSION_METHODS}
    vehicles = S.VEHICLES if args.vehicles is None else args.vehicles
    out_dir = S.OUTPUT_DIR / "BCa"
    rows = []
    for v in vehicles:
        frames = {m: C.read_predictions(folder, m, v) for m, folder in methods.items()}
        ref = next(iter(frames.values()))
        for m, df in frames.items():                     # same rows for every method
            C.check_same_targets(df["idx"].to_numpy(), ref["idx"].to_numpy())
            if not np.allclose(df["a1_true"], ref["a1_true"], atol=1e-4):
                raise ValueError(f"Vehicle {v}: a1_true differs between methods ({m})")
        rng = np.random.default_rng([S.SEED, v])
        idx = resample_indices(len(ref), args.B, rng)
        for m, df in frames.items():
            trials = {}
            for variant, col in S.PRED_COLUMN.items():
                r = bca_mae(ref["a1_true"], df[col], idx)
                trials[variant] = r.pop("boot")
                rows.append(dict(method=m, vehicle=v, variant=variant, **r))
            folder = out_dir / m
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(trials).to_csv(folder / f"V{v}_{m}_MAE_bootstrap.csv", index=False)
        print(f"  vehicle {v}: done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "bca_all_methods.csv", index=False)
    cols = ["MAE", "boot_mean", "boot_sd", "z0", "a", "alpha1", "alpha2", "ci_low", "ci_high"]
    for m in methods:
        wide = res[res.method == m].pivot(index="vehicle", columns="variant", values=cols)
        wide = wide.reorder_levels([1, 0], axis=1)[[(v, c) for v in S.VARIANTS for c in cols]]
        wide.columns = [f"{v} {c}" for v, c in wide.columns]
        wide.to_excel(out_dir / m / f"{m}_BCa_summary.xlsx")
    print(f"BCa: wrote {out_dir}")


if __name__ == "__main__":
    main()
