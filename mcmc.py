"""
Metropolis–Hastings calibration shared by Helly and IDM.

Procedure (methods text, Helly and IDM sections):
  1. Initialise priors: broad Gaussian priors on every parameter.
  2. Propose a new sample by perturbing the current parameters.
  3. Likelihood: apply the model equation to the training data and compare
     predicted with observed accelerations (Gaussian likelihood, noise σ).
  4. Accept or reject with the Metropolis ratio.
  5. Iterate; across epochs the previous posterior becomes the next prior.
  The calibrated parameters are the posterior means of the final epoch.

Implementation details
  - Several chains, each started from a random draw of the prior.
  - Burn-in tunes the proposal (one parameter at a time first, then joint
    moves using the burn-in covariance); the proposal is then frozen and the
    kept samples are drawn with a fixed random-walk proposal.
  - log σ is sampled together with the model parameters (not used to predict).
  - Convergence is reported with the split R-hat and the acceptance rate.
"""
import argparse
import time

import numpy as np
import pandas as pd

import common as C
import settings as S


# =============================================================================
# Prior and posterior
# =============================================================================
class GaussianPrior:
    def __init__(self, mean, sd, positive):
        self.mean = np.asarray(mean, float)
        self.sd = np.asarray(sd, float)
        self.positive = np.asarray(positive, bool)

    def logpdf(self, x):
        if np.any(x[self.positive] <= 0):
            return -np.inf
        z = (x - self.mean) / self.sd
        return -0.5 * float(np.dot(z, z))

    def draw(self, rng):
        for _ in range(100_000):
            x = rng.normal(self.mean, self.sd)
            if np.all(x[self.positive] > 0):
                return x
        raise RuntimeError("Could not draw a valid starting point from the prior")


class Identity:
    """Sampler coordinates = model parameters (+ log σ)."""
    def to_theta(self, w):
        return np.asarray(w, float)

    def from_theta(self, theta):
        return np.asarray(theta, float)

    def log_jacobian(self, w):
        return 0.0


def make_log_posterior(predict, X, y, prior, transform):
    """
    Log posterior of the sampler coordinates w. The prior is on the model
    parameters θ = transform.to_theta(w); log|∂θ/∂w| makes the target the
    posterior of θ, whatever coordinates the sampler moves in.
    """
    n = len(y)

    def log_post(w):
        with np.errstate(all="ignore"):
            theta = transform.to_theta(w)
            if not np.all(np.isfinite(theta)):
                return -np.inf
            lp = prior.logpdf(theta)
            if not np.isfinite(lp):
                return -np.inf
            lj = transform.log_jacobian(w)
            r = y - predict(theta[:-1], X)
        if not (np.isfinite(lj) and np.all(np.isfinite(r))):
            return -np.inf
        log_sigma = theta[-1]
        return lp + lj - n * log_sigma - 0.5 * float(np.dot(r, r)) * np.exp(-2.0 * log_sigma)

    return log_post


# =============================================================================
# One chain
# =============================================================================
def _chol(cov, fallback_sd):
    d = cov.shape[0]
    jitter = np.diag((1e-6 * fallback_sd) ** 2 + 1e-12)
    try:
        return np.linalg.cholesky(cov + jitter)
    except np.linalg.LinAlgError:
        return np.diag(fallback_sd)


class StudentTProposal:
    """Independence proposal: multivariate Student-t (heavy tails) with a fixed centre."""
    def __init__(self, mean, cov, df=5.0):
        self.mean = np.asarray(mean, float)
        self.L = np.linalg.cholesky(cov)
        self.df = df
        self.d = len(self.mean)

    def draw(self, rng):
        g = rng.chisquare(self.df) / self.df
        return self.mean + (self.L @ rng.standard_normal(self.d)) / np.sqrt(g)

    def logpdf(self, x):                          # up to a constant (cancels in the ratio)
        z = np.linalg.solve(self.L, np.asarray(x, float) - self.mean)
        return -0.5 * (self.df + self.d) * np.log1p(float(z @ z) / self.df)


def run_chain(log_post, x0, n_burn, n_samples, rng, target_accept, init_step,
              indep=None, p_indep=0.0):
    """
    Random-walk Metropolis–Hastings. If an independence proposal is given, a
    fraction p_indep of the moves are drawn from it instead (accepted with the
    Metropolis–Hastings ratio, so the target is unchanged); this lets a chain
    leave regions that random-walk moves cannot leave.
    """
    d = len(x0)
    x = np.array(x0, float)
    lp = log_post(x)
    n_acc, n_prop = 0, 0

    def indep_move(x, lp):
        prop = indep.draw(rng)
        lp_prop = log_post(prop)
        log_ratio = (lp_prop + indep.logpdf(x)) - (lp + indep.logpdf(prop))
        if np.log(rng.random()) < log_ratio:
            return prop, lp_prop, True
        return x, lp, False

    # Burn-in, part 1: one parameter at a time, each with its own step size
    n1 = max(n_burn // 2, 100)
    step = np.array(init_step, float)
    accepted = np.zeros(d)
    hist1 = np.empty((n1, d))
    for it in range(n1):
        for j in range(d):
            prop = x.copy()
            prop[j] += step[j] * rng.standard_normal()
            lp_prop = log_post(prop)
            if np.log(rng.random()) < lp_prop - lp:
                x, lp = prop, lp_prop
                accepted[j] += 1
        if indep is not None and rng.random() < p_indep:
            x, lp, _ = indep_move(x, lp)
        hist1[it] = x
        if (it + 1) % 50 == 0:
            step *= np.exp(accepted / 50 - 0.44)       # 0.44: good rate for 1-D moves
            accepted[:] = 0

    # Burn-in, part 2: joint moves with the covariance of part 1 (second half)
    scale_const = 2.38 ** 2 / d
    cov = np.atleast_2d(np.cov(hist1[n1 // 2:], rowvar=False))
    L = _chol(scale_const * cov, step)
    scale = 1.0
    n2 = max(n_burn - n1, 100)
    hist2 = np.empty((n2, d))
    acc, tried = 0, 0
    for it in range(n2):
        if indep is not None and rng.random() < p_indep:
            x, lp, _ = indep_move(x, lp)
        else:
            prop = x + scale * (L @ rng.standard_normal(d))
            lp_prop = log_post(prop)
            tried += 1
            if np.log(rng.random()) < lp_prop - lp:
                x, lp = prop, lp_prop
                acc += 1
        hist2[it] = x
        if tried == 100:
            scale *= np.exp(acc / 100 - target_accept)
            acc, tried = 0, 0
    cov2 = np.atleast_2d(np.cov(hist2[n2 // 2:], rowvar=False))
    if np.all(np.isfinite(cov2)) and np.all(np.diag(cov2) > 0):
        L = _chol(scale_const * cov2, step)

    # Kept samples: proposals frozen
    samples = np.empty((n_samples, d))
    for it in range(n_samples):
        if indep is not None and rng.random() < p_indep:
            x, lp, ok = indep_move(x, lp)
        else:
            prop = x + scale * (L @ rng.standard_normal(d))
            lp_prop = log_post(prop)
            ok = np.log(rng.random()) < lp_prop - lp
            if ok:
                x, lp = prop, lp_prop
        n_acc += int(ok)
        n_prop += 1
        samples[it] = x
    return samples, n_acc / n_prop, x


def split_rhat(chains):
    """chains: (m, n, d) → split R-hat per parameter (Gelman et al.)."""
    m, n, d = chains.shape
    h = n // 2
    sp = np.concatenate([chains[:, :h], chains[:, h:2 * h]], axis=0)
    means = sp.mean(axis=1)
    W = sp.var(axis=1, ddof=1).mean(axis=0)
    B = h * means.var(axis=0, ddof=1)
    var_hat = (h - 1) / h * W + B / h
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(var_hat / W)


# =============================================================================
# Calibration with epochs
# =============================================================================
def calibrate(predict, X, y, names, priors, positive, cfg, seed, transform=None):
    """
    names / priors / positive describe the model parameters; log σ is appended.
    transform: sampler coordinates (default: the parameters themselves).
    Returns posterior means (calibrated parameters), SDs, R-hat, acceptance,
    and a thinned set of posterior draws of the parameters.
    """
    transform = Identity() if transform is None else transform
    rng = np.random.default_rng(seed)
    all_names = list(names) + ["log_sigma"]
    mean = [priors[k][0] for k in names] + [S.LOG_SIGMA_PRIOR[0]]
    sd = [priors[k][1] for k in names] + [S.LOG_SIGMA_PRIOR[1]]
    pos = list(positive) + [False]
    prior = GaussianPrior(mean, sd, pos)

    # Optional independence proposal supplied by the sampler coordinates
    indep = transform.independence(X, y) if hasattr(transform, "independence") else None

    starts = None
    epochs = []
    for e in range(cfg["n_epochs"]):
        log_post = make_log_posterior(predict, X, y, prior, transform)
        chains, accs, ends = [], [], []
        for c in range(cfg["n_chains"]):
            if starts is None:                          # random initial guess
                for _ in range(10_000):
                    w0 = transform.from_theta(prior.draw(rng))
                    if np.isfinite(log_post(w0)):
                        break
                else:
                    raise RuntimeError("No valid starting point found")
            else:                                       # continue from last state
                w0 = starts[c]
            init_step = 0.1 * (prior.sd if isinstance(transform, Identity)
                               else np.maximum(np.abs(w0), 0.01))
            s, a, w_end = run_chain(log_post, w0, cfg["n_burn"], cfg["n_samples"], rng,
                                    cfg["target_accept"], init_step=init_step,
                                    indep=indep, p_indep=cfg.get("p_independence", 0.0))
            chains.append(transform.to_theta(s))        # store parameter values
            accs.append(a)
            ends.append(w_end)
        chains = np.stack(chains)                       # (chains, samples, params)
        pooled = chains.reshape(-1, chains.shape[-1])
        post_mean = pooled.mean(axis=0)
        post_sd = pooled.std(axis=0, ddof=1)
        rhat = split_rhat(chains)
        epochs.append(dict(epoch=e + 1, mean=post_mean, sd=post_sd, rhat=rhat,
                           accept=float(np.mean(accs))))
        # previous posterior → prior of the next epoch
        prior = GaussianPrior(post_mean, np.maximum(post_sd, 1e-12), pos)
        starts = ends

    last = epochs[-1]
    thin = pooled[rng.choice(len(pooled), size=min(2000, len(pooled)), replace=False)]
    return dict(names=all_names, mean=last["mean"], sd=last["sd"], rhat=last["rhat"],
                accept=last["accept"], epochs=epochs, draws=thin)


# =============================================================================
# Full Stage 1 run for one physics model (used by helly_mcmc.py and idm_mcmc.py)
# =============================================================================
def parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true",
                   help="Tiny MCMC settings to check that the code runs (not for results).")
    p.add_argument("--vehicles", type=int, nargs="+", default=None,
                   help="Subset of vehicles, for testing only (CDT sets are then built "
                        "from this subset). The Stage 1 experiment uses all 30.")
    return p.parse_args()


def predict_with(res, predict, X):
    """Test predictions from a calibrated model (settings.MCMC_PREDICTION)."""
    if S.MCMC_PREDICTION == "posterior_mean":          # methods text: posterior means
        return predict(res["mean"][:-1], X)
    if S.MCMC_PREDICTION == "posterior_predictive":    # average over posterior draws
        return np.mean([predict(d[:-1], X) for d in res["draws"]], axis=0)
    raise ValueError(f"Unknown MCMC_PREDICTION: {S.MCMC_PREDICTION}")


def run_physics_model(method, predict, names, priors, positive, args, transform=None):
    cfg = dict(S.MCMC)
    if args.quick:
        cfg.update(n_chains=2, n_epochs=1, n_burn=200, n_samples=200)

    data = C.load_all(args.vehicles)
    sets = C.physics_training_sets(data)
    out_dir = S.OUTPUT_DIR / method
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    def fit(label, pairs, seed):
        t0 = time.time()
        res = calibrate(predict, pairs["X"], pairs["y"], names, priors, positive, cfg, seed,
                        transform)
        res["seconds"] = time.time() - t0
        res["n_train"] = len(pairs["y"])
        # Training fit with the posterior means and with the posterior draws: a large
        # gap means the posterior is not well summarised by its means.
        res["train_rmse_posterior_mean"] = float(np.sqrt(np.mean(
            (pairs["y"] - predict(res["mean"][:-1], pairs["X"])) ** 2)))
        pp = np.mean([predict(d[:-1], pairs["X"]) for d in res["draws"][:200]], axis=0)
        res["train_rmse_posterior_predictive"] = float(np.sqrt(np.mean((pairs["y"] - pp) ** 2)))
        # Training MAE / MSE: the calibrated model on the pairs it was calibrated on
        res["train_metrics"] = C.mae_mse(pairs["y"], predict_with(res, predict, pairs["X"]))
        flag = "" if np.nanmax(res["rhat"]) < 1.05 else "   <-- check convergence"
        print(f"  {method} {label:>16}: n={res['n_train']:5d}  "
              f"max R-hat={np.nanmax(res['rhat']):.3f}  accept={res['accept']:.2f}  "
              f"train RMSE={res['train_rmse_posterior_mean']:.3f}  "
              f"({res['seconds']:.1f} s){flag}", flush=True)
        return res

    print(f"{method}: calibrating CDT models", flush=True)
    fits = {"CDT Full": fit("CDT Full", sets["full"], [S.SEED, 0]),
            "CDT Limited": fit("CDT Limited", sets["limited"], [S.SEED, 1])}

    param_rows, epoch_rows, metric_rows = [], [], []

    def record(label, res):
        row = dict(model=label, n_train=res["n_train"], acceptance=res["accept"],
                   max_rhat=float(np.nanmax(res["rhat"])), seconds=res["seconds"],
                   train_MAE=res["train_metrics"][0], train_MSE=res["train_metrics"][1],
                   train_rmse_posterior_mean=res["train_rmse_posterior_mean"],
                   train_rmse_posterior_predictive=res["train_rmse_posterior_predictive"])
        for k, m, s, r in zip(res["names"], res["mean"], res["sd"], res["rhat"]):
            row[k] = m
            row[f"{k}_sd"] = s
            row[f"{k}_rhat"] = r
        row["sigma"] = float(np.exp(res["mean"][-1]))
        param_rows.append(row)
        for ep in res["epochs"]:
            er = dict(model=label, epoch=ep["epoch"], acceptance=ep["accept"])
            for k, m, s, r in zip(res["names"], ep["mean"], ep["sd"], ep["rhat"]):
                er[k] = m
                er[f"{k}_sd"] = s
                er[f"{k}_rhat"] = r
            epoch_rows.append(er)

    record("CDT Full", fits["CDT Full"])
    record("CDT Limited", fits["CDT Limited"])

    print(f"{method}: calibrating User-specific models", flush=True)
    for v, (_, test) in data.items():
        user = fit(f"User-specific V{v}", sets["user"][v], [S.SEED, 100 + v])
        record(f"User-specific V{v}", user)
        tp = C.test_points(test)
        models = {"User-specific": user, "CDT Full": fits["CDT Full"],
                  "CDT Limited": fits["CDT Limited"]}
        preds = {k: predict_with(res, predict, tp["X"]) for k, res in models.items()}
        C.save_predictions(pred_dir, method, v, tp["target_idx"], tp["y"], preds)
        metric_rows += C.metrics_table(method, v, tp["y"], preds,
                                       train={k: res["train_metrics"] for k, res in models.items()})

    pd.DataFrame(param_rows).to_csv(out_dir / "calibrated_parameters.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out_dir / "mcmc_epochs.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
    print(f"{method}: wrote {pred_dir}, calibrated_parameters.csv, mcmc_epochs.csv, metrics.csv")
