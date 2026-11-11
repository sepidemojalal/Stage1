"""
Intelligent Driver Model (IDM) with MCMC calibration (Stage 1).

    a_n(t) = a_max·[1 − (v_n(t−T)/v0)^δ − (s*(v_n, Δv) / Δx_n(t−T))²]
    s*     = s0 + T_i·v_n(t−T) + v_n(t−T)·Δv_n(t−T) / (2·√(a_max·b))

T = 1 s (10 rows): the inputs of row τ−10 predict the acceleration of row τ.
Fixed: δ = 4, v0 = 32 m/s, s0 = 2 m.  Calibrated: q_IDM = (a_max, T_i, b),
separately for each of the 32 models, with Metropolis–Hastings (mcmc.py).
The posterior means are used to predict the 92 test targets.

Notes
  - The dynamic term uses v_n(t−T): the speed at the predicted time is not
    known when predicting.
  - Δv_n in the formula is (follower − leader). The data's delta_v is
    (leader − follower) and is not changed; settings.IDM_DELTA_V_SIGN applies
    the conversion inside the formula (−1 → Δv_n = −delta_v).
  - Δx_n is delta_x − settings.IDM_LEADER_LENGTH (0 by default, i.e. delta_x as is).
  - s* is used as written, without clipping.
  - 20 % of the MCMC moves come from an independence proposal centred on the
    least-squares fit (see IDMCoordinates); the target posterior is unchanged.

Usage:
    python idm_mcmc.py            # full Stage 1 run
    python idm_mcmc.py --quick    # tiny settings, only to check the code runs
"""
import numpy as np

import mcmc
import settings as S

PARAMS = ["a_max", "T_i", "b"]
POSITIVE = [True, True, True]           # priors truncated to positive values

V0 = S.IDM_FIXED["v0"]
S0 = S.IDM_FIXED["s0"]
DELTA = S.IDM_FIXED["delta"]


def idm_predict(theta, X):
    """theta = (a_max, T_i, b); X columns = v1, delta_x, delta_v, a1 at τ−T."""
    a_max, T_i, b = theta
    v, dx, dv = X[:, 0], X[:, 1], X[:, 2]
    gap = dx - S.IDM_LEADER_LENGTH
    dv_n = S.IDM_DELTA_V_SIGN * dv                       # Δv_n = v_n − v_(n−1)
    s_star = S0 + T_i * v + v * dv_n / (2.0 * np.sqrt(a_max * b))
    return a_max * (1.0 - (v / V0) ** DELTA - (s_star / gap) ** 2)


class IDMCoordinates(mcmc.Identity):
    """
    The sampler moves in (a_max, T_i, b, log σ) themselves. b only enters through
    √(a_max·b), so the posterior can have separate modes that random-walk moves
    do not cross; 20 % of the moves are therefore drawn from a heavy-tailed
    proposal centred on the prior-regularised least-squares fit (a Laplace
    approximation). Accepted with the Metropolis–Hastings ratio, so the target
    posterior is unchanged.
    """
    def independence(self, X, y):
        from scipy.optimize import least_squares
        prior_mean = np.array([S.IDM_PRIORS[k][0] for k in PARAMS])
        prior_sd = np.array([S.IDM_PRIORS[k][1] for k in PARAMS])
        n = len(y)

        def resid(th):
            return idm_predict(th, X) - y

        best = None
        for x0 in ([1.5, 1.5, 2.0], [0.5, 1.0, 1.0], [3.0, 2.0, 4.0], [1.0, 0.5, 0.5]):
            with np.errstate(all="ignore"):
                r = least_squares(resid, x0, bounds=([1e-3] * 3, [20.0, 10.0, 100.0]))
            if np.isfinite(r.cost) and (best is None or r.cost < best.cost):
                best = r
        s2 = max(2.0 * best.cost / max(n - 3, 1), 1e-8)
        precision = best.jac.T @ best.jac / s2 + np.diag(1.0 / prior_sd ** 2)
        cov = np.zeros((4, 4))
        cov[:3, :3] = 4.0 * np.linalg.inv(precision)
        cov[3, 3] = 4.0 / (2.0 * n)                          # log σ
        cov += np.diag(np.full(4, 1e-12))
        return mcmc.StudentTProposal(np.append(best.x, 0.5 * np.log(s2)), cov)


if __name__ == "__main__":
    args = mcmc.parse_args("IDM with MCMC calibration")
    mcmc.run_physics_model("IDM", idm_predict, PARAMS, S.IDM_PRIORS, POSITIVE, args,
                           transform=IDMCoordinates())
