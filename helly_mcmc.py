"""
Helly car-following model with MCMC calibration (Stage 1).

    a_n(t) = C1·Δv(t−T) + C2·[Δx(t−T) − α − β·v_n(t−T) − γ·a_n(t−T)]

T = 1 s (10 rows): the inputs of row τ−10 predict the acceleration of row τ.
Parameters q_Helly = (C1, C2, α, β, γ), calibrated separately for each of the
32 models (30 User-specific, CDT Full, CDT Limited) with Metropolis–Hastings
(see mcmc.py). The posterior means are used to predict the 92 test targets.

delta_v is used exactly as it is in the data (leader − follower); C1 is
calibrated freely, so it simply takes the matching sign (positive here).

Sampler coordinates. α, β and γ only enter the equation multiplied by C2, so
in (C1, C2, α, β, γ) the posterior is a curved ridge that random-walk
proposals cannot explore (chains do not converge). The sampler therefore
moves in the equation's linear coefficients (C1, C2, −C2·α, −C2·β, −C2·γ),
in which the likelihood is Gaussian. The Jacobian of the change of variables
is included, so the chains still sample the posterior of (C1, C2, α, β, γ)
under the Gaussian priors in settings.HELLY_PRIORS; everything reported is in
(C1, C2, α, β, γ).
Near C2 = 0 these coordinates form a narrow funnel that random-walk chains
cannot leave, so 20 % of the moves are drawn from an independence proposal
centred on the least-squares fit (accepted with the Metropolis–Hastings ratio,
which keeps the same target posterior).

Usage:
    python helly_mcmc.py            # full Stage 1 run
    python helly_mcmc.py --quick    # tiny settings, only to check the code runs
"""
import numpy as np

import mcmc
import settings as S

PARAMS = ["C1", "C2", "alpha", "beta", "gamma"]
POSITIVE = [False] * len(PARAMS)       # no sign restriction on Helly parameters


def helly_predict(theta, X):
    """theta = (C1, C2, alpha, beta, gamma); X columns = v1, delta_x, delta_v, a1 at τ−T."""
    C1, C2, alpha, beta, gamma = theta
    v, dx, dv, a = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return C1 * dv + C2 * (dx - alpha - beta * v - gamma * a)


class HellyLinearCoordinates:
    """
    w = (C1, C2, k_α, k_β, k_γ, log σ) with (k_α, k_β, k_γ) = −C2·(α, β, γ).
    θ = (C1, C2, α, β, γ, log σ) with (α, β, γ) = −(k_α, k_β, k_γ)/C2.
    |det ∂θ/∂w| = |C2|⁻³.
    """
    def to_theta(self, w):
        w = np.asarray(w, float)
        theta = w.copy()
        theta[..., 2:5] = -w[..., 2:5] / w[..., 1:2]
        return theta

    def from_theta(self, theta):
        theta = np.asarray(theta, float)
        w = theta.copy()
        w[..., 2:5] = -theta[..., 2:5] * theta[..., 1:2]
        return w

    def log_jacobian(self, w):
        return -3.0 * np.log(np.abs(w[1]))

    def independence(self, X, y):
        """
        Independence proposal in w: a heavy-tailed Student-t centred on the
        least-squares fit of the linear coefficients, with 4× their covariance.
        It only affects how fast the chains move, not the posterior they sample.
        """
        v, dx, dv, a = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        A = np.column_stack([dv, dx, np.ones_like(v), v, a])   # C1, C2, k_α, k_β, k_γ
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        n, p = A.shape
        s2 = float(resid @ resid) / max(n - p, 1)
        cov = np.zeros((p + 1, p + 1))
        cov[:p, :p] = 4.0 * s2 * np.linalg.pinv(A.T @ A)
        cov[p, p] = 4.0 / (2.0 * n)                             # log σ
        cov += np.diag(np.full(p + 1, 1e-12))
        return mcmc.StudentTProposal(np.append(coef, 0.5 * np.log(s2)), cov)


if __name__ == "__main__":
    args = mcmc.parse_args("Helly model with MCMC calibration")
    mcmc.run_physics_model("Helly", helly_predict, PARAMS, S.HELLY_PRIORS, POSITIVE, args,
                           transform=HellyLinearCoordinates())
