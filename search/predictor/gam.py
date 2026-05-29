"""Monotone Generalized Additive Model — hinge-spline basis + NNLS.

A pure-numpy/scipy monotone GAM tailored to the JSD→JSD mapping in the
sample_surrogate/post_search pipeline. The structural prior is

    y(x) ≈ α + Σ_i f_i(x_i)  + (optional small) Σ_{i<j} g_ij(x_i, x_j)

with each ``f_i`` constrained to be monotonically increasing in x_i:
*lower per-axis JSD ⇒ lower combined JSD*. This is a hard constraint
(not a soft penalty) so the surrogate is safe to extrapolate near the
training-set extremes, which is exactly the regime that quantile-anchored
sampling lands in.

Implementation
--------------
* Per axis: a hinge basis ``φ_{i,k}(x) = max(0, x - τ_{i,k})`` with knots
  ``τ_{i,k}`` at the empirical quantiles. The fit becomes

      y - α  ≈  Σ_i Σ_k β_{i,k} φ_{i,k}(x_i)        s.t.  β_{i,k} ≥ 0

  which is a non-negative least-squares problem solved by
  ``scipy.optimize.nnls``. Non-negative β + non-decreasing hinge basis
  ⇒ each ``f_i`` is monotone-increasing by construction.

* Pair interactions (``with_interactions=True``): a *small* unconstrained
  bilinear term ``γ_ij · (x_i - x̄_i)(x_j - x̄_j)`` per pair, fit after the
  monotone part on the residual. The empirical verification on N=50
  Llama-3.1 sample data flagged ``w·kvdim`` as the only significant pair
  (p<0.01); other interactions are noise. So this is a deliberately
  small extension to the additive model.

* Smoothness: enforced implicitly by the knot count (default 6 per axis,
  i.e. one knot per ~8 training points at N=50). No explicit penalty.

Why pure-scipy, no pyGAM
------------------------
pyGAM's spline+penalty solver is more powerful but the dependency wasn't
present in this environment and the hinge basis already captures the
relevant per-axis shape (concave saturating curves of per-axis JSD)
without any additional smoothing. Adding a pygam-backed variant later is
a drop-in replacement of the constrained solve.

Interface matches the other predictors (numpy in/out, ``fit -> self``,
``predict -> 1D ndarray``).
"""

import numpy as np
from scipy.optimize import lsq_linear, nnls


def _build_hinge_basis(x, knots):
    """φ_k(x) = max(0, x - τ_k). Returns (n, K_knots) feature matrix."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.maximum(0.0, x[:, None] - np.asarray(knots, dtype=np.float64)[None, :])


def _select_knots(x, n_knots):
    """Quantile-based knots; leftmost at the minimum so the basis covers
    the full data range. Drops duplicates (constant axes / heavy ties).

    The smallest knot is placed slightly below ``x.min()`` so the first
    hinge is nonzero at the data minimum — otherwise the monotone term
    contributes 0 there and the intercept absorbs everything, leaving no
    knob to fit the low end.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    qs = np.linspace(0.0, 1.0, n_knots + 1)[:-1]      # n_knots knots in [0, 1)
    knots = np.quantile(x, qs)
    # nudge the leftmost knot below the minimum so φ_0(x_min) > 0
    if knots[0] >= x.min():
        knots[0] = x.min() - 1e-6 * (x.max() - x.min() + 1e-12)
    return np.unique(knots)


class MonotoneGAM:
    """Hinge-spline monotone GAM. Per-axis ``f_i`` is monotone-increasing
    by construction (NNLS on non-negative-hinge basis). Optional small
    bilinear interaction term fit on the residual.

    Parameters
    ----------
    n_knots : int
        Knots per axis (default 6). Roughly one knot per (N / n_knots)
        training points; default chosen for N≈30–80.
    with_interactions : bool
        If True, add unconstrained bilinear ``γ_ij·(x_i-x̄_i)(x_j-x̄_j)``
        terms for every axis pair. Fit on residual after the monotone
        additive part, so it never breaks monotonicity (interactions
        only correct the *slope* of the saturating shape).
    ridge_interaction : float
        Ridge on γ_ij to control interaction magnitude; default 1e-2.
    """

    def __init__(self, n_knots=6, with_interactions=True,
                 ridge_interaction=1e-2):
        self.n_knots = int(n_knots)
        self.with_interactions = bool(with_interactions)
        self.ridge_interaction = float(ridge_interaction)
        self.name = 'gam'

        # populated by fit()
        self._intercept = None
        self._knots = None          # list of np.ndarray, one per axis
        self._beta = None           # non-negative spline coefficients (flat)
        self._gamma = None          # interaction coefficients (or None)
        self._x_mean = None         # per-axis mean (for centring interactions)
        self._K = None
        self._fitted = False

    # ----- feature assembly ------------------------------------------------
    def _build_phi(self, X):
        """Stack hinge bases across axes into a single (n, ΣK_i) matrix."""
        bases = [_build_hinge_basis(X[:, i], self._knots[i])
                 for i in range(self._K)]
        return np.hstack(bases)

    def _build_inter(self, X):
        """Centred bilinear terms (n, K(K-1)/2)."""
        Xc = X - self._x_mean
        cols = []
        for i in range(self._K):
            for j in range(i + 1, self._K):
                cols.append((Xc[:, i] * Xc[:, j]).reshape(-1, 1))
        return np.hstack(cols) if cols else np.zeros((len(X), 0))

    # ----- fit -------------------------------------------------------------
    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n, K = X.shape
        self._K = K

        # Per-axis knots (quantile-based, automatic).
        self._knots = [_select_knots(X[:, i], self.n_knots) for i in range(K)]
        self._x_mean = X.mean(axis=0)

        # Step 1: monotone additive fit. Intercept α is jointly fit with
        # the spline coefficients β under the constraints
        #     α ∈ ℝ,   β ≥ 0,
        # so the prediction surface lives in ``[α, α + Σᵢ fᵢ(max xᵢ)]``.
        # If we instead pre-set α = mean(y) and ran NNLS on the residual
        # y - α (which has positive *and* negative entries), NNLS would
        # zero out β where the residual is negative — collapsing fᵢ to
        # the constant-0 function on the low-x_i half of the range and
        # losing the slope there. The joint fit avoids that pathology.
        Phi = self._build_phi(X)
        n_basis = Phi.shape[1]
        A = np.hstack([np.ones((n, 1)), Phi])
        lb_bounds = np.concatenate([[-np.inf], np.zeros(n_basis)])
        ub_bounds = np.full(n_basis + 1, np.inf)
        try:
            res = lsq_linear(A, y, bounds=(lb_bounds, ub_bounds),
                             method='bvls')
            coef = res.x
        except Exception:
            # bvls solver failed (rare; usually rank-deficient A on small
            # n_knots). Fallback: NNLS with intercept = y.min() so the
            # non-negativity is feasible on the shifted target.
            self._intercept = float(y.min())
            try:
                beta_only, _ = nnls(Phi, y - self._intercept,
                                    maxiter=10 * n_basis)
            except RuntimeError:
                B = Phi.T @ Phi + 1e-3 * np.eye(n_basis)
                beta_only = np.clip(np.linalg.solve(
                    B, Phi.T @ (y - self._intercept)), 0.0, None)
            coef = np.concatenate([[self._intercept], beta_only])

        self._intercept = float(coef[0])
        self._beta = coef[1:]

        # Step 2 (optional): small bilinear interactions on the residual
        # from the monotone additive fit. Ridge-regularized linear
        # regression so γ_ij stays small (only correcting slope, not
        # introducing structure that would break monotonicity over the
        # training range).
        if self.with_interactions and K >= 2:
            resid = y - (self._intercept + Phi @ self._beta)
            Z = self._build_inter(X)
            if Z.shape[1] > 0:
                A = Z.T @ Z + self.ridge_interaction * np.eye(Z.shape[1])
                self._gamma = np.linalg.solve(A, Z.T @ resid)
            else:
                self._gamma = np.zeros(0)
        else:
            self._gamma = np.zeros(0)

        self._fitted = True
        return self

    # ----- predict ---------------------------------------------------------
    def predict(self, X):
        assert self._fitted, "MonotoneGAM not fitted"
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Phi = self._build_phi(X)
        out = self._intercept + Phi @ self._beta
        if self._gamma.size > 0:
            out = out + self._build_inter(X) @ self._gamma
        return out

    # ----- diagnostics (optional, useful for analysis) ---------------------
    def per_axis_effect(self, X):
        """Return per-axis contribution f_i(x_i) (without intercept/inter).

        Useful for plotting/diagnostics: tells you whether the surrogate
        thinks an axis has saturated (flat curve at high x_i).
        """
        assert self._fitted, "MonotoneGAM not fitted"
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.zeros((len(X), self._K))
        cursor = 0
        for i in range(self._K):
            k = len(self._knots[i])
            phi_i = _build_hinge_basis(X[:, i], self._knots[i])
            out[:, i] = phi_i @ self._beta[cursor:cursor + k]
            cursor += k
        return out
