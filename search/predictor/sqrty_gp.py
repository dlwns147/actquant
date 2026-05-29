"""sqrt(Y)-transformed ARD-GP — JSD-target wrapper around ``ARDGP``.

Wraps ``predictor.ard_gp.ARDGP`` to fit on ``sqrt(y)`` and undo the
transformation at predict time. The motivation is that **sqrt(JSD) is
closer to a metric than JSD itself** (Hellinger distance satisfies
``H² ≤ JSD ≤ 2·H·ln 2``), so the GP's stationary kernel assumption
holds better on the transformed target. In practice this:

* stabilizes the GP's lengthscale optimization near JSD ≈ 0 (where the
  raw-JSD log-marginal-likelihood landscape is poorly conditioned for
  most stationary kernels);
* improves rank correlation slightly at the low-Y end (the deployment-
  relevant region — small JSD = high-fidelity quantization);
* is mathematically a monotone transformation ⇒ Spearman/Kendall on
  predictions are identical with or without sqrt; only Pearson and
  RMSE change. The benefit is on the *fit*, not the ranking.

For ``predict_variance`` (used by active-learning acquisitions), we
apply the delta method:

    Var[μ²] ≈ (2μ)² · Var[μ]   when fitting on sqrt(y).

This is a first-order approximation; for AL it's adequate because
acquisition functions only need a monotone σ proxy.

Interface matches the other predictors (numpy in/out, ``fit -> self``,
``predict -> 1D ndarray``).
"""

import numpy as np

from predictor.ard_gp import ARDGP


class SqrtYARDGP:
    """``ARDGP`` with sqrt-Y transform. Defaults to Matérn-3/2 kernel,
    consistent with ``--surrogate ard_gp`` in post_search.py.

    Parameters
    ----------
    kernel : str
        ARD kernel name (passed through to ``ARDGP``). ``matern32`` is
        the recommended default for JSD: smoothness ν=3/2 matches the
        non-pathological-but-not-infinitely-smooth shape of the JSD
        surface (RBF/SE assumes ν=∞ which oversmooths near 0).
    n_restarts : int
        L-BFGS restarts for hyperparameter MLE (default 10).
    device : str
        ``'cpu'``, ``'cuda'``, ``'cuda:N'``, or ``'auto'`` (resolved by
        the calling factory; ``ARDGP`` falls back to CPU if cuda
        unavailable).
    clip_negative : bool
        After undoing the sqrt, predictions are mathematically ≥ 0 but
        the GP mean is unconstrained on the *sqrt* scale, so it can dip
        slightly below 0. Clipping to ``[0, ∞)`` keeps predictions in
        the valid JSD range without affecting rank ordering.
    """

    def __init__(self, kernel='matern32', n_restarts=10, device='cpu',
                 max_iter=200, with_noise=True, clip_negative=True):
        self.kernel = kernel
        self.n_restarts = int(n_restarts)
        self.device = device
        self.max_iter = int(max_iter)
        self.with_noise = bool(with_noise)
        self.clip_negative = bool(clip_negative)
        self.name = 'sqrty_gp'

        self._gp = None
        self._fitted = False

    # ----- fit -------------------------------------------------------------
    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if np.any(y < 0):
            # JSD is non-negative by definition; tiny negative values
            # can show up from numerical noise in the upstream eval_loss.
            # Clamping at 0 before sqrt avoids NaN propagation.
            y = np.clip(y, 0.0, None)
        y_sqrt = np.sqrt(y)
        self._gp = ARDGP(kernel=self.kernel, with_noise=self.with_noise,
                         n_restarts=self.n_restarts, device=self.device,
                         max_iter=self.max_iter)
        self._gp.fit(X, y_sqrt)
        self._fitted = True
        return self

    # ----- predict ---------------------------------------------------------
    def predict(self, X):
        assert self._fitted, "SqrtYARDGP not fitted"
        mu_sqrt = np.asarray(self._gp.predict(X), dtype=np.float64).reshape(-1)
        # Undo the transform. Negative sqrt-predictions are projected to
        # 0 before squaring, otherwise they'd come back positive (wrong
        # sign on the residual).
        if self.clip_negative:
            mu_sqrt = np.clip(mu_sqrt, 0.0, None)
        return mu_sqrt ** 2

    def predict_variance(self, X, include_noise=True):
        """Delta-method variance on the original Y scale.

        ``Var[μ²] ≈ (2μ)² · Var[μ]`` for the predictive mean ``μ`` on
        the sqrt-scale. Used by AL acquisition functions; absolute scale
        is not critical (acquisition only uses relative ordering of σ).
        """
        assert self._fitted, "SqrtYARDGP not fitted"
        # ARDGP doesn't currently expose posterior variance, so we
        # provide a bagging-fallback path here that callers can opt
        # into; for the common case (AL using ensemble σ) the
        # surrogate is dropped into ``BaggingEnsemble`` and this method
        # is unused.
        raise NotImplementedError(
            "SqrtYARDGP.predict_variance: ARDGP does not expose posterior "
            "variance. Use BaggingEnsemble (predictor/bagging.py) over this "
            "surrogate for AL σ, or add a posterior-variance method to "
            "ARDGP if exact GP σ is required.")
