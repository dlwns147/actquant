"""Generic Y-target transform wrapper for surrogate predictors.

Wraps any predictor that follows the project's standard interface
(``fit(X, y) -> self``, ``predict(X) -> 1D ndarray``) to train and
predict on a *transformed* target. The transformation is applied
forward at fit time (``y_t = g(y)``) and undone at predict time
(``ŷ = g⁻¹(ŷ_t)``), so the surrogate sees a target whose distribution
better matches its modelling assumptions.

Available transforms
--------------------
* ``sqrt``  — for JSD targets. sqrt(JSD) is closer to a metric
  (Hellinger distance: ``H² ≤ JSD ≤ 2·H·ln 2``); stationary kernels
  (RBF, Matérn GP) condition better on the transformed scale,
  especially in the low-Y region (high-fidelity quantization) which is
  what NAS deployment cares about.
* ``log``   — for PPL targets. PPL has a log-distributed tail; fitting
  in log-space removes outlier influence on MSE-loss surrogates.
* ``logit`` — for bounded [0, 1] targets (e.g. zeroshot accuracies).
  Maps to (-∞, ∞) so GP / MLP Gaussian noise models are well-posed.
* ``identity`` — pass-through (debug / explicit no-op).

The wrapper is *predictor-agnostic*: it works on RBF, ARDGP, MLP,
GAM, BayesianAdditiveQuadratic, etc. The marginal benefit of each
transform depends on the base predictor's modelling assumptions
(stationary kernels gain the most; tree models and additive-feature
linear models gain less because they don't assume a metric on Y).

Predict-variance handling
-------------------------
If the base predictor exposes ``predict_variance(X)``, the wrapper
propagates it via the delta method:

    Var[Y_pred] ≈ (dg⁻¹/dF)² · Var[F_pred]

For ``sqrt`` this is ``(2·F)² · Var[F]``; for ``log`` it's
``exp(2·F) · Var[F]``. Absolute calibration is approximate but AL
acquisition functions only use *relative* ordering of σ across
candidates, which the delta method preserves monotonically.

If the base predictor does not expose ``predict_variance``, the
wrapper raises ``NotImplementedError`` — callers wanting AL σ should
either wrap an inherently-Bayesian base (``ard_gp``, ``badd_quad``)
or stack a ``BaggingEnsemble`` underneath this wrapper.
"""

import numpy as np


# Each transform: (forward g, inverse g⁻¹, derivative of inverse |g⁻¹′(f)|).
# All operate elementwise on numpy arrays.
def _sqrt_fwd(y):
    return np.sqrt(np.clip(y, 0.0, None))


def _sqrt_inv(f):
    # Negative mean predictions on the sqrt scale → 0 on original scale,
    # otherwise they'd come back positive after squaring (wrong residual
    # sign). Order-preserving for f ≥ 0 (the well-modelled region).
    return np.clip(f, 0.0, None) ** 2


def _sqrt_dinv(f):
    return 2.0 * np.clip(f, 0.0, None)


def _log_fwd(y):
    return np.log(np.clip(y, 1e-30, None))


def _log_inv(f):
    return np.exp(f)


def _log_dinv(f):
    return np.exp(f)


def _logit_fwd(y, eps=1e-6):
    yc = np.clip(y, eps, 1.0 - eps)
    return np.log(yc / (1.0 - yc))


def _logit_inv(f):
    return 1.0 / (1.0 + np.exp(-f))


def _logit_dinv(f):
    p = 1.0 / (1.0 + np.exp(-f))
    return p * (1.0 - p)


TRANSFORMS = {
    'identity': (lambda y: y, lambda f: f, lambda f: np.ones_like(f)),
    'sqrt':     (_sqrt_fwd,  _sqrt_inv,  _sqrt_dinv),
    'log':      (_log_fwd,   _log_inv,   _log_dinv),
    'logit':    (_logit_fwd, _logit_inv, _logit_dinv),
}


class TargetTransformPredictor:
    """Wraps an already-instantiated base predictor; trains it on ``g(y)``,
    predicts ``g⁻¹(ŷ_t)``.

    Parameters
    ----------
    base : predictor
        Must have ``fit(X, y) -> self`` and ``predict(X) -> 1D or 2D
        ndarray``. RBF returns (n, 1); ARDGP returns (n,). Both
        handled here via ``.reshape(-1)``.
    transform : str
        One of ``TRANSFORMS`` keys: ``sqrt``, ``log``, ``logit``,
        ``identity``.
    """

    def __init__(self, base, transform='sqrt'):
        if transform not in TRANSFORMS:
            raise ValueError(
                f"unknown transform {transform!r}; valid: {list(TRANSFORMS)}")
        self.base = base
        self.transform = transform
        base_name = getattr(base, 'name', type(base).__name__).lower()
        prefix = transform if transform != 'sqrt' else 'sqrty'
        self.name = f'{prefix}_{base_name}'
        self._fwd, self._inv, self._dinv = TRANSFORMS[transform]
        self._fitted = False

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.base.fit(X, self._fwd(y))
        self._fitted = True
        return self

    def predict(self, X):
        assert self._fitted, f"{self.name} not fitted"
        f = np.asarray(self.base.predict(X), dtype=np.float64).reshape(-1)
        return self._inv(f)

    def predict_variance(self, X, include_noise=True):
        """Delta-method variance on the original Y scale.

        Forwards to the base's ``predict_variance`` (if it has one).
        For ``include_noise`` semantics: the base's σ²_obs is on the
        transformed scale; multiplying by ``(g⁻¹′)²`` gives the
        approximate observation variance on the original scale.
        """
        assert self._fitted, f"{self.name} not fitted"
        if not hasattr(self.base, 'predict_variance'):
            raise NotImplementedError(
                f"base predictor {type(self.base).__name__} does not expose "
                f"predict_variance; wrap it in a bagging ensemble for AL σ.")
        f = np.asarray(self.base.predict(X), dtype=np.float64).reshape(-1)
        # Try include_noise kwarg first, fall back to no-kwarg call.
        try:
            v_f = self.base.predict_variance(X, include_noise=include_noise)
        except TypeError:
            v_f = self.base.predict_variance(X)
        v_f = np.asarray(v_f, dtype=np.float64).reshape(-1)
        d = self._dinv(f)
        return np.clip(d * d * v_f, 0.0, None)
