"""Surrogate predictor factory.

Dispatches ``model`` name → constructed-and-fit predictor. Base
predictors live in ``predictor/<name>.py``; Y-target transforms are
handled by ``predictor.target_transform.TargetTransformPredictor`` via
the ``<transform>_<base>`` naming convention.

Naming convention for target-transformed surrogates:
    ``sqrty_<base>``  — sqrt(Y) on top of any base predictor
    ``logy_<base>``   — log(Y)  on top of any base predictor
    ``logity_<base>`` — logit(Y) on top of any base predictor

Examples: ``sqrty_rbf``, ``sqrty_ard_gp``, ``sqrty_mlp``,
``sqrty_badd_quad``, ``sqrty_gam``, ``logy_rbf``.

Backward-compat aliases (kept until callers migrate):
    ``sqrty_gp`` == ``sqrty_ard_gp``
"""

# Transform prefix → kwarg name used by TargetTransformPredictor.
_TRANSFORM_PREFIXES = {
    'sqrty_': 'sqrt',
    'logy_': 'log',
    'logity_': 'logit',
}

# Names registered with argparse choices in post_search.py. Keep
# enumerated so ``--help`` lists every valid surrogate.
BASE_PREDICTORS = ('rbf', 'gp', 'mlp', 'carts', 'as', 'ard_gp',
                   'badd_quad', 'gam')


def _resolve_device(spec):
    """'auto'/None → 'cpu'; otherwise pass through.
    DEFAULT IS CPU: the RBF surrogate solves an ill-conditioned cubic-RBF saddle system whose
    GPU (cuSOLVER) solve returns garbage on near-singular inputs (cond >~1e18 in the 2nd-stage
    capped regime) — same matrix gives GPU ||coeff||~1e39 rho<0 vs CPU rho~1.0. CPU LAPACK is
    stable and the fit is not the bottleneck (model evals dominate), so default to CPU. Pass an
    explicit 'cuda'/'cuda:N' to opt back into GPU (still guarded by RBF's CPU-lstsq fallback)."""
    if spec is None or spec == 'auto':
        return 'cpu'
    return spec


def _strip_transform(model):
    """If ``model`` carries a target-transform prefix, return
    ``(transform_name, base_model)``; else ``(None, model)``."""
    if model == 'sqrty_gp':              # legacy alias
        return 'sqrt', 'ard_gp'
    for prefix, transform in _TRANSFORM_PREFIXES.items():
        if model.startswith(prefix):
            return transform, model[len(prefix):]
    return None, model


def all_surrogates():
    """Enumerate (base_predictor) ∪ (transform × base) names — used by
    ``post_search.py``'s argparse ``choices``. Dedupes against the
    legacy ``sqrty_gp`` alias which would otherwise collide with the
    base-name ``gp``'s sqrt variant."""
    seen = set()
    out = []
    for n in BASE_PREDICTORS:
        if n not in seen:
            seen.add(n); out.append(n)
    for prefix in _TRANSFORM_PREFIXES:
        for base in BASE_PREDICTORS:
            n = f'{prefix}{base}'
            if n not in seen:
                seen.add(n); out.append(n)
    if 'sqrty_gp' not in seen:           # legacy alias
        out.append('sqrty_gp')
    return tuple(out)


def _build_base(model, inputs, targets, device='auto', **kwargs):
    """Build + fit a base predictor (no target transform)."""
    device = _resolve_device(device)
    if model == 'rbf':
        from predictor.rbf import RBF
        predictor = RBF(device=device, **kwargs)
        predictor.fit(inputs, targets)

    elif model == 'carts':
        from predictor.carts import CART
        predictor = CART(n_tree=5000)
        predictor.fit(inputs, targets)

    elif model == 'gp':
        from predictor.gp import GP
        predictor = GP()
        predictor.fit(inputs, targets)

    elif model == 'mlp':
        from predictor.mlp import MLP
        predictor = MLP(n_feature=inputs.shape[1], device=device)
        predictor.fit(x=inputs, y=targets, device=device)

    elif model == 'as':
        from predictor.adaptive_switching import AdaptiveSwitching
        predictor = AdaptiveSwitching()
        predictor.fit(inputs, targets)

    elif model == 'ard_gp':
        from predictor.ard_gp import ARDGP
        predictor = ARDGP(kernel=kwargs.get('ard_kernel', 'matern32'),
                          n_restarts=kwargs.get('gp_n_restarts', 10),
                          device=device)
        predictor.fit(inputs, targets)

    elif model == 'badd_quad':
        # Bayesian additive quadratic. Closed-form, deterministic, gives
        # predict_variance for AL. JSD-target friendly (additive prior
        # matches per-axis JSD decomposition). See predictor/badd_quad.py.
        from predictor.badd_quad import BayesianAdditiveQuadratic
        predictor = BayesianAdditiveQuadratic(
            alpha=kwargs.get('badd_alpha', 1e-3),
            optimize_alpha=kwargs.get('badd_optimize_alpha', False))
        predictor.fit(inputs, targets)

    elif model == 'gam':
        # Hinge-spline monotone GAM (per-axis f_i monotone-increasing).
        # Hard monotonicity constraint via NNLS — safe to extrapolate at
        # the quantile-anchored extremes. See predictor/gam.py.
        from predictor.gam import MonotoneGAM
        predictor = MonotoneGAM(
            n_knots=kwargs.get('gam_n_knots', 6),
            with_interactions=kwargs.get('gam_interactions', True))
        predictor.fit(inputs, targets)

    else:
        raise NotImplementedError(f"unknown base predictor {model!r}")

    return predictor


def get_predictor(model, inputs, targets, device='auto', **kwargs):
    """Build, fit, and return a predictor.

    Handles target-transform prefixes (``sqrty_``, ``logy_``,
    ``logity_``) by transforming ``targets`` before fitting the base
    predictor and wrapping the result in ``TargetTransformPredictor``
    (which undoes the transform at predict time).

    ``device='auto'`` (default) resolves to ``'cpu'`` (GPU cuSOLVER is
    unreliable on the ill-conditioned RBF saddle system — see
    ``_resolve_device``). Pass an explicit ``'cuda'`` to opt into GPU.
    """
    import numpy as np
    device = _resolve_device(device)
    transform, base_name = _strip_transform(model)

    if transform is None:
        return _build_base(base_name, inputs, targets, device=device, **kwargs)

    # Apply forward transform, fit the base on transformed Y, wrap.
    from predictor.target_transform import TargetTransformPredictor, TRANSFORMS
    fwd, _, _ = TRANSFORMS[transform]
    y_t = fwd(np.asarray(targets, dtype=np.float64))
    base = _build_base(base_name, inputs, y_t, device=device, **kwargs)
    wrapped = TargetTransformPredictor(base, transform=transform)
    wrapped._fitted = True               # base already fit; mark ready
    return wrapped
