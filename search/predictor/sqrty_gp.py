"""Backward-compat alias: ``sqrty_gp`` = ``sqrt`` transform over ``ard_gp``.

This module used to host a hardcoded sqrt(Y)+ARDGP wrapper. That logic
moved into the generic ``predictor.target_transform.TargetTransformPredictor``
which can wrap *any* base predictor (rbf, mlp, badd_quad, gam, …) so the
sqrt transform isn't tied to ARDGP.

Direct imports of ``SqrtYARDGP`` keep working via this thin shim. New
callers should use the factory: ``get_predictor('sqrty_<base>', ...)``,
e.g. ``sqrty_rbf``, ``sqrty_ard_gp``, ``sqrty_mlp``, ``sqrty_badd_quad``,
``sqrty_gam``.
"""

from predictor.ard_gp import ARDGP
from predictor.target_transform import TargetTransformPredictor


def SqrtYARDGP(kernel='matern32', n_restarts=10, device='cpu', max_iter=200,
               with_noise=True, **_unused):
    """Construct ``sqrt(Y) ∘ ARDGP``. Returned object exposes the same
    ``fit/predict`` interface as the previous standalone class."""
    base = ARDGP(kernel=kernel, with_noise=with_noise,
                 n_restarts=n_restarts, device=device, max_iter=max_iter)
    return TargetTransformPredictor(base, transform='sqrt')
