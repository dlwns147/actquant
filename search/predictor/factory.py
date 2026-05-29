def get_predictor(model, inputs, targets, device='cpu', **kwargs):

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

    elif model == 'sqrty_gp':
        # sqrt-Y transformed ARD-GP. Hellinger-style metric on JSD →
        # better-conditioned GP MLE near Y≈0. See predictor/sqrty_gp.py.
        from predictor.sqrty_gp import SqrtYARDGP
        predictor = SqrtYARDGP(
            kernel=kwargs.get('ard_kernel', 'matern32'),
            n_restarts=kwargs.get('gp_n_restarts', 10),
            device=device)
        predictor.fit(inputs, targets)

    else:
        raise NotImplementedError

    return predictor

