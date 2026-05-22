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

    else:
        raise NotImplementedError

    return predictor

