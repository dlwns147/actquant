"""Bayesian Additive Quadratic surrogate — closed-form, JSD-target friendly.

Linear regression on the additive-quadratic feature map
``[1, x_i, x_i^2, x_i*x_j]`` with a Gaussian prior on the coefficients
(ridge-equivalent). Designed for the (per-axis JSD)→(combined JSD)
mapping in actquant's sample_surrogate/post_search pipeline, where the
ground truth is "additive with small pair-wise interactions" — see
``MEMORY.md`` project_actquant_analysis. The full feature dimension is
``1 + K + K + K*(K-1)/2`` (= 10 for K=3) so N=50 already gives a safe
5:1 sample/parameter ratio.

Why this fits JSD better than RBF/GAM at small N:
* parameters are bounded by the structure of JSD's analytic decomposition
  (additive baseline + per-axis curvature + axis pair interactions);
* posterior is closed-form so fit time is microseconds and there are no
  hyper-parameters to tune (one ridge ``alpha`` only);
* posterior variance ``predict_variance`` is exposed for active learning
  acquisition functions (UCB / σ × coverage).

Marginal-likelihood (type-II MLE) optimization of ``alpha`` is available
via ``optimize_alpha=True``; the default keeps a small fixed ``alpha`` so
behaviour stays deterministic for callers that don't ask for it.

Interface matches the other predictors (numpy in/out, ``fit(X, y) -> self``,
``predict(X) -> 1D ndarray``).
"""

import numpy as np


def _features(X):
    """Additive-quadratic basis: [1, x_i, x_i^2, x_i*x_j].

    Order: 1 column for the intercept, then K linear terms, K squared
    terms, then K*(K-1)/2 pair-wise interaction terms in (i, j) with i<j.
    Returned matrix is (n, 1 + 2K + K(K-1)/2).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n, K = X.shape
    parts = [np.ones((n, 1)), X, X ** 2]
    for i in range(K):
        for j in range(i + 1, K):
            parts.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(parts)


def _n_features(K):
    return 1 + 2 * K + K * (K - 1) // 2


class BayesianAdditiveQuadratic:
    """Closed-form Bayesian linear regression on additive-quadratic features.

    The posterior over coefficients β under prior N(0, prior_sigma² I) and
    Gaussian likelihood N(Φβ, noise_sigma² I) is

        Σ = (Φᵀ Φ / σ² + I / τ²)⁻¹
        μ = Σ Φᵀ y / σ²

    Predictive: ŷ(x*) = φ(x*) μ;  Var ŷ(x*) = φ(x*) Σ φ(x*)ᵀ (+ σ² for the
    *observation* variance, which is what AL acquisition usually wants).

    The intercept column is *not* shrunk (large prior_sigma) so that the
    posterior mean is unbiased when y is not centred. All other columns
    share ``prior_sigma`` — the additive-quadratic feature map already
    encodes the right structural prior, so an isotropic Gaussian on the
    remaining 9 coefficients is appropriate.
    """

    def __init__(self, alpha=1e-3, noise_sigma=None, prior_sigma=None,
                 optimize_alpha=False, alpha_grid=None,
                 standardize_features=True):
        # ``alpha`` is the ridge regularizer = σ²/τ². If ``noise_sigma``
        # and ``prior_sigma`` are both given, they take precedence.
        self.alpha = float(alpha)
        self.noise_sigma = noise_sigma
        self.prior_sigma = prior_sigma
        self.optimize_alpha = bool(optimize_alpha)
        # log-spaced grid for type-II MLE; covers 5 orders of magnitude.
        self.alpha_grid = (alpha_grid if alpha_grid is not None
                           else np.logspace(-6, 1, 25))
        self.standardize_features = bool(standardize_features)
        self.name = 'badd_quad'

        # populated by fit()
        self._mu = None              # posterior mean over β
        self._Sigma = None           # posterior covariance over β
        self._sigma2 = None          # noise variance (observation σ²)
        self._feat_mean = None
        self._feat_std = None
        self._K = None
        self._fitted = False

    # ----- feature scaling -------------------------------------------------
    def _scale_features(self, Phi):
        """Standardize non-intercept feature columns to unit variance.

        Keeps the intercept (column 0) untouched. Standardization makes a
        single isotropic Gaussian prior reasonable across heterogeneous
        feature magnitudes (linear vs squared vs product terms differ by
        orders of magnitude on the [0, 1] domain).
        """
        if self._feat_mean is None:
            mean = Phi.mean(axis=0).copy()
            std = Phi.std(axis=0).copy()
            mean[0] = 0.0                 # intercept: do not centre
            std[0] = 1.0                  # intercept: do not scale
            std = np.where(std < 1e-12, 1.0, std)
            self._feat_mean, self._feat_std = mean, std
        return (Phi - self._feat_mean) / self._feat_std

    # ----- type-II MLE for alpha -------------------------------------------
    def _marginal_log_likelihood(self, Phi, y, alpha):
        """Closed-form log p(y | α) under N(0, I/α) prior on non-intercept β.

        Used by ``optimize_alpha`` to pick alpha by Empirical Bayes. The
        intercept column is given a near-flat prior (small α) so it's not
        penalized.
        """
        n, p = Phi.shape
        # per-feature prior precision: small for intercept, alpha for rest
        a_diag = np.full(p, alpha, dtype=np.float64)
        a_diag[0] = 1e-8
        A = Phi.T @ Phi + np.diag(a_diag)
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return -np.inf
        rhs = Phi.T @ y
        mu = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        resid = y - Phi @ mu
        # σ² point estimate (closed-form max wrt σ²)
        sigma2 = float(np.dot(resid, resid) + (a_diag * mu * mu).sum()) / n
        sigma2 = max(sigma2, 1e-12)
        # log evidence (drops constants, keeps α-dependent terms)
        log_det_A = 2.0 * np.log(np.diag(L)).sum()
        log_det_prior = float(np.log(a_diag).sum())     # ∑ log α_i
        loglik = (0.5 * log_det_prior
                  - 0.5 * log_det_A
                  - 0.5 * n * np.log(sigma2)
                  - 0.5 * n)
        return loglik

    # ----- fit -------------------------------------------------------------
    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self._K = X.shape[1]

        Phi = _features(X)
        if self.standardize_features:
            Phi = self._scale_features(Phi)

        # Resolve prior / noise. Default: ridge with alpha = σ²/τ².
        if self.noise_sigma is not None and self.prior_sigma is not None:
            sigma2 = float(self.noise_sigma) ** 2
            tau2 = float(self.prior_sigma) ** 2
            alpha = sigma2 / max(tau2, 1e-30)
        else:
            alpha = self.alpha
            if self.optimize_alpha:
                # type-II MLE on a log-grid
                best, best_ll = alpha, -np.inf
                for a in self.alpha_grid:
                    ll = self._marginal_log_likelihood(Phi, y, a)
                    if ll > best_ll:
                        best_ll, best = ll, a
                alpha = float(best)
            sigma2 = None       # estimated below

        n, p = Phi.shape
        # Non-intercept ridge: intercept gets near-flat prior so it absorbs
        # the mean without bias.
        a_diag = np.full(p, alpha, dtype=np.float64)
        a_diag[0] = 1e-8
        A = Phi.T @ Phi + np.diag(a_diag)
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            # Add jitter and retry; falls back to lstsq as last resort.
            A = A + 1e-6 * np.eye(p)
            try:
                L = np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                mu, *_ = np.linalg.lstsq(Phi, y, rcond=None)
                self._mu = mu
                self._Sigma = np.eye(p) * 1e-3
                self._sigma2 = float(np.var(y - Phi @ mu))
                self._fitted = True
                return self

        rhs = Phi.T @ y
        mu = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        # Estimate σ² from residuals if not user-provided.
        if sigma2 is None:
            resid = y - Phi @ mu
            sigma2 = float(np.dot(resid, resid) / max(n - p, 1))
            sigma2 = max(sigma2, 1e-12)
        # Posterior covariance = σ² A⁻¹
        Ainv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(p)))
        self._mu = mu
        self._Sigma = sigma2 * Ainv
        self._sigma2 = sigma2
        self._fitted = True
        return self

    # ----- predict ---------------------------------------------------------
    def predict(self, X):
        assert self._fitted, "BayesianAdditiveQuadratic not fitted"
        Phi = _features(X)
        if self.standardize_features:
            Phi = (Phi - self._feat_mean) / self._feat_std
        return Phi @ self._mu

    def predict_variance(self, X, include_noise=True):
        """Predictive variance for active learning acquisition.

        Returns σ²(x) = φ(x)ᵀ Σ φ(x) (+ σ²_obs if ``include_noise``).
        """
        assert self._fitted, "BayesianAdditiveQuadratic not fitted"
        Phi = _features(X)
        if self.standardize_features:
            Phi = (Phi - self._feat_mean) / self._feat_std
        v = np.einsum('ij,jk,ik->i', Phi, self._Sigma, Phi)
        if include_noise:
            v = v + self._sigma2
        return np.clip(v, 0.0, None)
