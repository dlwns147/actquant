"""ARD Gaussian Process surrogate — pure PyTorch, GPU/CPU selectable.

Pure-PyTorch reimplementation of the previous sklearn
``GaussianProcessRegressor``-backed ``ARDGP`` (per-dimension lengthscales),
used by ``post_search.py --surrogate ard_gp`` (ported originally from
``analysis/v5/_common.fit_ard_gp``). It keeps the same modelling choices so
results stay comparable:

* kernel ``k = c * core (+ white)`` where ``c`` is a ConstantKernel-style
  output scale and ``white`` is a WhiteKernel-style noise term;
* ``normalize_y=True`` (targets standardized before fitting);
* hyper-parameters fit by maximizing the exact log marginal likelihood with
  ``n_restarts`` random restarts (L-BFGS), bounds matching the old sklearn
  kernel (``constant`` 1e-4..1e2, ``lengthscale`` 1e-4..1e4,
  ``white`` 1e-9..1e-2, ``rq alpha`` 1e-2..1e2);
* a small jitter ``alpha`` on the diagonal (1e-8 with noise, else 1e-10).

Cores: ``rbf``, ``matern52``, ``matern32`` (ARD lengthscales), and ``rq``
(isotropic, single lengthscale + alpha) — same set the sklearn version
accepted. ``.predict`` returns the posterior mean only (1-D numpy), exactly
like the old ``sklearn`` ``.predict`` so callers are unaffected.

`device`: ``'cpu'``, ``'cuda'``, ``'cuda:N'`` or a ``torch.device``;
``'cuda'`` falls back to CPU when no GPU is visible. Math runs in float64.
"""

import numpy as np
import torch


def _resolve_device(device):
    if isinstance(device, torch.device):
        return device
    if device is None or device == 'auto':
        # predictor default is CPU (see predictor/factory._resolve_device); pass 'cuda' to opt in
        return torch.device('cpu')
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        print("[ARDGP] CUDA requested but not available — falling back to CPU")
        return torch.device('cpu')
    return dev


class ARDGP:
    def __init__(self, kernel='matern32', with_noise=True, n_restarts=10,
                 device='auto', max_iter=200, predict_batch=None,
                 predict_mem_budget=512 * 1024 ** 2):
        self.kernel = kernel
        self.with_noise = with_noise
        self.n_restarts = n_restarts
        self.device = _resolve_device(device)
        self.max_iter = max_iter
        # cross-covariance Ks is (chunk x n_train); chunk the query set so
        # peak memory is bounded regardless of candidate-set size.
        self.predict_batch = predict_batch
        self.predict_mem_budget = predict_mem_budget
        if kernel not in ('rbf', 'matern52', 'matern32', 'rq'):
            raise ValueError("unknown ard_kernel '%s'" % kernel)
        self._fitted = False

    # ----- kernel core -----------------------------------------------------
    def _core(self, A, B, ls, rq_alpha):
        """core(A, B) with ARD lengthscales `ls` (and `rq_alpha` for rq)."""
        if self.kernel == 'rq':
            # isotropic: single lengthscale (ls[0]) + alpha
            d2 = torch.cdist(A, B) ** 2
            l2 = ls[0] ** 2
            return (1.0 + d2 / (2.0 * rq_alpha * l2)) ** (-rq_alpha)
        # ARD scaling
        As = A / ls
        Bs = B / ls
        if self.kernel == 'rbf':
            d2 = torch.cdist(As, Bs) ** 2
            return torch.exp(-0.5 * d2)
        r = torch.cdist(As, Bs).clamp_min(1e-12)
        if self.kernel == 'matern32':
            c = np.sqrt(3.0)
            return (1.0 + c * r) * torch.exp(-c * r)
        # matern52
        c = np.sqrt(5.0)
        return (1.0 + c * r + (5.0 / 3.0) * r ** 2) * torch.exp(-c * r)

    # ----- (un)constrained hyper-parameters --------------------------------
    # raw params live in R; mapped to bounded positive values via sigmoid.
    _BOUNDS = None  # set per-fit (depends on dim / kernel)

    def _unpack(self, raw):
        """raw -> dict of positive, bounded hyper-parameters."""
        out = {}
        i = 0
        for key, (lo, hi) in self._BOUNDS:
            n = self._PSIZE[key]
            seg = raw[i:i + n]
            lo_t = torch.as_tensor(lo, dtype=raw.dtype, device=raw.device)
            hi_t = torch.as_tensor(hi, dtype=raw.dtype, device=raw.device)
            # log-uniform bounded mapping (matches sklearn's log-space search)
            frac = torch.sigmoid(seg)
            out[key] = torch.exp(torch.log(lo_t)
                                 + frac * (torch.log(hi_t) - torch.log(lo_t)))
            i += n
        return out

    def _nll(self, raw, X, y):
        hp = self._unpack(raw)
        n = X.shape[0]
        c = hp['constant']
        ls = hp['lengthscale']
        rq_a = hp.get('rq_alpha', None)
        K = c * self._core(X, X, ls, rq_a)
        diag = self._alpha
        if self.with_noise:
            diag = diag + hp['white']
        K = K + diag * torch.eye(n, dtype=X.dtype, device=X.device)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y, L)
        # 0.5 y^T K^-1 y + sum(log diag L) + n/2 log 2pi
        nll = 0.5 * (y * alpha).sum()
        nll = nll + torch.log(torch.diagonal(L)).sum()
        nll = nll + 0.5 * n * np.log(2.0 * np.pi)
        return nll

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n, d = X.shape
        dev = self.device

        # standardize targets (normalize_y=True)
        self._ymu = float(y.mean())
        self._ystd = float(y.std()) or 1.0
        yt = torch.as_tensor((y - self._ymu) / self._ystd,
                             dtype=torch.float64, device=dev)
        Xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
        self._X = Xt

        self._alpha = 1e-8 if self.with_noise else 1e-10

        # parameter layout + bounds (mirror old sklearn kernel)
        bounds = [('constant', (1e-4, 1e2)),
                  ('lengthscale', (1e-4, 1e4))]
        psize = {'constant': 1,
                 'lengthscale': (1 if self.kernel == 'rq' else d)}
        if self.kernel == 'rq':
            bounds.append(('rq_alpha', (1e-2, 1e2)))
            psize['rq_alpha'] = 1
        if self.with_noise:
            bounds.append(('white', (1e-9, 1e-2)))
            psize['white'] = 1
        self._BOUNDS = bounds
        self._PSIZE = psize
        n_param = sum(psize[k] for k, _ in bounds)

        best_nll = float('inf')
        best_raw = None
        gen = torch.Generator(device='cpu').manual_seed(0)
        n_starts = max(1, self.n_restarts) + 1
        for s in range(n_starts):
            if s == 0:
                init = torch.zeros(n_param, dtype=torch.float64)
            else:
                init = torch.randn(n_param, generator=gen,
                                   dtype=torch.float64) * 1.5
            raw = init.to(dev).detach().clone().requires_grad_(True)
            opt = torch.optim.LBFGS([raw], max_iter=self.max_iter,
                                    line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad()
                loss = self._nll(raw, Xt, yt)
                loss.backward()
                return loss

            try:
                opt.step(closure)
                with torch.no_grad():
                    val = float(self._nll(raw, Xt, yt))
            except RuntimeError:
                continue
            if np.isfinite(val) and val < best_nll:
                best_nll = val
                best_raw = raw.detach().clone()

        if best_raw is None:
            raise RuntimeError("ARDGP: all hyper-parameter restarts failed")

        self._raw = best_raw
        with torch.no_grad():
            hp = self._unpack(best_raw)
            K = hp['constant'] * self._core(Xt, Xt, hp['lengthscale'],
                                            hp.get('rq_alpha'))
            diag = self._alpha + (hp['white'] if self.with_noise else 0.0)
            K = K + diag * torch.eye(n, dtype=torch.float64, device=dev)
            self._L = torch.linalg.cholesky(K)
            self._alpha_vec = torch.cholesky_solve(yt, self._L)
            self._hp = hp
        self._fitted = True
        return self

    def _chunk_rows(self, m, n):
        if self.predict_batch is not None:
            return max(1, int(self.predict_batch))
        rows = int(self.predict_mem_budget // max(1, n * 8 * 4))
        return int(min(m, max(4096, rows)))

    def predict(self, X, return_std=False):
        """Posterior mean (1-D numpy). With ``return_std=True`` also returns
        the posterior standard deviation of the LATENT function (observation
        noise NOT added), on the original (un-standardized) Y scale — used by
        active-learning acquisitions (ALM / UCB)."""
        assert self._fitted, "ARDGP not fitted; call fit() first"
        Xq = np.atleast_2d(np.asarray(X, dtype=np.float64))
        dev = self.device
        hp = self._hp
        n = self._X.shape[0]
        m = Xq.shape[0]
        step = self._chunk_rows(m, n)

        out = np.empty(m, dtype=np.float64)
        std = np.empty(m, dtype=np.float64) if return_std else None
        with torch.no_grad():
            for s in range(0, m, step):
                e = min(m, s + step)
                Xt = torch.as_tensor(Xq[s:e], dtype=torch.float64, device=dev)
                Ks = hp['constant'] * self._core(Xt, self._X,
                                                 hp['lengthscale'],
                                                 hp.get('rq_alpha'))
                mean = (Ks @ self._alpha_vec).reshape(-1)   # standardized
                out[s:e] = mean.detach().cpu().numpy()
                if return_std:
                    # prior var = k(x*,x*) = constant (core diag == 1 for
                    # rbf/matern/rq); posterior var = prior - ks^T K^-1 ks.
                    v = torch.cholesky_solve(Ks.t(), self._L)     # (n, chunk)
                    qf = (Ks * v.t()).sum(dim=1)                  # ks^T K^-1 ks
                    pv = (hp['constant'] - qf).clamp_min(0.0)
                    std[s:e] = pv.sqrt().detach().cpu().numpy()
        if return_std:
            return out * self._ystd + self._ymu, std * self._ystd
        return out * self._ystd + self._ymu

    def predict_cov(self, X):
        """Full posterior covariance over the query set X (un-standardized Y
        scale), latent (no observation noise). O(m^2 + m·n); the caller must
        keep m bounded (used by the IMSE/ALC acquisition on a capped pool)."""
        assert self._fitted, "ARDGP not fitted; call fit() first"
        Xq = np.atleast_2d(np.asarray(X, dtype=np.float64))
        dev = self.device
        hp = self._hp
        with torch.no_grad():
            Xt = torch.as_tensor(Xq, dtype=torch.float64, device=dev)
            Kaa = hp['constant'] * self._core(Xt, Xt, hp['lengthscale'],
                                              hp.get('rq_alpha'))
            Ks = hp['constant'] * self._core(Xt, self._X, hp['lengthscale'],
                                             hp.get('rq_alpha'))   # (m, n)
            v = torch.cholesky_solve(Ks.t(), self._L)              # (n, m)
            cov = Kaa - Ks @ v
        return (cov.detach().cpu().numpy()) * (self._ystd ** 2)
