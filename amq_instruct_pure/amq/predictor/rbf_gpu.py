import numpy as np
import torch


class RBFGPU:
    """Radial Basis Function interpolant on GPU.

    Mirrors pySOT's RBFInterpolant formulation (cubic/TPS kernel,
    linear/constant tail, eta-regularized block system) using PyTorch
    in float64 for numerical parity with the CPU version.

    Persistent state (`_coef`, `_X_unit`) is kept on CPU. GPU memory is
    only held for the duration of a fit/predict call and is released +
    cache-emptied before returning, so other processes sharing the device
    (e.g. vLLM workers) are not starved.
    """

    def __init__(self, kernel='cubic', tail='linear', lb=None, ub=None,
                 eta=1e-6, device=None):
        if kernel not in ('cubic', 'tps'):
            raise NotImplementedError("unknown RBF kernel")
        if tail not in ('linear', 'constant'):
            raise NotImplementedError("unknown RBF tail")

        self.kernel = kernel
        self.tail = tail
        self.name = 'rbf_gpu'
        self.lb = lb
        self.ub = ub
        self.eta = eta
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self._X_unit = None   # CPU tensor, (n, d)
        self._coef = None     # CPU tensor, (ntail + n, 1)
        self._ntail = None
        self._dim = None

    def _release_gpu(self):
        if not str(self.device).startswith('cuda'):
            return
        # cuBLAS keeps a per-stream workspace (~8 MB on H100) that empty_cache()
        # alone does not free. Clear it so a co-tenant on this GPU sees 0 MB.
        if hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
            torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()

    def _to_unit(self, X, device):
        lb = torch.as_tensor(self.lb, dtype=torch.float64, device=device).reshape(1, -1)
        ub = torch.as_tensor(self.ub, dtype=torch.float64, device=device).reshape(1, -1)
        return (X - lb) / (ub - lb)

    def _kernel_eval(self, D):
        if self.kernel == 'cubic':
            return D ** 3
        eps = torch.finfo(D.dtype).eps
        D = torch.clamp(D, min=eps)
        return (D ** 2) * torch.log(D)

    def _tail_eval(self, X):
        n = X.shape[0]
        ones = torch.ones((n, 1), dtype=X.dtype, device=X.device)
        if self.tail == 'linear':
            return torch.cat([ones, X], dim=1)
        return ones

    def _tail_dim(self, dim):
        return 1 + dim if self.tail == 'linear' else 1

    def fit(self, train_data, train_label):
        X_np = np.asarray(train_data)
        y_np = np.asarray(train_label).reshape(-1, 1)

        n, dim = X_np.shape
        self._dim = dim
        self._ntail = self._tail_dim(dim)
        ntail = self._ntail

        if self.lb is None or self.ub is None:
            raise ValueError("RBFGPU requires lb and ub for unit-box scaling")
        if n < ntail:
            raise ValueError(f"need at least {ntail} points, got {n}")

        try:
            X = torch.as_tensor(X_np, dtype=torch.float64, device=self.device)
            y = torch.as_tensor(y_np, dtype=torch.float64, device=self.device)

            X_unit = self._to_unit(X, self.device)
            D = torch.cdist(X_unit, X_unit, p=2.0)
            Phi = self._kernel_eval(D) + self.eta * torch.eye(n, dtype=torch.float64, device=self.device)
            P = self._tail_eval(X_unit)

            nact = ntail + n
            A = torch.zeros((nact, nact), dtype=torch.float64, device=self.device)
            A[:ntail, ntail:] = P.T
            A[ntail:, :ntail] = P
            A[ntail:, ntail:] = Phi

            rhs = torch.zeros((nact, 1), dtype=torch.float64, device=self.device)
            rhs[ntail:, :] = y

            coef = torch.linalg.solve(A, rhs)

            self._coef = coef.detach().cpu()
            self._X_unit = X_unit.detach().cpu()

            del X, y, X_unit, D, Phi, P, A, rhs, coef
        finally:
            self._release_gpu()

    def predict(self, test_data):
        assert self._coef is not None, "RBFGPU model does not exist, call fit first"

        X_np = np.atleast_2d(np.asarray(test_data))

        try:
            X = torch.as_tensor(X_np, dtype=torch.float64, device=self.device)
            X_unit = self._to_unit(X, self.device)

            X_train = self._X_unit.to(self.device, non_blocking=True)
            coef = self._coef.to(self.device, non_blocking=True)

            ds = torch.cdist(X_unit, X_train, p=2.0)
            K = self._kernel_eval(ds)
            P = self._tail_eval(X_unit)

            ntail = self._ntail
            lam = coef[:ntail]
            c = coef[ntail:]

            out = K @ c + P @ lam
            result = out.detach().cpu().numpy()

            del X, X_unit, X_train, coef, ds, K, P, lam, c, out
            return result
        finally:
            self._release_gpu()
