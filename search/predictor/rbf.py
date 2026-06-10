"""Radial Basis Function interpolant — pure PyTorch, GPU/CPU selectable.

This is a drop-in replacement for the previous pySOT-backed ``RBF``. The math
is a faithful port of ``pySOT.surrogate.RBFInterpolant`` (v0.3.3) so search
behaviour and the existing ``.stats`` archives stay comparable:

* domain is scaled to the unit box with the supplied ``lb``/``ub``
  (``to_unit_box``: ``(x - lb) / (ub - lb)``);
* the same saddle-point system is solved::

      [ 0    P^T ] [ lambda ]   [ 0  ]
      [ P  Phi+eI ] [   c    ] = [ fX ]

  with regularization ``eta = 1e-6`` added to the kernel block only
  (matching pySOT's code, not its docstring);
* identity output transformation (pySOT's default);
* kernels ``cubic`` (r^3), ``tps`` (r^2 log r), ``linear`` (r);
  tails ``linear`` ([1, x]) and ``constant`` ([1]).

The only behavioural additions over pySOT are robustness guards that pySOT
would otherwise crash on (degenerate ``ub == lb`` dims, missing bounds,
singular system) — these only fire on inputs pySOT could not handle anyway.

`device` picks where the linear algebra runs: ``'cpu'``, ``'cuda'``,
``'cuda:N'``, or a ``torch.device``. ``'cuda'`` silently falls back to CPU if
no GPU is visible. Computation is done in float64 for conditioning; inputs and
outputs stay numpy so every caller (search.py / post_search.py /
adaptive_switching) is unaffected.
"""

import numpy as np
import torch

_EPS = float(np.finfo(np.float64).eps)


def _resolve_device(device):
    if isinstance(device, torch.device):
        return device
    if device is None or device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        print("[RBF] CUDA requested but not available — falling back to CPU")
        return torch.device('cpu')
    return dev


class RBF:
    """ Radial Basis Function """

    def __init__(self, kernel='cubic', tail='linear', lb=None, ub=None,
                 eta=1e-6, device='auto', predict_batch=None,
                 predict_mem_budget=512 * 1024 ** 2):
        self.kernel = kernel
        self.tail = tail
        self.name = 'rbf'
        self.model = None
        self.lb = lb
        self.ub = ub
        self.eta = eta
        self.device = _resolve_device(device)
        # query points are predicted in row-chunks so peak memory stays
        # bounded no matter how large the candidate set is (the m x n_train
        # kernel matrix is the dominant term). predict_batch fixes the chunk
        # size; None = derive it from predict_mem_budget and n_train.
        self.predict_batch = predict_batch
        self.predict_mem_budget = predict_mem_budget

        if kernel not in ('cubic', 'tps', 'linear'):
            raise NotImplementedError("unknown RBF kernel '%s'" % kernel)
        if tail not in ('linear', 'constant'):
            raise NotImplementedError("unknown RBF tail '%s'" % tail)

        # internal state (filled by fit)
        self._U = None        # unit-box-scaled training points (n, d)
        self._coeff = None     # [tail lambda ; rbf c]  (ntail + n, 1)
        self._ntail = None
        self._lb_t = None
        self._ub_t = None

    # ----- kernel / tail on a distance (or design) matrix -------------------
    def _kernel_eval(self, dists):
        if self.kernel == 'cubic':
            return dists ** 3
        if self.kernel == 'linear':
            return dists
        # thin plate spline: r^2 log r, with r floored at eps (pySOT behaviour)
        d = dists.clamp_min(_EPS)
        return (d ** 2) * torch.log(d)

    def _tail_eval(self, U):
        ones = torch.ones((U.shape[0], 1), dtype=U.dtype, device=U.device)
        if self.tail == 'constant':
            return ones
        return torch.cat((ones, U), dim=1)

    @property
    def _ntail_dim(self):
        # set after fit; kept as a helper for clarity
        return self._ntail

    # ----- domain scaling --------------------------------------------------
    def _to_unit_box(self, X):
        # X: torch tensor (m, d) on self.device, float64
        denom = self._ub_t - self._lb_t
        # pySOT divides straight through; guard the degenerate single-option
        # dims (ub == lb) that would otherwise produce inf/nan.
        safe = torch.where(denom.abs() < _EPS,
                           torch.ones_like(denom), denom)
        return (X - self._lb_t) / safe

    # ----- fit / predict ---------------------------------------------------
    def fit(self, train_data, train_label):
        X = np.atleast_2d(np.asarray(train_data, dtype=np.float64))
        y = np.asarray(train_label, dtype=np.float64).reshape(-1, 1)
        n, d = X.shape

        if self.lb is None or self.ub is None:
            # pySOT would crash here; fall back to data extent so the
            # surrogate still works when bounds are not supplied.
            lb = X.min(axis=0)
            ub = X.max(axis=0)
        else:
            lb = np.asarray(self.lb, dtype=np.float64).reshape(-1)
            ub = np.asarray(self.ub, dtype=np.float64).reshape(-1)

        dev = self.device
        self._lb_t = torch.as_tensor(lb, dtype=torch.float64, device=dev)
        self._ub_t = torch.as_tensor(ub, dtype=torch.float64, device=dev)

        Xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
        yt = torch.as_tensor(y, dtype=torch.float64, device=dev)
        U = self._to_unit_box(Xt)
        self._U = U

        P = self._tail_eval(U)               # (n, ntail)
        ntail = P.shape[1]
        self._ntail = ntail
        assert n >= ntail, (
            "RBF needs at least ntail=%d points, got %d" % (ntail, n))

        D = torch.cdist(U, U)                 # (n, n) euclidean
        Phi = self._kernel_eval(D)
        Phi = Phi + self.eta * torch.eye(n, dtype=torch.float64, device=dev)

        zero_tt = torch.zeros((ntail, ntail), dtype=torch.float64, device=dev)
        A = torch.cat((
            torch.cat((zero_tt, P.t()), dim=1),   # [0 , P^T]
            torch.cat((P, Phi), dim=1),           # [P , Phi+eI]
        ), dim=0)                                 # (ntail+n, ntail+n)

        rhs = torch.cat((
            torch.zeros((ntail, 1), dtype=torch.float64, device=dev),
            yt,
        ), dim=0)                                 # (ntail+n, 1)

        try:
            coeff = torch.linalg.solve(A, rhs)
        except RuntimeError:
            # ill-conditioned / singular — least-squares fallback
            coeff = torch.linalg.lstsq(A, rhs).solution
        self._coeff = coeff
        self.model = True  # sentinel so predict()'s assert mirrors old API
        return self

    def _chunk_rows(self, m, n):
        if self.predict_batch is not None:
            return max(1, int(self.predict_batch))
        # peak is dominated by the (chunk x n) distance + kernel tensors
        # (float64, ~2 live copies). Size the chunk to the memory budget.
        rows = int(self.predict_mem_budget // max(1, n * 8 * 4))
        return int(min(m, max(4096, rows)))

    def predict(self, test_data):
        assert self.model is not None, \
            "RBF model does not exist, call fit to obtain rbf model first"

        Xq = np.atleast_2d(np.asarray(test_data, dtype=np.float64))
        dev = self.device
        ntail = self._ntail
        n = self._U.shape[0]
        m = Xq.shape[0]
        step = self._chunk_rows(m, n)

        out = np.empty((m, 1), dtype=np.float64)
        for s in range(0, m, step):
            e = min(m, s + step)
            Vt = self._to_unit_box(
                torch.as_tensor(Xq[s:e], dtype=torch.float64, device=dev))
            ds = torch.cdist(Vt, self._U)              # (chunk, n)
            rbf_part = self._kernel_eval(ds) @ self._coeff[ntail:]
            tail_part = self._tail_eval(Vt) @ self._coeff[:ntail]
            out[s:e] = (rbf_part + tail_part).detach().cpu().numpy()
        return out
