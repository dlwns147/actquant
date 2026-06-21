"""utils/joint_search.py — reusable pieces for the 2nd-stage JOINT (W × eff_kvbits)
NAS (second_search.py). Built on the SAME LlamaSearchSpace encoding as search.py:

  genome = ss.encode(arch): flat int (n_linear+4)*n_block = [w..., k..., v..., k_dim..., v_dim...].
  W-block = [0:nw] (nw = n_linear*n_block); KV-block = [nw:] (k,v,k_dim,v_dim rows).

Building blocks come from the 1st-stage per-axis Pareto fronts. CROSSOVER unit = whole
axis block (additive W⊥KV). MUTATION = importance-weighted by 1st-stage lever strength
(ε-floor → coverage; direction free → non-monotone stays reachable; NO monotone repair).
PREDICTOR input = ss.encode_predictor (frozen/pass_module cols dropped), like search.py.
"""
import os, json, glob
import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

from utils.func import get_net_info
from utils.select import maximin_extras


# ─────────────── encoding bounds / block split ───────────────
def encoding_xu(ss):
    """per-gene max option index for the (n_linear+4)*n_block encoding."""
    nb = ss.n_block
    xu = []
    for linear in ss.config['linear']:
        xu += [len(getattr(ss, f"{linear.split('.')[-1]}_option")) - 1] * nb
    xu += [len(ss.k_option) - 1] * nb
    xu += [len(ss.v_option) - 1] * nb
    xu += [len(ss.k_pruning_dim_option) - 1] * nb
    xu += [len(ss.v_pruning_dim_option) - 1] * nb
    return np.array(xu)


def nw_split(ss):
    return ss.n_linear * ss.n_block            # W-block size; KV-block = [nw:]


# ─────────────── 1st-stage archive → block pools + proxy curves ───────────────
def _last_stats(d):
    if d.endswith('.stats'):
        return d
    fs = glob.glob(f"{d}/iter_*.stats")
    return sorted(fs, key=lambda p: int(os.path.basename(p).split('iter_')[1].split('.')[0]))[-1]


def _front(jsd, comp):
    o = np.argsort(comp); m = np.inf; keep = []
    for i in o:
        if jsd[i] < m: keep.append(i); m = jsd[i]
    return np.array(keep)


def select_eps_band(jsd, comp, eps, eps_rel=0.0):
    """archs within the Pareto-front envelope at their comp, by an ABSOLUTE jsd margin
    `eps` and/or a RELATIVE one `eps_rel` (fraction of the front jsd):
        keep  jsd ≤ front(comp) · (1 + eps_rel) + eps
    eps=eps_rel=0 → the front. Relative is scale-free AND auto-adaptive: the band is
    WIDE where the front jsd is high (aggressive low-bit corner, where joint-optimality
    leaves the per-axis front) and NARROW in the saturated region — exactly where we want."""
    f = _front(jsd, comp)
    o = np.argsort(comp[f]); fc, fj = comp[f][o], jsd[f][o]
    best = np.interp(comp, fc, fj)                 # achievable front jsd at each arch's comp
    return np.where(jsd <= best * (1.0 + eps_rel) + eps)[0]


def per_comp_bin_topq(jsd, comp, idx, q, n_bins):
    """within `idx`, keep the q lowest-jsd archs per comp bin → budget-UNIFORM density
    (fixes the ε-band's skew toward the flat/saturated region; guarantees the steep
    aggressive corner is represented)."""
    if q <= 0 or len(idx) == 0:
        return idx
    edges = np.linspace(comp[idx].min(), comp[idx].max(), n_bins + 1)
    keep = []
    for b in range(n_bins):
        m = idx[(comp[idx] >= edges[b]) & (comp[idx] <= edges[b + 1])]
        if len(m): keep += list(m[np.argsort(jsd[m])[:q]])
    return np.array(sorted(set(keep)))


def per_comp_window_topq(jsd, comp, idx, q, h, n_centers):
    """CONTINUOUS/uniform version of per_comp_bin_topq: q-best within an OVERLAPPING
    window [c-h, c+h] at n_centers uniform comp positions → smooth budget-uniform density
    (no hard bin-edge artifacts; overlap h smooths). Use instead of bins for 'continuous'."""
    if q <= 0 or len(idx) == 0:
        return idx
    ci, ji = comp[idx], jsd[idx]
    keep = set()
    for ct in np.linspace(ci.min(), ci.max(), n_centers):
        m = np.where(np.abs(ci - ct) <= h)[0]
        if len(m): keep.update(idx[m[np.argsort(ji[m])[:q]]].tolist())
    return np.array(sorted(keep))


def diversity_prune(blocks, k, seed=0):
    """keep k blocks that are FARTHEST apart in encoding space (maximin) → drop near-
    duplicates, keep structurally-distinct allocation patterns (richer crossover)."""
    if k <= 0 or len(blocks) <= k:
        return np.arange(len(blocks))
    return np.asarray(maximin_extras(blocks.astype(float), anchor_idx=[], K=k, seed=seed))


def _select_blocks(j, c, encode_fn, eps, eps_rel, per_bin_q, n_bins, window_h, div_k, seed):
    """ε-band → per-comp-bin top-q (hard bins) OR sliding-window (window_h>0) → encode →
    structural-diversity prune. Returns blocks."""
    f = _front(j, c)
    idx = select_eps_band(j, c, eps, eps_rel)
    if window_h > 0:
        idx = per_comp_window_topq(j, c, idx, per_bin_q, window_h, n_bins)
    else:
        idx = per_comp_bin_topq(j, c, idx, per_bin_q, n_bins)
    blocks = np.stack([encode_fn(i) for i in idx])
    di = diversity_prune(blocks, div_k, seed)
    return f, idx, blocks[di], c[idx][di]      # also per-block comp (for budget-box filtering)


def _w_bits_of(b):
    """A W-arch entry is scalar bits (plain HQQ) OR [bits, n_outlier] (QEFT-on-HQQ)."""
    return int(b[0]) if isinstance(b, (list, tuple)) else int(b)


def _archs(*exprs):
    out = []
    for p in exprs:
        d = json.load(open(_last_stats(p))); out += [e[0] for e in d['archive'] + d.get('candidates', [])]
    return out


def derive_options(w_expr, kv_expr):
    """auto-derive the joint search-space options from the 1st-stage archives so ss.encode()
    covers EVERY arch in both (model/archive-general; no hardcoded bit/gs/prune grids).
    Handles QEFT W entries [bits, n_outlier] — the bit-widths are read off b[0]."""
    A = _archs(w_expr, kv_expr)
    wbits = sorted({_w_bits_of(b) for a in A for mod in a['q']['w'].values() for b in mod})
    kvbits = sorted({int(p[0]) for a in A for p in a['q']['k']} | {int(p[0]) for a in A for p in a['q']['v']})
    gsfor = {b: set() for b in kvbits}
    for a in A:
        for p in a['q']['k'] + a['q']['v']: gsfor[int(p[0])].add(int(p[1]))
    gs_lists = [sorted(gsfor[b]) for b in kvbits]
    kprune = sorted({int(d) for a in A for d in a['p']['k']})
    vprune = sorted({int(d) for a in A for d in a['p']['v']})
    return wbits, kvbits, gs_lists, kprune, vprune


def derive_qeft(w_expr, kv_expr):
    """QEFT outlier-column options from the 1st-stage archives. W entries are either scalar
    bits (plain HQQ) or [bits, n_outlier] (QEFT-on-HQQ). Returns (n_qeft_column, qeft_bits):
      n_qeft_column = sorted distinct n_outlier values seen (always incl 0), so ss rebuilds the
                      SAME (bits, n_outlier) W option ladder the archive used → ss.encode covers it.
      qeft_bits     = the bit-widths that actually carry outliers (the eligible/ladder bits).
    Defaults ([0], None) for a plain (non-QEFT) archive → unchanged scalar-bit behaviour."""
    A = _archs(w_expr, kv_expr)
    nqc, qbits = set(), set()
    for a in A:
        for mod in a['q']['w'].values():
            for b in mod:
                if isinstance(b, (list, tuple)):
                    nqc.add(int(b[1]))
                    if int(b[1]) > 0: qbits.add(int(b[0]))
    n_qeft_column = sorted(nqc | {0}) if nqc else [0]
    qeft_bits = sorted(qbits) if qbits else None
    return n_qeft_column, qeft_bits


def load_block_pools(w_expr, kv_expr, ss, w_eps=0.0, kv_eps=0.0, eps_rel=0.0,
                     per_bin_q=0, n_bins=10, window_h=0.0, div_k=0, seed=0):
    """W/KV building-block pools from the 1st-stage archives, via ε-band (abs `*_eps` and/or
    relative `eps_rel`) ∪ per-comp-bin top-q ∪ structural-diversity (data-driven, no random)."""
    nw = nw_split(ss)
    dw = json.load(open(_last_stats(w_expr))); Ew = dw['archive'] + dw.get('candidates', [])
    jw = np.array([e[1] for e in Ew]); cw = np.array([e[2] for e in Ew])
    fw, sw, Wg, w_comp = _select_blocks(jw, cw, lambda i: ss.encode(Ew[i][0])[:nw],
                                        w_eps, eps_rel, per_bin_q, n_bins, window_h, div_k, seed)
    dk = json.load(open(_last_stats(kv_expr))); Ek = dk['archive'] + dk.get('candidates', [])
    jk = np.array([e[1] for e in Ek]); ck = np.array([e[2] for e in Ek])
    fk, sk, KVg, kv_comp = _select_blocks(jk, ck, lambda i: ss.encode(Ek[i][0])[nw:],
                                          kv_eps, eps_rel, per_bin_q, n_bins, window_h, div_k, seed)
    ow = np.argsort(cw[fw]); wcurve = (cw[fw][ow], jw[fw][ow])     # proxy curve = front (not band)
    ok = np.argsort(ck[fk]); kcurve = (ck[fk][ok], jk[fk][ok])
    print(f"[pools] W: front {len(fw)} → ε{w_eps}-band {len(sw)} → blocks {len(Wg)} | "
          f"KV: front {len(fk)} → ε{kv_eps}-band {len(sk)} → blocks {len(KVg)}  (div_k={div_k})")
    return Wg, KVg, wcurve, kcurve, w_comp, kv_comp


class JointComp:
    """Vectorized comp_obj over a BATCH of encoded genomes — skips decode() + dict build +
    get_net_info() (the per-individual cost that dominates NSGA _next wall-clock). Faithful to
    utils.func.compute_bits / get_net_info for every key EXCEPT 'memory' (token-partition sweep,
    not a per-cell linear form) which falls back to per-row get_net_info.

    Genome layout (ss.encode): reshape (n_linear+4, n_block) = [w-modules…, k, v, k_dim, v_dim];
    each cell is an OPTION INDEX. wbits = numel-weighted mean of per-cell bits (+scale overhead);
    {kvbits,eff_kvbits,kvdim,…} = simple means over the k/v (+prune) cells via option LUTs."""
    SIMPLE = {'wbits', 'kvbits', 'kbits', 'vbits', 'eff_kvbits', 'eff_kbits', 'eff_vbits',
              'kvdim', 'kdim', 'vdim'}

    def __init__(self, ss):
        self.ss = ss
        cfg, nb = ss.config, ss.n_block
        self.nl, self.nb, self.nw = ss.n_linear, nb, ss.n_linear * nb
        wgs = ss.group_size['w']; hd = int(cfg['head_dim']); self.hd = hd
        # W: per-module option→total-weight-bits LUT (matches compute_bits('w'): out·in·bits
        # + scale/zp (bits<16) + out·n_outlier·16 for QEFT FP16 columns). Options are scalar
        # bits (plain HQQ) OR (bits, n_outlier) tuples (QEFT-on-HQQ) — handled uniformly.
        self.w_numel, self.w_mem_lut = [], []
        for linear in cfg['linear']:
            out_dim, in_dim = map(int, cfg['linear_shape'][linear])
            lgs = in_dim if wgs == -1 else wgs
            self.w_numel.append(out_dim * in_dim)
            mem = []
            for o in getattr(ss, f"{linear.split('.')[-1]}_option"):
                b, no = (int(o[0]), int(o[1])) if isinstance(o, (list, tuple)) else (int(o), 0)
                m = out_dim * in_dim * b
                if b < 16:
                    m += (in_dim // lgs) * out_dim * 32        # scale + zero point
                m += out_dim * no * 16                          # FP16 outlier columns (QEFT)
                mem.append(m)
            self.w_mem_lut.append(np.asarray(mem, float))
        self.w_numel = np.asarray(self.w_numel, float)
        self.w_nparam_total = float((self.w_numel * nb).sum())
        # KV: option→(bits + 32/gs) effective-bits LUT, and option→retain-factor / retain-dim LUT
        eff = lambda opt: np.asarray([b + (32.0 / g if g else 0.0) for b, g in opt], float)
        self.k_eff, self.v_eff = eff(ss.k_option), eff(ss.v_option)
        self.k_ret = np.asarray([1.0 - d / hd for d in ss.k_pruning_dim_option], float)
        self.v_ret = np.asarray([1.0 - d / hd for d in ss.v_pruning_dim_option], float)
        self.k_dim = np.asarray([hd - d for d in ss.k_pruning_dim_option], float)
        self.v_dim = np.asarray([hd - d for d in ss.v_pruning_dim_option], float)

    def _wbits(self, Xr):
        mem = np.zeros(len(Xr))
        for m in range(self.nl):
            mem += self.w_mem_lut[m][Xr[:, m, :]].sum(1)          # (N,) total weight bits-memory
        return mem / self.w_nparam_total

    def batch(self, X, keys):
        """X: (N, n_var) int → (N, len(keys)) float comp matrix."""
        X = np.asarray(X, int)
        Xr = X.reshape(len(X), self.nl + 4, self.nb)
        ki, vi = Xr[:, self.nl, :], Xr[:, self.nl + 1, :]
        kpi, vpi = Xr[:, self.nl + 2, :], Xr[:, self.nl + 3, :]
        ke, ve = self.k_eff[ki], self.v_eff[vi]                   # (N, nb) effective bits
        kr, vr = self.k_ret[kpi], self.v_ret[vpi]                 # (N, nb) retain factors
        denom = 2 * self.nb
        out = np.empty((len(X), len(keys)))
        fallback = [k for k in keys if k not in self.SIMPLE]
        fb_vals = None
        if fallback:                                             # 'memory' etc. — per-row get_net_info
            fb_vals = {k: np.empty(len(X)) for k in fallback}
            for i, g in enumerate(X):
                ni = get_net_info(self.ss.decode(g), self.ss.config, self.ss.group_size)
                for k in fallback: fb_vals[k][i] = ni[k]
        for j, key in enumerate(keys):
            if key == 'wbits':           out[:, j] = self._wbits(Xr)
            elif key == 'kvbits':        out[:, j] = (ke.sum(1) + ve.sum(1)) / denom
            elif key == 'kbits':         out[:, j] = ke.mean(1)
            elif key == 'vbits':         out[:, j] = ve.mean(1)
            elif key == 'eff_kvbits':    out[:, j] = ((ke * kr).sum(1) + (ve * vr).sum(1)) / denom
            elif key == 'eff_kbits':     out[:, j] = (ke * kr).mean(1)
            elif key == 'eff_vbits':     out[:, j] = (ve * vr).mean(1)
            elif key == 'kvdim':         out[:, j] = (self.k_dim[kpi].sum(1) + self.v_dim[vpi].sum(1)) / denom
            elif key == 'kdim':          out[:, j] = self.k_dim[kpi].mean(1)
            elif key == 'vdim':          out[:, j] = self.v_dim[vpi].mean(1)
            else:                        out[:, j] = fb_vals[key]
        return out


def gene_weights(Wg, KVg, eps=0.05):
    """per-gene lever strength = std over the 1st-stage front blocks, ε-floored (never 0)."""
    w = np.concatenate([Wg.std(0), KVg.std(0)])
    return np.clip(w / (w.max() + 1e-9), eps, 1.0)


def modal_agreement(Wg, KVg):
    """per-gene modal-value fraction across the 1st-stage blocks (1.0 = ALL blocks put the
    same option in that cell). High agreement ⇒ the pool has effectively DECIDED that cell."""
    def mf(G):
        return np.array([np.unique(G[:, c], return_counts=True)[1].max() / len(G)
                         for c in range(G.shape[1])])
    return np.concatenate([mf(Wg), mf(KVg)])


def freeze_mask(Wg, KVg, agree_frac):
    """L2 direct space reduction: cells where ≥ agree_frac of the 1st-stage blocks AGREE are
    FROZEN at the pool consensus (excluded from mutation → they stay at the sampled-block
    value; sampling/crossover already respect them). Returns (contested_bool, modal_frac):
    contested=True ⇒ a free/search dim. agree_frac≥1.0+ freezes only unanimous cells."""
    mf = modal_agreement(Wg, KVg)
    return mf < agree_frac, mf


def proxy_loss(ni, wcurve, kcurve, g=None, w=None, xu=None):
    """CPU smoke loss: 1st-stage front additive interp (+ allocation term so the loss
    depends on the full genome, not just the 2 comp axes — keeps the surrogate well-posed)."""
    base = float(np.interp(ni['wbits'], *wcurve) + np.interp(ni['eff_kvbits'], *kcurve))
    if g is not None:
        base += 0.30 * float(np.mean(w * (1.0 - g / np.maximum(xu, 1))))
    return base


# ─────────────── operators (axis-block crossover, importance-weighted mutation) ───────────────
class FrontierProductSampling(Sampling):
    """assemble a genome = (random W-front block, random KV-front block)."""
    def __init__(self, Wg, KVg): super().__init__(); self.Wg, self.KVg = Wg, KVg

    def _do(self, problem, n, **kw):
        wi = np.random.randint(0, len(self.Wg), n); ki = np.random.randint(0, len(self.KVg), n)
        return np.array([np.concatenate([self.Wg[wi[i]], self.KVg[ki[i]]]) for i in range(n)])


class AxisBlockCrossover(Crossover):
    """swap a whole axis block (W rows or KV rows) between two parents — additive W⊥KV."""
    def __init__(self, nw, n_var, prob=0.9):
        super().__init__(2, 2); self.nw, self.n_var, self.prob = nw, n_var, prob

    def _do(self, problem, X, **kw):
        _, n, _ = X.shape; Y = np.copy(X)
        for k in range(n):
            if np.random.random() > self.prob: continue
            seg = slice(0, self.nw) if np.random.random() < 0.5 else slice(self.nw, self.n_var)
            Y[0, k, seg], Y[1, k, seg] = X[1, k, seg].copy(), X[0, k, seg].copy()
        return Y


class WeightedIntMutation(Mutation):
    """per-gene ±1 step, prob ∝ 1st-stage lever weight (ε-floor → coverage; direction free)."""
    def __init__(self, w, xu, base=0.06): super().__init__(); self.w, self.xu, self.base = w, xu, base

    def _do(self, problem, X, **kw):
        X = X.copy()
        for i in range(len(X)):
            hit = np.random.random(len(self.xu)) < (self.base * self.w)
            X[i] = np.where(hit, np.clip(X[i] + np.random.choice([-1, 1], len(self.xu)), 0, self.xu), X[i])
        return X


def block_segments(ss):
    """coherent gene segments = the 7 W modules + (k, v, k_dim, v_dim) KV rows, each ×n_block.
    Used by knowledge-mutation to transplant a whole 1st-stage sub-pattern."""
    nb = ss.n_block; segs = [(m * nb, (m + 1) * nb, 'w') for m in range(ss.n_linear)]
    nw = ss.n_linear * nb
    segs += [(nw + j * nb, nw + (j + 1) * nb, 'kv') for j in range(4)]
    return segs


class KnowledgeMutation(Mutation):
    """1st-stage-KNOWLEDGE-guided mutation (vs WeightedIntMutation's lever-weighted ±1):
      WHERE: cells chosen ∝ 1st-stage lever weight (ε-floor → all reachable).
      HOW per cell: with prob p_val → set to a value DRAWN FROM that cell's 1st-stage
        value distribution (the option indices it took across the near-front band) — a
        meaningful, 1st-stage-plausible value; else ±1 local step (direction-free).
      + module-transplant (prob p_mod/indiv): copy a whole module/segment from a random
        1st-stage block → a coherent 1st-stage sub-pattern (not an incoherent cell jitter)."""
    def __init__(self, w, xu, Wg, KVg, nw, segments, base=0.06, p_val=0.5, p_mod=0.15):
        super().__init__()
        self.w, self.xu, self.nw, self.base, self.p_val, self.p_mod = w, xu, nw, base, p_val, p_mod
        self.Wg, self.KVg, self.segments = Wg, KVg, segments

    def _do(self, problem, X, **kw):
        X = X.copy(); nv = len(self.xu)
        for i in range(len(X)):
            if np.random.random() < self.p_mod:                      # coherent 1st-stage sub-pattern
                s, e, ax = self.segments[np.random.randint(len(self.segments))]
                src = self.Wg if ax == 'w' else self.KVg
                off = 0 if ax == 'w' else self.nw
                X[i, s:e] = src[np.random.randint(len(src)), s - off:e - off]
            for g in np.where(np.random.random(nv) < self.base * self.w)[0]:
                if np.random.random() < self.p_val:                  # 1st-stage-plausible value
                    col = self.Wg[:, g] if g < self.nw else self.KVg[:, g - self.nw]
                    X[i, g] = col[np.random.randint(len(col))]
                else:                                                # local ±1 (direction-free)
                    X[i, g] = min(max(X[i, g] + (1 if np.random.random() < 0.5 else -1), 0), self.xu[g])
        return X


class JointAuxProblem(Problem):
    """surrogate problem for _next (mirrors AuxiliarySingleLevelProblemThink): fitness =
    predictor(decode_encode_predictor(x)) for loss [arch-input]; objs = (loss, wbits,
    eff_kvbits) with comp_obj box constraints. predictor=None → loss col 0."""
    def __init__(self, ss, predictor, active, xu, comp_obj, comp_obj_min, comp_obj_max,
                 n_token=0, attn_sink=0, comp=None):
        super().__init__(n_var=len(xu), n_obj=len(comp_obj) + 1, n_constr=2 * len(comp_obj),
                         xl=0, xu=xu.copy(), vtype=int)
        self.ss, self.pred, self.active, self.comp_obj = ss, predictor, active, comp_obj
        self.cmin = np.asarray(comp_obj_min, float); self.cmax = np.asarray(comp_obj_max, float)
        self.n_token, self.attn_sink = n_token, attn_sink
        self.comp = comp if comp is not None else JointComp(ss)   # vectorized comp (no decode loop)

    def _evaluate(self, X, out, *a, **kw):
        Xc = np.clip(np.round(X), 0, self.xu).astype(int)
        loss = (np.asarray(self.pred.predict(Xc[:, self.active].astype(float))).ravel()
                if self.pred is not None else np.zeros(len(Xc)))
        comp = self.comp.batch(Xc, self.comp_obj)                # (N, n_obj-1) vectorized
        out['F'] = np.column_stack([loss, comp])
        cmin, cmax = self.cmin, self.cmax                        # constraints g<=0 (box), vectorized
        glo = np.where(cmin != 0, (cmin - comp) / np.where(cmin != 0, cmin, 1.0), 0.0)
        ghi = np.where(cmax != 0, comp / np.where(cmax != 0, cmax, 1.0) - 1.0, 0.0)
        G = np.empty((len(Xc), 2 * len(self.comp_obj)))
        G[:, 0::2] = glo; G[:, 1::2] = ghi
        out['G'] = G
