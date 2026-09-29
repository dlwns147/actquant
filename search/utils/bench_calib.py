"""Benchmark-calibrated tie-breaking for post_search's joint (second_expr) path.

Trained on a correlation.py campaign directory (correlation.csv + archs.csv,
e.g. the 200-arch Llama-3.1-8B run) and used ONLY to re-rank measured-loss
NEAR-TIES inside the budget box — never to override a clear loss verdict.

Design decisions (each backed by a measured comparison; see the 2608 audit):
  * input  = raw ONE-HOT genome (no hand-crafted features; option vocabulary
    is taken from the calibration archs, so nothing here is tuned by hand).
    Unseen option values in a scored arch encode as all-zero for that cell
    (prediction-neutral) and are counted + warned about.
  * front  = PLS-8 supervised on the BENCHMARK target ('ruler' or
    'longbench', per target). Supervising on the search objective itself
    (sqrt-JSD plstyp) was measurably worse: those latents inherit exactly the
    proxy blindness this module exists to break. Loss enters (optionally) as a
    model INPUT via loss_col, never as a target.
  * head   = predictor.factory 'rbf' (tps) or 'ard_gp' on the 8 latents.
    Raw targets, no sqrty/logy/logity transform. rbf is an interpolant with
    no noise term: fine on the spread-out calibration design, but prefer
    'ard_gp' if labels are ever added adaptively/clustered.
  * output = RANK-ONLY scores. Absolute values are NOT calibrated (bias grows
    with context length); never threshold or report them as scores.
"""
import os
import json

import numpy as np


_W_LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
              "mlp.down_proj")

def _target_vector(rows, tgt):
    """y (HIGHER-IS-BETTER) for a benchmark target — 'ruler' or 'longbench'.
    Loss/proxy columns are deliberately NOT accepted as targets; measured loss
    may enter as a model input (loss_col) instead."""
    if tgt == 'ruler':
        cols = [c for c in rows[0] if c.startswith('ruler__')
                and c != 'ruler__avg']
        return np.clip(np.array([[float(r[c]) for c in cols] for r in rows]),
                       0, 1).mean(1)
    if tgt == 'longbench':
        return np.array([float(r['longbench_e__avg']) for r in rows])
    raise SystemExit(f"[bench-calib] unknown target '{tgt}' "
                     f"(valid: ruler | longbench)")


def check_loss_protocol(loss_col, protocol):
    """The measured loss fed at scoring time must mean the same thing as the
    calibration column: compare the archive's stored protocol against the
    metric registry spec of `loss_col`. Mismatch is a hard error; a loss_col
    unknown to the registry only warns (nothing to compare against)."""
    if not protocol:
        print(f"[bench-calib] archive stats carry no protocol — cannot verify "
              f"loss_col '{loss_col}' consistency")
        return
    try:
        from utils.metric_specs import resolve_tasks, groups_for
        tasks = resolve_tasks([loss_col])
    except Exception:
        print(f"[bench-calib] loss_col '{loss_col}' not in the metric registry "
              f"— protocol check skipped")
        return
    key, grp, ds, kw = tasks[0]
    spec = dict(groups_for(tasks))[grp]
    expect = dict(dataset=ds, n_sample=spec['n_sample'], seqlen=spec['seqlen'],
                  loss_func=kw['loss_func'], stride=kw['stride'],
                  prefill_prompt=kw['prefill_prompt'],
                  last_tokens=kw['last_tokens'])
    mismatch = {k: (protocol.get(k), v) for k, v in expect.items()
                if k in protocol and protocol.get(k) != v}
    if mismatch:
        raise SystemExit(
            f"[bench-calib] --bench_calib_loss_col '{loss_col}' does not match "
            f"the archive's loss protocol (archive vs {loss_col}): {mismatch}")


def _collect_vocab(archs):
    """Option sets actually present in the calibration archs."""
    w, kv, pr = set(), set(), set()
    for a in archs:
        for lin in _W_LINEARS:
            w.update(int(b) for b in a['q']['w'][lin])
        for key in ('k', 'v'):
            kv.update((int(b), int(g)) for b, g in a['q'][key])
            pr.update(int(p) for p in a['p'][key])
    return dict(w=sorted(w), kv=sorted(kv), pr=sorted(pr))


def _onehot(arch, vocab, miss=None):
    """Flat one-hot genome under `vocab`; unseen values -> all-zero cell."""
    v = []
    for lin in _W_LINEARS:
        for b in arch['q']['w'][lin]:
            v += [1.0 if int(b) == o else 0.0 for o in vocab['w']]
            if miss is not None and int(b) not in vocab['w']:
                miss.append(('w', int(b)))
    for key in ('k', 'v'):
        for b, g in arch['q'][key]:
            v += [1.0 if (int(b), int(g)) == o else 0.0 for o in vocab['kv']]
            if miss is not None and (int(b), int(g)) not in vocab['kv']:
                miss.append(('kv', (int(b), int(g))))
    for key in ('k', 'v'):
        for p in arch['p'][key]:
            v += [1.0 if int(p) == o else 0.0 for o in vocab['pr']]
            if miss is not None and int(p) not in vocab['pr']:
                miss.append(('prune', int(p)))
    return np.asarray(v, float)


class BenchCalib:
    """Per-target (PLS-8 -> factory predictor) rankers over one-hot genomes."""

    def __init__(self, calib_dir, targets=('ruler', 'longbench'),
                 predictor='rbf', model_name='', loss_col=''):
        import csv as _csv
        if model_name and model_name not in os.path.abspath(calib_dir):
            raise SystemExit(
                f"[bench-calib] calibration dir does not mention model "
                f"'{model_name}': {calib_dir}\nCross-model calibration is "
                f"unvalidated — point --bench_calib_dir at a campaign for "
                f"this model or drop the flag.")
        with open(os.path.join(calib_dir, 'correlation.csv'), newline='') as f:
            rows = list(_csv.DictReader(f))
        with open(os.path.join(calib_dir, 'archs.csv'), newline='') as f:
            arows = list(_csv.DictReader(f))
        if len(rows) != len(arows):
            raise SystemExit(f"[bench-calib] correlation.csv ({len(rows)}) and "
                             f"archs.csv ({len(arows)}) row counts differ")
        archs = [json.loads(r['arch_json']) for r in arows]
        self.vocab = _collect_vocab(archs)
        X = np.stack([_onehot(a, self.vocab) for a in archs])
        ys = {tgt: _target_vector(rows, tgt) for tgt in targets}
        # optional measured-loss covariate: benchmark ≈ g(loss) + arch effects.
        # The caller must then pass the archive's measured loss to scores();
        # it MUST be the same protocol as this column (check_loss_protocol).
        self.loss_col = loss_col
        loss_vec = None
        if loss_col:
            if loss_col not in rows[0]:
                raise SystemExit(f"[bench-calib] loss_col '{loss_col}' is not "
                                 f"a correlation.csv column")
            loss_vec = np.array([float(r[loss_col]) for r in rows])
        cols = list(ys.values())
        if loss_vec is not None:
            cols.append(loss_vec)
        ok = np.all(np.isfinite(np.column_stack(cols)), axis=1)
        if ok.sum() < 100:
            raise SystemExit(f"[bench-calib] only {int(ok.sum())} complete "
                             f"target rows in {calib_dir}; need >= 100")
        self.models = {}
        for tgt, y in ys.items():
            self.models[tgt] = self._fit(
                X[ok], y[ok], predictor,
                loss=loss_vec[ok] if loss_vec is not None else None)
        self.n_labels = int(ok.sum())
        self.predictor = predictor

    @staticmethod
    def _fit(X, y, predictor, loss=None):
        from sklearn.cross_decomposition import PLSRegression
        from predictor.factory import get_predictor
        pls = PLSRegression(n_components=8).fit(X, y)
        L = pls.transform(X)
        if loss is not None:
            # loss joins AFTER the PLS front so it is not diluted among the
            # 1568 one-hot columns; it enters the predictor head directly.
            L = np.hstack([L, np.asarray(loss, float)[:, None]])
        mu, sd = L.mean(0), L.std(0) + 1e-12
        Z = (L - mu) / sd
        kw = (dict(kernel='tps', lb=Z.min(0), ub=Z.max(0) + 1e-9)
              if predictor == 'rbf' else
              dict(ard_kernel='matern32', gp_n_restarts=3))
        head = get_predictor(predictor, Z, y, device='cpu', **kw)
        return dict(pls=pls, mu=mu, sd=sd, head=head, use_loss=loss is not None)

    def scores(self, archs, target, loss=None):
        """RANK-ONLY predicted benchmark scores (higher = better). When the
        model was fit with loss_col, `loss` = measured loss of `archs` (same
        protocol as loss_col) is REQUIRED."""
        miss = []
        X = np.stack([_onehot(a, self.vocab, miss) for a in archs])
        if miss:
            uniq = sorted(set(miss))
            print(f"[bench-calib] WARNING: {len(miss)} option values unseen in "
                  f"calibration (prediction-neutral): {uniq[:6]}"
                  + (' …' if len(uniq) > 6 else ''))
        m = self.models[target]
        L = m['pls'].transform(X)
        if m['use_loss']:
            if loss is None:
                raise SystemExit("[bench-calib] model was fit with loss_col "
                                 "but scores() got no measured loss")
            L = np.hstack([L, np.asarray(loss, float)[:, None]])
        Z = (L - m['mu']) / m['sd']
        return np.asarray(m['head'].predict(Z)).reshape(-1)


# ═════════════════════════════════════════════════════════════════════════════
# PRE-SEARCH METRIC RECOMMENDER
# ═════════════════════════════════════════════════════════════════════════════
# Picks the search objective (a utils/metric_specs name) for ONE model from that
# model's labelled correlation campaign, so a per-model metric choice is a pipeline
# step and not a hand analysis. Run it after correlation_eval, before search:
#
#   python -m utils.bench_calib --recommend save/correlation/<pool>  [--chat_only]
#       -> prints the name, writes save/metric_recommend/<pool>.json
#   METRIC_TASK=auto:save/correlation/<pool>      (search.sh / second_search*.sh)
#       -> scripts/metric_task.sh resolves it through --resolve below
#
# Why it is built this way (tests/bench_selection_regret.py findings):
#   * objective = FRONT + BAND regret, both in RULER points. The search keeps a
#     Pareto front over the WHOLE W/KV space (front regret needs global AND local
#     rank order), and select_joint then picks inside a memory band (band regret).
#     Band regret alone picked gov_jsd for Qwen: best in-band, worst front (28).
#   * candidates = only names the search scripts can actually measure
#     (metric_specs.task_knobs with require=loss/no-key-token — the same check
#     metric_task.sh --loss_only enforces).
#   * no theory, only measured regret: every loading-based point prediction
#     failed (24). Selection over many candidates on few boxes overfits by
#     ~0.4 pts (26), so the pick is BAGGED over stratified split-halves and the
#     reported regret is the HELD-OUT half, never the in-sample best.
RULER_TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3',
               'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
               'niah_multivalue', 'niah_multiquery', 'ruler_vt', 'ruler_cwe',
               'ruler_fwe', 'ruler_qa_squad', 'ruler_qa_hotpot')
_NON_METRIC_KEYS = {'idx', 'arch', 'complexity', 'ruler', 'longbench',
                    'longbench_e'}


def load_labelled_pool(pool_dir, measure_dir=None):
    """(names, M [n_arch x n_metric], ruler [n_arch], mem [n_arch], idx) from a
    correlation campaign. Only archs with all 13 RULER tasks are kept. Metric
    columns may be partially measured (NaN) — the caller filters coverage."""
    import csv
    import glob
    md = measure_dir or pool_dir
    if not glob.glob(os.path.join(md, 'result_*.json')):
        subs = sorted(glob.glob(os.path.join(pool_dir, 'm_*')))
        if subs:
            md = subs[0]
    with open(os.path.join(pool_dir, 'archs.csv')) as f:
        arch_rows = {int(r['idx']): r for r in csv.DictReader(f)}
    recs = []
    for i, ar in sorted(arch_rows.items()):
        p = os.path.join(md, f'result_{i}.json')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            res = json.load(f)
        rl = res.get('ruler')
        if not isinstance(rl, dict) or not all(t in rl for t in RULER_TASKS):
            continue
        y = float(np.mean([min(max(float(rl[t]), 0.0), 1.0) for t in RULER_TASKS]))
        vals = {k: float(v) for k, v in res.items()
                if k not in _NON_METRIC_KEYS and not k.startswith('_')
                and isinstance(v, (int, float))}
        recs.append((i, float(ar['memory']), y, vals))
    names = sorted({k for _, _, _, v in recs for k in v})
    M = np.array([[v.get(k, np.nan) for k in names] for _, _, _, v in recs], float)
    return (names, M, np.array([r[2] for r in recs]),
            np.array([r[1] for r in recs]), np.array([r[0] for r in recs]))


def _search_legal(name):
    from utils.metric_specs import task_knobs
    try:
        task_knobs(name, require={'metric': 'loss', 'use_key_token': False},
                   context='the pre-search recommender')
        return True
    except (SystemExit, KeyError, ValueError):
        return False


def _front_regret(m, y, mem, n_budget=17):
    """Mean over budgets b of  max RULER{mem<=b} - RULER[argmin metric{mem<=b}].
    This is what a Pareto search + 'lowest loss that fits the budget' delivers."""
    out = []
    for b in np.quantile(mem, np.linspace(0.15, 0.95, n_budget)):
        ok = np.where(mem <= b)[0]
        if len(ok) >= 2:
            out.append(y[ok].max() - y[ok][int(np.argmin(m[ok]))])
    return float(np.mean(out)) * 100.0


def _band_regret(m, y, mem, box=10):
    """Mean over sliding memory boxes of in-box regret (select_joint's decision)."""
    o = np.argsort(mem); st = max(1, box // 2); out = []
    for s in range(0, len(o) - box + 1, st):
        w = o[s:s + box]
        out.append(y[w].max() - y[w][int(np.argmin(m[w]))])
    return float(np.mean(out)) * 100.0 if out else float('nan')


def _objective(m, y, mem, objective):
    f = _front_regret(m, y, mem); b = _band_regret(m, y, mem)
    return {'front': f, 'band': b, 'both': 0.5 * (f + b)}[objective], f, b


def _metric_cost(name):
    """Per-arch metric wall-clock in seconds, and whether it was MEASURED.
    MEASURED medians come from save/metric_recommend/metric_costs.json, built by
    `python -m utils.bench_calib --build_costs <correlation_eval log dir>` from the
    '[correlation/eval]   <name> = <v>  (<s>s)' lines (same hardware, one metric
    per line, excludes AWQ and the one-off FP-teacher pass).
    Do NOT fall back to n_sample x seqlen across corpora: that model is wrong by
    ~20x for wikitext2@2048 (measured ~190k tok/s vs ~9-11k tok/s at 8192/16384),
    which inverted every wt2-vs-gov cost comparison until finding 32. The fallback
    therefore scales a MEASURED metric of the same corpus and seqlen by n_sample and
    answer chunks, and is flagged as an estimate."""
    import re
    from utils.metric_specs import METRIC_TASKS, GROUPS
    reg = {t[0]: t for t in METRIC_TASKS}
    table = _load_costs()
    if name in table:
        return table[name], True
    _, g, ds, sp = reg[name]; G = GROUPS[g]
    def key(n):
        # corpus, seqlen AND code path: PPL (eval_ppl) runs ~16x slower than the
        # loss path JSD/CE share (measured gov_ppl ~110 s vs gov_jsd ~7 s, same docs)
        _, gg, dd, ssp = reg[n]
        return (str(dd).replace('chat:', ''), GROUPS[gg]['seqlen'], ssp.get('metric'))
    def chunks(spec):
        lt = spec.get('last_tokens') or 0; st = spec.get('stride') or 0
        return max(1, -(-lt // st)) if (lt and st) else 1
    same = [n for n in table if n in reg and key(n) == key(name)]
    if same:
        ref = min(same, key=lambda n: abs(GROUPS[reg[n][1]]['n_sample'] - G['n_sample']))
        _, rg, _, rsp = reg[ref]
        t = table[ref] * G['n_sample'] / GROUPS[rg]['n_sample'] \
            * (0.8 + 0.2 * chunks(sp) / chunks(rsp))
        return t, False
    return G['n_sample'] * G['seqlen'] / 9550.0, False   # last resort, long-context rate


_COSTS = None


def _load_costs():
    global _COSTS
    if _COSTS is None:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'save', 'metric_recommend', 'metric_costs.json')
        _COSTS = json.load(open(p))['median_s'] if os.path.exists(p) else {}
    return _COSTS


def build_costs(log_dir, min_runs=3):
    import glob, re, statistics
    pat = re.compile(r"\[correlation/eval\]\s+([a-z0-9_]+) = [-0-9.eE+]+\s+\(([0-9.]+)s\)")
    tim = {}
    for f in glob.glob(os.path.join(log_dir, '*.out')):
        try:
            txt = open(f, errors='ignore').read()
        except OSError:
            continue
        for n, sec in pat.findall(txt):
            tim.setdefault(n, []).append(float(sec))
    med = {n: round(statistics.median(v), 2) for n, v in tim.items() if len(v) >= min_runs}
    out = {'created': __import__('time').strftime('%Y-%m-%d %H:%M:%S'), 'log_dir': log_dir,
           'min_runs': min_runs, 'runs': {n: len(tim[n]) for n in med}, 'median_s': med}
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'save', 'metric_recommend', 'metric_costs.json')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w'), indent=1)
    return p, len(med)


def _band_units(m, y, mem, box=10):
    o = np.argsort(mem); st = max(1, box // 2)
    return np.array([y[w].max() - y[w][int(np.argmin(m[w]))]
                     for w in (o[k:k + box] for k in range(0, len(o) - box + 1, st))]) * 100.0


def _front_units(m, y, mem, n_budget=17):
    out = []
    for b in np.quantile(mem, np.linspace(0.15, 0.95, n_budget)):
        ok = np.where(mem <= b)[0]
        if len(ok) >= 2:
            out.append(y[ok].max() - y[ok][int(np.argmin(m[ok]))])
    return np.array(out) * 100.0


def ruler_label_se(pool_dir, idx):
    """Per-arch standard error of the 13-task RULER mean, in points.
    EXACT from <pool>/ruler_<idx>_len<L>_s<n>/per_example_s0.jsonl when present;
    otherwise a guaranteed UPPER BOUND sqrt(sum_t p_t(1-p_t)/n)/13 from the task
    means (for any score in [0,1] with mean p, variance <= p(1-p)). Measured on
    Qwen2.5-7B, where both exist, the bound is 20% conservative (median 1.43 vs
    1.20 pts). Returns (se_points array, n_exact)."""
    import glob, re
    base = os.path.basename(os.path.normpath(pool_dir))
    mL = re.search(r'_t(\d+)_', base); L = mL.group(1) if mL else '16384'
    md = pool_dir if glob.glob(os.path.join(pool_dir, 'result_*.json')) else \
        (sorted(glob.glob(os.path.join(pool_dir, 'm_*'))) or [pool_dir])[0]
    se, n_exact = [], 0
    for i in idx:
        dumps = glob.glob(os.path.join(md, f'ruler_{int(i)}_len{L}_s*',
                                       'per_example_s0.jsonl'))
        if dumps:
            by = {t: [] for t in RULER_TASKS}
            with open(dumps[0]) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get('task') in by:
                        by[r['task']].append(min(max(float(r['score']), 0.0), 1.0))
            if all(len(v) > 1 for v in by.values()):
                se.append(np.sqrt(sum(np.var(v, ddof=1) / len(v)
                                      for v in by.values())) / 13 * 100.0)
                n_exact += 1
                continue
        with open(os.path.join(md, f'result_{int(i)}.json')) as f:
            rl = json.load(f)['ruler']
        ns = 50
        se.append(np.sqrt(sum(min(max(rl[t], 0), 1) * (1 - min(max(rl[t], 0), 1)) / ns
                              for t in RULER_TASKS)) / 13 * 100.0)
    return np.array(se), n_exact


def label_noise_floor(y, mem, se_pts, draws=300, seed=0):
    """Regret a PERFECT metric (one that knows true RULER) still shows when scored
    against labels this noisy, using the recommender's own band/front definitions.
    Truth := observed y; labels := truth + N(0, se). Below this floor no metric can
    be distinguished from perfect with these labels (finding 30)."""
    rng = np.random.default_rng(seed)
    t = y * 100.0
    o = np.argsort(mem)
    B = [o[k:k + 10] for k in range(0, len(o) - 10 + 1, 5)]
    F = [np.where(mem <= b)[0] for b in np.quantile(mem, np.linspace(0.15, 0.95, 17))]
    fb, ff = [], []
    for _ in range(draws):
        lab = t + rng.normal(0, 1, len(t)) * se_pts
        fb.append(np.mean([lab[w].max() - lab[w][int(np.argmax(t[w]))] for w in B]))
        ff.append(np.mean([lab[w].max() - lab[w][int(np.argmax(t[w]))] for w in F]))
    return float(np.mean(fb)), float(np.mean(ff))


def recommend_search_metric(pool_dir, candidates=None, chat_only=False,
                            objective='both', n_splits=40, seed=0,
                            min_coverage=1.0, tie_tol=0.3, n_boot=4000,
                            verbose=True):
    names, M, y, mem, _ = load_labelled_pool(pool_dir)
    n = len(y)
    if n < 20:
        raise SystemExit(f"[recommend] only {n} labelled archs in {pool_dir}; "
                         f"need >= 20 (and ~65+ for a stable pick).")
    cand = list(candidates) if candidates else names
    cand = [c for c in cand if c in names]
    if chat_only:
        cand = [c for c in cand if '_chat' in c]
    cand = [c for c in cand
            if np.isfinite(M[:, names.index(c)]).mean() >= min_coverage]
    cand = [c for c in cand if _search_legal(c)]
    if not cand:
        raise SystemExit("[recommend] no search-legal candidate is fully measured "
                         "on this pool (check --chat_only / --candidates).")
    col = {c: M[:, names.index(c)] for c in cand}
    rng = np.random.default_rng(seed)
    order = np.argsort(mem)
    train_obj = {c: [] for c in cand}; held, picks = [], []
    for _ in range(n_splits):
        # stratified split-half: pair neighbours in memory, send one of each pair
        # to each half, so both halves span the whole budget range
        a = []
        for k in range(0, n - 1, 2):
            a.append(order[k] if rng.random() < 0.5 else order[k + 1])
        A = np.array(sorted(a)); Bh = np.setdiff1d(np.arange(n), A)
        sc = {c: _objective(col[c][A], y[A], mem[A], objective)[0] for c in cand}
        for c in cand:
            train_obj[c].append(sc[c])
        best = min(cand, key=sc.get); picks.append(best)
        held.append(_objective(col[best][Bh], y[Bh], mem[Bh], objective)[0])
    bag = {c: float(np.mean(v)) for c, v in train_obj.items()}
    best = min(cand, key=bag.get)
    held = np.array(held)
    from collections import Counter
    freq = Counter(picks)

    # ── cost-aware tie-break (finding 30) ──────────────────────────────────────
    # The top candidates routinely differ by 0.1-0.3 pts, below the label-noise
    # floor (~0.35-0.5 pts), and the bagged winner flips between runs. Ranking
    # inside that band is noise, so: a candidate is TIED with the bagged best when
    # its paired-bootstrap 95% CI vs the best contains 0 on BOTH band (per memory
    # box) AND front (per budget), and its full-data objective is within tie_tol.
    # Among the tied, take the CHEAPEST (n_sample x seqlen); cost ties -> bagged.
    brng = np.random.default_rng(seed + 1)
    ub = {c: _band_units(col[c], y, mem) for c in cand}
    uf = {c: _front_units(col[c], y, mem) for c in cand}
    Ib = brng.integers(0, len(ub[best]), (n_boot, len(ub[best])))
    If = brng.integers(0, len(uf[best]), (n_boot, len(uf[best])))
    fullobj = {c: {'front': uf[c].mean(), 'band': ub[c].mean(),
                   'both': 0.5 * (uf[c].mean() + ub[c].mean())}[objective] for c in cand}
    tied, why = [best], {best: 'bagged best'}
    for c in cand:
        if c == best:
            continue
        gap = fullobj[c] - fullobj[best]
        db = (ub[c] - ub[best])[Ib].mean(1); df = (uf[c] - uf[best])[If].mean(1)
        cb = np.percentile(db, [2.5, 97.5]); cf = np.percentile(df, [2.5, 97.5])
        if gap <= tie_tol and cb[0] <= 0 <= cb[1] and cf[0] <= 0 <= cf[1]:
            tied.append(c); why[c] = f'gap {gap:+.2f}, band CI [{cb[0]:+.2f},{cb[1]:+.2f}], front CI [{cf[0]:+.2f},{cf[1]:+.2f}]'
    cost_m = {c: _metric_cost(c) for c in cand}
    cost = {c: v[0] for c, v in cost_m.items()}
    rec = min(tied, key=lambda c: (round(cost[c], 1), bag[c]))

    # ── label-noise floor (finding 30) ─────────────────────────────────────────
    _, _, _, _, idx = load_labelled_pool(pool_dir)
    se_pts, n_exact = ruler_label_se(pool_dir, idx)
    floor_band, floor_front = label_noise_floor(y, mem, se_pts, seed=seed)
    floor_obj = {'front': floor_front, 'band': floor_band,
                 'both': 0.5 * (floor_front + floor_band)}[objective]

    full = []
    for c in cand:
        o, f, b = _objective(col[c], y, mem, objective)
        full.append({'metric': c, 'objective': round(o, 3), 'front': round(f, 3),
                     'band': round(b, 3), 'bagged': round(bag[c], 3),
                     'pick_freq': freq.get(c, 0) / n_splits,
                     'cost_s_per_arch': round(cost[c], 1),
                     'cost_measured': cost_m[c][1], 'tied': c in tied})
    full.sort(key=lambda r: r['bagged'])
    warnings = []
    if n < 65:
        warnings.append(f"only {n} labelled archs: selection is noisy below ~65 "
                        f"(finding 27); label more before trusting the pick.")
    if len(cand) > 10 and n < 125:
        warnings.append(f"{len(cand)} candidates on {n} archs risks best-of-N "
                        f"overfit (finding 26); prefer a short candidate list.")
    if n_exact < len(idx):
        warnings.append(f"label SE is an UPPER BOUND for {len(idx) - n_exact}/{len(idx)} "
                        f"archs (no per-example RULER dump), so the noise floor is too.")
    from utils.metric_specs import spec_sha8
    out = {
        'created': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'pool': os.path.normpath(pool_dir),
        'model': os.path.basename(os.path.normpath(pool_dir)).split('_')[1],
        'recommended': rec, 'spec': spec_sha8(rec),
        'objective': objective, 'chat_only': bool(chat_only),
        'n_labelled_archs': int(n), 'n_candidates': len(cand),
        'selection': {'bagged_best': best, 'tie_tol': tie_tol,
                      'tied': {c: {'cost_s_per_arch': round(cost[c], 1),
                                   'bagged': round(bag[c], 3), 'why': why[c]}
                               for c in tied},
                      'rule': 'cheapest among candidates statistically tied with the bagged best'},
        'recommended_regret': {'front': round(float(uf[rec].mean()), 3),
                               'band': round(float(ub[rec].mean()), 3),
                               'bagged_half_pool': round(bag[rec], 3)},
        'label_noise_floor': {'band': round(floor_band, 3), 'front': round(floor_front, 3),
                              'objective': round(floor_obj, 3),
                              'label_se_median_pts': round(float(np.median(se_pts)), 3),
                              'se_exact_archs': int(n_exact),
                              'se_is_upper_bound': bool(n_exact < len(idx))},
        'selection_procedure_held_out': {'mean': round(float(held.mean()), 3),
                                         'p10': round(float(np.percentile(held, 10)), 3),
                                         'p90': round(float(np.percentile(held, 90)), 3)},
        'ranking': full[:12], 'warnings': warnings,
    }
    if verbose:
        print(f"[recommend] {out['model']}: {n} labelled archs, {len(cand)} "
              f"search-legal candidates{' (chat only)' if chat_only else ''}, "
              f"objective={objective}")
        print(f"  {'metric':34s} {'front':>6} {'band':>6} {'bagged':>7} {'s/arch':>7} {'tied':>5}")
        for r in full[:10]:
            print(f"  {r['metric']:34s} {r['front']:6.2f} {r['band']:6.2f} "
                  f"{r['bagged']:7.2f} {r['cost_s_per_arch']:6.1f}{'' if r['cost_measured'] else '~'} "
                  f"{'yes' if r['tied'] else '':>5}")
        bnd = '<=' if out['label_noise_floor']['se_is_upper_bound'] else '~'
        print(f"  label-noise floor {bnd} band {floor_band:.2f}  front {floor_front:.2f}   "
              f"(median label SE {np.median(se_pts):.2f} pts, exact for {n_exact}/{len(idx)})")
        print(f"  bagged best: {best} ({cost[best]:.1f} s/arch);  {len(tied)} statistically tied")
        print(f"  RECOMMENDED: {rec}  ({cost[rec]:.1f} s/arch)  front {uf[rec].mean():.2f} "
              f"band {ub[rec].mean():.2f}  vs floor {floor_front:.2f}/{floor_band:.2f}")
        for w in warnings:
            print(f"  WARNING: {w}")
    return out


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def recommendation_path(pool_dir):
    """Canonical home of a campaign's recommendation: save/metric_recommend/<pool>.json.
    NOT inside the pool: correlation campaign dirs are created by the Docker jobs and
    are owned by root, so the host user cannot write there. save/ is user-owned and
    already gitignored, and one fixed location means there is never a question of
    which copy is current."""
    return os.path.join(_REPO, 'save', 'metric_recommend',
                        os.path.basename(os.path.normpath(pool_dir)) + '.json')


def resolve_recommended(pool_dir, model_name=None):
    """What METRIC_TASK=auto:<pool> becomes. Fails loudly on anything stale."""
    # keyed by the pool's basename, so it resolves the same from any CWD (the sbatch
    # ports resolve on the HOST from $SLURM_SUBMIT_DIR, which is not the repo)
    p = recommendation_path(pool_dir)
    if not os.path.exists(p):
        raise SystemExit(f"[auto] no recommendation for this campaign ({p}). Run: "
                         f"python -m utils.bench_calib --recommend {pool_dir}")
    with open(p) as f:
        rec = json.load(f)
    if model_name and rec.get('model') and rec['model'] != model_name:
        raise SystemExit(f"[auto] {p} was made for {rec['model']}, not "
                         f"{model_name}. Use that model's own campaign.")
    from utils.metric_specs import spec_sha8
    if spec_sha8(rec['recommended']) != rec.get('spec'):
        raise SystemExit(f"[auto] the registry definition of {rec['recommended']} "
                         f"changed since it was recommended; re-run --recommend.")
    return rec['recommended']


def _main():
    import argparse
    ap = argparse.ArgumentParser(
        description='pre-search metric recommender (see block comment above)')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--recommend', metavar='POOL_DIR')
    g.add_argument('--resolve', metavar='POOL_DIR')
    g.add_argument('--build_costs', metavar='LOG_DIR',
                   help='measure per-metric seconds from correlation_eval logs')
    ap.add_argument('--model_name', default=None)
    ap.add_argument('--chat_only', action='store_true')
    ap.add_argument('--candidates', nargs='+', default=None)
    ap.add_argument('--objective', choices=['both', 'front', 'band'], default='both')
    ap.add_argument('--n_splits', type=int, default=40)
    ap.add_argument('--tie_tol', type=float, default=0.3,
                    help='max RULER-pt gap to the bagged best for a candidate to count as tied')
    ap.add_argument('--no_write', action='store_true')
    a = ap.parse_args()
    if a.resolve:
        print(resolve_recommended(a.resolve, a.model_name))
        return
    if a.build_costs:
        p, n = build_costs(a.build_costs)
        print(f"  measured {n} metrics -> {p}")
        return
    out = recommend_search_metric(a.recommend, candidates=a.candidates,
                                  chat_only=a.chat_only, objective=a.objective,
                                  n_splits=a.n_splits, tie_tol=a.tie_tol)
    if not a.no_write:
        p = recommendation_path(a.recommend)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            json.dump(out, f, indent=1)
        print(f"  wrote {p}")


if __name__ == '__main__':
    _main()
