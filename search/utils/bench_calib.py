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
