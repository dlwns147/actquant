"""
Test: method A (ratio) vs method B (absolute diff) for constraint formulation
- Scenario 1: comp_obj_min = [2.5, 2.0]  (모두 > 0, 정상 케이스)
- Scenario 2: comp_obj_min = [0, 2.0]    (하나가 0 → 버그 케이스)
- Scenario 3: comp_obj_min = [0, 0]      (모두 0)

각 시나리오에서:
  - CV 분포
  - feasible 해 비율
  - 최종 Pareto front 비교
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

SEED = 42


# ------------------------------------------------------------------
# Synthetic problem: 2D objective (x0, x1), 2 constraints per obj
# True Pareto front: x0+x1 = const, x0∈[min0, max0], x1∈[min1, max1]
# ------------------------------------------------------------------

class ProblemA(Problem):
    """Method A: ratio form  (1 - info/min, info/max - 1)"""
    def __init__(self, obj_min, obj_max):
        super().__init__(n_var=2, n_obj=2, n_ieq_constr=4, xl=0.0, xu=10.0)
        self.obj_min = np.array(obj_min, dtype=float)
        self.obj_max = np.array(obj_max, dtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = x.copy()  # objective = variable itself
        g = np.zeros((len(x), 4))
        for j in range(2):
            if self.obj_min[j] != 0:
                g[:, 2*j]   = 1 - f[:, j] / self.obj_min[j]
            else:
                g[:, 2*j]   = 0.0   # min=0 → always feasible
            if self.obj_max[j] != 0:
                g[:, 2*j+1] = f[:, j] / self.obj_max[j] - 1
            else:
                g[:, 2*j+1] = 0.0
        out["F"] = f
        out["G"] = g


class ProblemB(Problem):
    """Method B: absolute diff form  (min - info, info - max)"""
    def __init__(self, obj_min, obj_max):
        super().__init__(n_var=2, n_obj=2, n_ieq_constr=4, xl=0.0, xu=10.0)
        self.obj_min = np.array(obj_min, dtype=float)
        self.obj_max = np.array(obj_max, dtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f = x.copy()
        g = np.zeros((len(x), 4))
        for j in range(2):
            g[:, 2*j]   = self.obj_min[j] - f[:, j]   # min=0 → g=-f ≤ 0, always feasible
            g[:, 2*j+1] = f[:, j] - self.obj_max[j]
        out["F"] = f
        out["G"] = g


def run_nsga2(problem, seed=SEED, pop_size=100, n_gen=50):
    algo = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
    res = minimize(problem, algo, ('n_gen', n_gen), seed=seed, verbose=False)
    return res


def pareto_front(F):
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return F[front]


def summarize(name, res, obj_min, obj_max):
    pop = res.pop
    F = pop.get("F")
    CV = pop.get("CV").flatten()
    feasible_mask = CV <= 0.0

    n_total = len(pop)
    n_feasible = feasible_mask.sum()
    print(f"\n[{name}]  obj_min={obj_min}  obj_max={obj_max}")
    print(f"  총 해: {n_total},  feasible: {n_feasible} ({100*n_feasible/n_total:.1f}%)")
    print(f"  CV  — mean: {CV.mean():.4f}, max: {CV.max():.4f}, has nan/inf: {not np.isfinite(CV).all()}")

    if n_feasible > 0:
        F_feas = F[feasible_mask]
        pf = pareto_front(F_feas)
        print(f"  Pareto front 해 수: {len(pf)}")
        print(f"  Pareto front f0 range: [{pf[:,0].min():.3f}, {pf[:,0].max():.3f}]")
        print(f"  Pareto front f1 range: [{pf[:,1].min():.3f}, {pf[:,1].max():.3f}]")
    else:
        print("  feasible 해 없음")

    return F, CV, feasible_mask


def plot_scenario(ax, name, F, CV, feasible_mask, obj_min, obj_max, color):
    infeas = F[~feasible_mask]
    feas   = F[feasible_mask]
    if len(infeas):
        ax.scatter(infeas[:,0], infeas[:,1], c='lightgray', s=10, label='infeasible')
    if len(feas):
        ax.scatter(feas[:,0], feas[:,1], c=color, s=15, alpha=0.7, label='feasible')
        pf = pareto_front(feas)
        pf = pf[np.argsort(pf[:,0])]
        ax.plot(pf[:,0], pf[:,1], 'k-', lw=1.5, label='Pareto front')
    ax.axvline(obj_min[0], color='blue', ls='--', lw=0.8, alpha=0.6)
    ax.axvline(obj_max[0], color='blue', ls='--', lw=0.8, alpha=0.6)
    ax.axhline(obj_min[1], color='red',  ls='--', lw=0.8, alpha=0.6)
    ax.axhline(obj_max[1], color='red',  ls='--', lw=0.8, alpha=0.6)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('f0'); ax.set_ylabel('f1')
    ax.legend(fontsize=7)


# ------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------

scenarios = [
    {"name": "Scenario 1: min=[2.5, 2.0]",  "obj_min": [2.5, 2.0], "obj_max": [7.0, 7.0]},
    {"name": "Scenario 2: min=[0,   2.0]",  "obj_min": [0,   2.0], "obj_max": [7.0, 7.0]},
    {"name": "Scenario 3: min=[0,   0  ]",  "obj_min": [0,   0  ], "obj_max": [7.0, 7.0]},
]

fig, axes = plt.subplots(len(scenarios), 2, figsize=(10, 4*len(scenarios)))
fig.suptitle('Method A (ratio) vs Method B (absolute diff)', fontsize=12)

for row, sc in enumerate(scenarios):
    print("\n" + "="*60)
    print(sc["name"])
    print("="*60)

    pb_a = ProblemA(sc["obj_min"], sc["obj_max"])
    pb_b = ProblemB(sc["obj_min"], sc["obj_max"])

    res_a = run_nsga2(pb_a)
    res_b = run_nsga2(pb_b)

    Fa, CVa, feas_a = summarize("Method A (ratio)", res_a, sc["obj_min"], sc["obj_max"])
    Fb, CVb, feas_b = summarize("Method B (abs)",   res_b, sc["obj_min"], sc["obj_max"])

    # Pareto front similarity (if both have feasible solutions)
    if feas_a.sum() > 0 and feas_b.sum() > 0:
        pfa = pareto_front(Fa[feas_a])
        pfb = pareto_front(Fb[feas_b])
        print(f"\n  Pareto front 비교:")
        print(f"    Method A — f0 mean={pfa[:,0].mean():.3f}, f1 mean={pfa[:,1].mean():.3f}")
        print(f"    Method B — f0 mean={pfb[:,0].mean():.3f}, f1 mean={pfb[:,1].mean():.3f}")

    # nan/inf check
    for method, CV in [("A", CVa), ("B", CVb)]:
        bad = ~np.isfinite(CV)
        if bad.any():
            print(f"  ⚠️  Method {method}: nan/inf in CV at {bad.sum()} solutions!")
        else:
            print(f"  ✅  Method {method}: CV all finite")

    plot_scenario(axes[row, 0], f"Method A — {sc['name']}", Fa, CVa, feas_a, sc["obj_min"], sc["obj_max"], 'steelblue')
    plot_scenario(axes[row, 1], f"Method B — {sc['name']}", Fb, CVb, feas_b, sc["obj_min"], sc["obj_max"], 'tomato')

plt.tight_layout()
plt.savefig('/NAS/SJ/actquant/search/test_constraint_methods.png', dpi=120)
print("\n\n그래프 저장: test_constraint_methods.png")
