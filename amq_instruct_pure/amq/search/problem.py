import numpy as np

from pymoo.core.problem import Problem

from utils.func import get_bits_usage
from evaluation.data_io import task_jsd_mean


class AuxiliarySingleLevelProblem(Problem):
    """ The optimization problem for finding the next N candidate architectures.

    Objective 0 is the mean over per-dataset JSD predictors, each normalized
    by that task's population JSD mean (so every task contributes equally
    regardless of its absolute JSD scale).
    """

    def __init__(self, search_space, jsd_predictors,
                 config, group_size):
        n_block, n_linear = search_space.n_block, search_space.n_linear
        super().__init__(n_var=n_block * (n_linear), n_obj=2, n_constr=2, type_var=int)

        self.search_space = search_space
        self.jsd_predictors = jsd_predictors
        self.xl = np.zeros((n_linear, n_block))
        self.xu = np.ones((n_linear, n_block))

        for linear_idx, linear in enumerate(config['linear']):
            self.xu[linear_idx] = len(self.search_space.bits_range) - 1

        self.config = config
        self.group_size = group_size

        for pass_linear in self.search_space.pass_linear_list:
            blk, linear = pass_linear.split('.', maxsplit=1)
            blk = int(blk)

            linear_idx = config['linear'].index(linear)
            self.xl[linear_idx, int(blk)] = len(self.search_space.bits_range) - 1

        self.xl = self.xl.flatten()
        self.xu = self.xu.flatten()

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)
        g = np.full((x.shape[0], self.n_constr), np.nan)

        x_pred = self.search_space.decode_encode_predictor(x)
        jsd_per_ds = np.stack(
            [np.asarray(p.predict(x_pred)).reshape(-1) / task_jsd_mean(ds)
             for ds, p in self.jsd_predictors.items()],
            axis=0,
        )
        metrics = jsd_per_ds.mean(axis=0)
        # metrics = jsd_per_ds

        for i, (_x, metric) in enumerate(zip(x, metrics)):
            architecture = self.search_space.decode(_x)
            bits_usage = get_bits_usage(architecture, self.config, self.group_size)
            f[i, 0] = metric
            f[i, 1] = bits_usage
            # f[i, 0] = bits_usage
            # f[i, 1:] = metrics

            g[i, 0] = 1 - bits_usage / (self.search_space.bits_range[0] + 32 / self.group_size)
            g[i, 1] = bits_usage / (self.search_space.bits_range[-1] + 32 / self.group_size) - 1

        out["F"] = f
        out["G"] = g

class SubsetProblem(Problem):
    """ select a subset to diversify the pareto front """
    def __init__(self, candidates, archive, K):
        super().__init__(n_var=len(candidates), n_obj=1,
                         n_constr=1, xl=0, xu=1, type_var=bool)
        self.archive = archive
        self.candidates = candidates
        self.n_max = K

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)

        for i, _x in enumerate(x):
            # append selected candidates to archive then sort
            tmp = np.sort(np.concatenate((self.archive, self.candidates[_x])))
            f[i, 0] = np.std(np.diff(tmp))
            g[i, 0] = (self.n_max - np.sum(_x)) ** 2

        out["F"] = f
        out["G"] = g