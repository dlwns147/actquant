import os
import json
import torch
import argparse
import numpy as np
from evaluator import LlamaEvaluator
from tqdm import tqdm
from time import time
from copy import deepcopy
import csv

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.indicators.hv import Hypervolume
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PolynomialMutation

from search_space.llama import LlamaSearchSpace
from predictor.factory import get_predictor
from utils.func import get_net_info, init_accelerator, set_seed, get_correlation
from utils.ga import MySampling, BinaryCrossover, MyMutation, IntPolynomialMutation, MyTwoPointCrossover, MyUniformCrossover, IntegerFromFloatMutation, IntMutation

import warnings
warnings.simplefilter("ignore")

class Search:
    def __init__(self, config, accelerator, device_map, kwargs):
        """
        탐색 프로세스를 위한 전체 환경을 설정하는 초기화 메서드.
        사용자 입력, 설정 파일, 가속기 등을 기반으로 탐색에 필요한 모든 구성 요소를 준비합니다.
        """
        self.args = deepcopy(kwargs)
        self.config = config
        self.device_map = device_map

        # --- 탐색 기본 설정 ---
        self.save_path = kwargs.pop('save', 'save')  # 결과 저장 경로
        self.result_file = kwargs.pop('result_file', 'results.txt')  # 결과 파일명
        self.resume = kwargs.pop('resume', None)  # 중단된 탐색을 재개할 체크포인트 경로
        self.iterations = kwargs.pop('iterations', 30)  # 탐색 반복 횟수
        self.n_doe = kwargs.pop('n_doe', 100)  # 초기 실험설계(DOE) 샘플 개수
        self.n_iter = kwargs.pop('n_iter', 8)  # 각 반복마다 실제 평가를 수행할 아키텍처 개수
        self.predictor = kwargs.pop('predictor', 'rbf')  # 사용할 대리 모델(surrogate model) 종류
        self.dataset = kwargs.pop('dataset', 'wikitext2')  # 탐색에 사용할 데이터셋
        self.loss_func = kwargs.pop('loss_func', 'jsd') # 손실 함수

        # --- 양자화 및 모델 관련 설정 ---
        self.method = kwargs.pop('method', '') # 양자화 방법 (e.g., hqq, awq)
        self.quant_model_paths = kwargs.pop('quant_model_paths', []) # 미리 양자화된 모델 경로들

        model_path = kwargs.pop('model_path', 'meta-llama')
        model_name = kwargs.pop('model_name', 'Llama-2-7b-hf')
        model_id=f'{model_path}/{model_name}'
        self.metric = kwargs.pop('metric', 'loss') # 평가 지표 (ppl 또는 loss)
        
        # --- Outlier 채널 관련 설정 ---
        outlier_path = kwargs.pop('outlier_path' , '')
        base_outlier_bits = sorted(kwargs.pop('base_outlier_bits', []))
        n_outlier = kwargs.pop('n_outlier' , 0)
        assert (outlier_path and base_outlier_bits and n_outlier > 0) or (not outlier_path and not base_outlier_bits and n_outlier == 0), "outlier 관련 인자는 함께 사용되어야 합니다."
        
        # Outlier 채널을 고려하여 평균 비트 수를 계산
        outlier_bits = {l: [] for l in config['linear']}
        if outlier_path and base_outlier_bits and n_outlier > 0:
            for linear in config['linear']:
                for base_bits in base_outlier_bits:
                    _, in_dim = config['linear_shape'][linear]
                    avg_linear_bits = ((in_dim - n_outlier) * base_bits + n_outlier * 16) / (in_dim)
                    outlier_bits[linear].append(avg_linear_bits)

        # --- 탐색 공간(Search Space) 정의를 위한 비트 및 그룹 크기 설정 ---
        w_bits = kwargs.pop('w_bits', []) # 가중치(Weight) 비트 후보
        assert len(w_bits) == len(self.quant_model_paths)
        k_bits = kwargs.pop('k_bits', []) # Key 비트 후보
        v_bits = kwargs.pop('v_bits', []) # Value 비트 후보
        bits = {'w': w_bits, 'k': k_bits, 'v': v_bits}
        self.bits = bits

        w_group_size = kwargs.pop('w_group_size', 128) # 가중치 그룹 크기
        k_group_size = kwargs.pop('k_group_size', [128]) # Key 그룹 크기 후보
        v_group_size = kwargs.pop('v_group_size', [128]) # Value 그룹 크기 후보
        self.group_size = {'w': w_group_size, 'k': k_group_size, 'v': v_group_size}

        self.residual_length = kwargs.pop('residual_length', 128)

        # --- 다중 목표 최적화(Multi-objective Optimization) 설정 ---
        self.comp_obj = kwargs.pop('comp_obj', ['wbits', 'kvbits'])  # 최적화할 복잡도 목표 (e.g., 가중치 비트, KV 캐시 비트)
        self.comp_obj_min = kwargs.pop('comp_obj_min', [min(w_bits), min(k_bits)]) # 복잡도 목표의 최솟값
        self.comp_obj_max = kwargs.pop('comp_obj_max', [max(w_bits), max(k_bits)]) # 복잡도 목표의 최댓값
        assert len(self.comp_obj) == len(self.comp_obj_min) and len(self.comp_obj_min) == len(self.comp_obj_max)

        # --- 민감도 분석 기반 탐색 공간 축소 ---
        # 민감도가 높은 (성능 영향이 큰) 모듈은 탐색에서 제외하여 효율성을 높임
        self.sensitivity_result_path = kwargs.pop('sensitivity_result_path', '')
        total_module = dict()
        total_sensitivity = dict()
        pass_module = {'w': [], 'k': [], 'v': []} # 탐색에서 제외할(통과시킬) 모듈 리스트       
        
        if self.sensitivity_result_path:
            # 민감도 분석 결과 파일을 읽어옴
            for target in pass_module.keys():
                if any([target in obj for obj in self.comp_obj]):
                    with open(os.path.join(self.sensitivity_result_path, f'{target}.csv'), 'r') as f:
                        module_list, sensitivity = list(csv.reader(f))
                        sensitivity = list(map(float, sensitivity))
                        total_module[target] = list(map(int, module_list)) if target in ['k', 'v'] else module_list
                        total_sensitivity[target] = sensitivity
            total_sensitivity_list = np.nan_to_num(np.concatenate(list(total_sensitivity.values())), nan=float('inf'))
            upper_bound = np.median(total_sensitivity_list) * 2
            print(f'upper_bound: {upper_bound}')
            pass_idx_list = np.where(total_sensitivity_list > upper_bound)[0].tolist()

            start = 0
            for target in pass_module.keys():
                if any([target in obj for obj in self.comp_obj]):
                    end = start + len(total_module[target])
                    for idx in pass_idx_list:
                        if start <= idx and idx < end:
                            pass_module[target].append(total_module[target][idx - start])
                    start = end

        self.pass_module = pass_module
        self.args['pass_module'] = pass_module
        print(f'pass_module: {pass_module}')

        # --- 핵심 컴포넌트 객체 생성 ---
        # 1. LlamaEvaluator: 특정 아키텍처의 실제 성능(PPL, loss)을 평가하는 역할
        self.evaluator = LlamaEvaluator(
            self.config,
            accelerator=accelerator,
            model_id=model_id,
            method=self.method,
            # quant_model_bits=self.quant_model_bits,
            quant_model_paths=self.quant_model_paths,
            outlier=torch.load(outlier_path) if outlier_path else None,
            seqlen=kwargs.pop('seqlen', 2048),
            n_sample=kwargs.pop('n_sample', 128),
            datasets=[self.dataset],
            loss_func=self.loss_func,
            device_map=device_map,
            bits=bits,
            group_size=self.group_size,
            residual_length=self.residual_length,
            use_flash=kwargs.pop('use_flash', False),
            quant_kv_output=kwargs.pop('quant_kv_output', True),
            k_quant_per=kwargs.pop('k_quant_per', 'channel'),
            v_quant_per=kwargs.pop('v_quant_per', 'token'),
        )
        # 2. LlamaSearchSpace: 탐색할 하이퍼파라미터(비트, 그룹 크기)의 범위와 제약 조건을 정의
        self.search_space = LlamaSearchSpace(
            bits=self.bits,
            group_size=self.group_size,
            pass_module=self.pass_module,
            comp_obj=self.comp_obj,
            comp_obj_min=self.comp_obj_min,
            comp_obj_max=self.comp_obj_max,
            config=self.config,
            outlier_bits=outlier_bits,
            only_outlier_bits=kwargs.pop('only_outlier_bits', False),
        )
        
        # --- 유전 알고리즘(GA) 하이퍼파라미터 설정 ---
        self.ga_pop_size = kwargs.pop('ga_pop_size', 40)
        self.subset_pop_size = kwargs.pop('subset_pop_size', 100)
        self.debug = kwargs.pop('debug', False)
        self.ga_algorithm = kwargs.pop('ga_algorithm', 'nsga2')
        self.max_value = kwargs.pop('max_value', 50)
        self.mut_prob = kwargs.pop('mut_prob', 0.1)
        self.crossover_prob = kwargs.pop('crossover_prob', 0.9)
        self.save_iter = kwargs.pop('save_iter', 1)
        accelerator.wait_for_everyone()
        
    def search(self, accelerator):
        """
        신경망 아키텍처 탐색(NAS)의 메인 루프.
        DoE -> [Predictor 학습 -> 후보 생성(GA) -> 후보 평가 -> Archive 업데이트] 반복
        """
        total_start = time()
        start_it = 1
        
        if self.resume:
            # 체크포인트가 있으면 해당 지점부터 탐색 재개
            archive, start_it = self._resume_from_dir()

        else:
            # 1. 초기 집단 생성 (Design Of Experiment, DoE)
            # 탐색 공간에서 무작위로 초기 아키텍처 집단을 샘플링하여 초기 데이터셋(archive)을 구축
            archive = []

            if accelerator.is_main_process:
                if self.iterations < 1:
                    # 반복이 없을 경우, 샘플링만 수행
                    arch_doe = self.search_space.sample(
                        n_samples=self.n_doe,
                        pool=[x[0] for x in archive])
                else:
                    # 초기 집단을 생성
                    arch_doe = self.search_space.initialize(self.n_doe, pool=[x[0] for x in archive])
            else:
                arch_doe = list()
            arch_doe = accelerator.gather_for_metrics(arch_doe, use_gather_object=True)
            accelerator.wait_for_everyone()

            # 2. 초기 집단 평가 (High-fidelity Evaluation)
            # 샘플링된 아키텍처들의 실제 성능과 복잡도를 측정
            metric, complexity = self._evaluate(archs=arch_doe, accelerator=accelerator)

            if accelerator.is_main_process:
                # 평가된 아키텍처들을 archive에 저장
                for a, m, c in zip(arch_doe, metric, complexity):
                    archive.append([a, m, *c])

        if accelerator.is_main_process:
            # 하이퍼볼륨 계산을 위한 참조점(nadir point) 설정
            ref_pt = np.array([np.max([x[i] for x in archive]) for i in range(1, len(self.comp_obj) + 2)])
            accelerator.print(f'data preparation time : {time() - total_start:.2f}s')
        accelerator.wait_for_everyone()

        # 3. 반복적 탐색 메인 루프
        for it in range(start_it, self.iterations + 1):
            if accelerator.is_main_process:
                accelerator.print(self.args)
                iter_start = time()

                # 3-1. 성능 예측 대리 모델(Surrogate Model) 학습
                # archive에 축적된 데이터로 아키텍처의 성능을 빠르게 예측하는 모델을 학습
                predictor_start = time()
                metric_predictor, a_metric_pred = self._fit_predictor(archive, device=accelerator.device)
                predictor_time = time() - predictor_start

                # 3-2. 다음 세대 후보 아키텍처 탐색 및 선택
                # 대리 모델을 이용하는 유전 알고리즘(NSGA2)을 통해 유망한 후보군을 생성하고, 그 중 일부를 선택
                next_start = time()
                candidates, c_metric_pred = self._next(archive, metric_predictor, self.n_iter)
                next_time = time() - next_start
            else:
                candidates = list()
            accelerator.wait_for_everyone()
            candidates = accelerator.gather_for_metrics(candidates, use_gather_object=True)

            # 3-3. 후보 아키텍처 실제 성능 평가
            # 선택된 후보들의 실제 성능과 복잡도를 측정
            c_metric, complexity = self._evaluate(archs=candidates, accelerator=accelerator)

            if accelerator.is_main_process:
                # 대리 모델의 예측 성능 검증 (RMSE, Spearman's Rho 등)
                rmse, rho, tau = get_correlation(
                    np.vstack((a_metric_pred, c_metric_pred)), np.array([x[1] for x in archive] + c_metric))

                # 3-4. Archive 업데이트
                # 새로 평가된 후보들을 archive에 추가
                for a, m, c in zip(candidates, c_metric, complexity):
                    archive.append([a, m, *c])

                # 파레토 전선(Pareto front)의 개선도를 나타내는 하이퍼볼륨(Hypervolume) 계산
                hv = self._calc_hv(
                    ref_pt, np.column_stack([[x[i] for x in archive] for i in range(1, len(self.comp_obj) + 2)]))
                    # ref_pt, np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))) 

                iter_time = time() - iter_start
                # 반복 단계별 통계 출력
                accelerator.print(f"Iter {it}: hv = {hv:.2f}, iter time : {(time() - iter_start):.2f}s, predictor_time : {predictor_time:.2f}, next_time : {next_time:.2f}")
                accelerator.print(f"fitting {self.predictor}: RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, Kendall’s Tau = {tau:.4f}")
                accelerator.print(f'iteration time : {iter_time:.2f}s')

                # dump the statistics
                if it % self.save_iter == 0:
                    os.makedirs(self.save_path, exist_ok=True)
                    with open(os.path.join(self.save_path, "iter_{}.stats".format(it)), "w") as handle:
                        json.dump({'archive': archive, 'candidates': archive[-self.n_iter:], 'hv': hv,
                                'surrogate': {
                                    'model': self.predictor, 'name': metric_predictor.name,
                                    'winner': metric_predictor.winner if self.predictor == 'as' else metric_predictor.name,
                                    'rmse': rmse, 'rho': rho, 'tau': tau, 'total_time': iter_time}, 'iteration' : it}, handle)
                    if self.debug:
                        import matplotlib.pyplot as plt
                        n_obj = len(self.comp_obj)
                        comp_np = np.array(complexity)
                        # fig, axes = plt.subplots(nrows=1, ncols=n_obj + 1, figsize=(5 * (n_obj + 1), 5))
                        fig, axes = plt.subplots(nrows=1, ncols=n_obj, figsize=(5 * n_obj, 5))
                        if not isinstance(axes, np.ndarray):
                            axes = [axes]
                        for i in range(n_obj):
                            comp = np.array([x[i+2] for x in archive])  # comp obj
                            perf = np.array([x[1] for x in archive])  # performance
                            axes[i].scatter(comp, perf, s=5, facecolors='none', edgecolors='b', label='archive')
                            comp = comp_np[:, i]
                            perf = np.array(c_metric)
                            axes[i].scatter(comp, perf, s=10, color='r', label='candidates evaluated')
                            perf = c_metric_pred[:, 0]
                            axes[i].scatter(comp, perf, s=10, facecolors='none', edgecolors='g', label='candidates predicted')
                            axes[i].legend(loc="upper right")
                            axes[i].set_xlabel(f'f{i+2}')
                            axes[i].grid(c='0.8') 
                        axes[0].set_ylabel('f1')
                        # axes[-1].scatter(np.array([x[2] for x in archive]), np.array([x[3] for x in archive]), s=5, facecolors='none', edgecolors='b', label='archive')
                        # axes[-1].scatter(np.array(complexity)[:, 0], np.array(complexity)[:, 1], s=10, color='r', label='candidates')
                        # axes[-1].legend(loc="upper right")
                        # axes[-1].set_xlabel('f2')
                        # axes[-1].set_ylabel('f3')
                        # axes[-1].grid(c='0.8') 
                        fig.tight_layout() 
                        plt.savefig(os.path.join(self.save_path, 'iter_{}.png'.format(it)))
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            # 최종 결과 저장
            total_time_elapsed = time() - total_start
            accelerator.print(f'total time elapsed : {total_time_elapsed:.2f}s')

            sentences = []
            for k, v in self.args.items():
                sentences.append(f"{k}: {v}\n")
            sentences.append(f'Total time: {total_time_elapsed:.2f}s')
            # sentences.append("\n")

            with open(os.path.join(self.save_path, self.result_file), 'w') as f:
                for sentence in sentences:
                    f.write(sentence)

            accelerator.print(self.args)
        return

    def _resume_from_dir(self):
        """ 이전 탐색 상태(체크포인트)로부터 복원 """

        with open(self.resume, 'r') as f:
            resume_file = json.load(f)
            archive = resume_file['archive'] + resume_file['candidates']
            it = resume_file['iteration']

        return archive, it + 1

    def _evaluate(self, archs, accelerator):
        """
        아키텍처 리스트를 받아 실제 성능(High-fidelity)을 평가.
        LlamaEvaluator를 호출하여 시간이 오래 걸리는 실제 추론 및 평가를 수행.
        """
        metric_list, complexity_list = [], [] # {obj: [] for obj in self.comp_obj}
        for arch in tqdm(archs, desc='Eval Arch'):
            # LlamaEvaluator를 통해 성능(metric)과 복잡도(complexity)를 얻음
            metric, complexity = self.evaluator.eval(accelerator=accelerator, arch=arch, metric=self.metric, loss_func=self.loss_func)
            metric_list.append(min(self.max_value, np.nan_to_num(metric[self.dataset], nan=self.max_value)))
            complexity_list.append([complexity[obj] for obj in self.comp_obj])

        return metric_list, complexity_list

    def _fit_predictor(self, archive, device='cpu'):
        """
        Archive 데이터를 이용해 아키텍처 성능을 예측하는 대리 모델(Surrogate model)을 학습.
        이를 통해 실제 평가 없이 빠르고 효율적으로 아키텍처의 성능을 예측 가능.
        """
        # 아키텍처를 predictor가 이해할 수 있는 숫자 벡터로 인코딩
        inputs = np.array([self.search_space.encode_predictor(x[0]) for x in archive])
        targets = np.array([x[1] for x in archive])

        kwargs = {}
        if self.predictor == 'rbf':
            # RBF 모델을 위한 추가 파라미터 설정 (상한/하한 등)
            n_block = self.config['n_block']
            n_linear = self.config['n_linear']
            lb = np.zeros((n_linear + 2, n_block))
            ub = np.ones((n_linear + 2, n_block))
            
            for linear_idx, linear in enumerate(self.config['linear']):
                ub[linear_idx] = len(getattr(self.search_space, f"{linear.split('.')[-1]}_option")) - 1            
            ub[n_linear] = len(self.search_space.k_option) - 1
            ub[n_linear + 1] = len(self.search_space.v_option) - 1
            
            lb = np.delete(lb.flatten(), self.search_space.pass_idx_list, axis=-1)
            ub = np.delete(ub.flatten(), self.search_space.pass_idx_list, axis=-1)

            kwargs = {'lb': lb, 'ub': ub}

        # predictor 팩토리를 통해 지정된 종류의 대리 모델을 생성하고 학습
        metric_predictor = get_predictor(self.predictor, inputs, targets, device=device, **kwargs)

        return metric_predictor, metric_predictor.predict(inputs)
    
    def _next(self, archive, predictor, K):
        """ 
        대리 모델을 사용하여 다음 세대의 유망한 후보 아키텍처 K개를 생성.
        유전 알고리즘(NSGA-II)을 활용하여 파레토 최적 해를 탐색.
        """

        # 1. 현재까지 발견된 파레토 최적 해(non-dominated front)를 찾음
        F = np.column_stack([[x[i] for x in archive] for i in range(1, len(self.comp_obj) + 2)])
        # 이렇게하면 맨 앞의 front만 추출됨, https://github.com/anyoptimization/pymoo/blob/main/pymoo/util/nds/non_dominated_sorting.py
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # 파레토 최적 해들을 유전 알고리즘의 초기 집단(seed)으로 사용
        nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

        # 2. 다중 목표 최적화를 위한 유전 알고리즘(NSGA2) 설정
        # Crossover, Mutation 등의 연산자 정의
        # 현재 세대인 self.ga_pop_size만큼만 최적회 생성에 참여
        method = NSGA2(pop_size=self.ga_pop_size, sampling=nd_X,  # 현재 파레토 최적 해들로 초기화
            # 왜 n_offsprings=1인지?
            crossover=BinomialCrossover(prob=self.crossover_prob, n_offsprings=1),
            mutation=IntMutation(prob=self.mut_prob),
            eliminate_duplicates=True)
        
        # 3. 최적화 문제 정의
        # 대리 모델을 목적 함수로 사용하여, 예측 성능과 복잡도를 최적화하는 문제
        problem = AuxiliarySingleLevelProblem(self.search_space, predictor, self.config, self.comp_obj, self.comp_obj_max, self.comp_obj_min, self.group_size)
        
        # 4. 유전 알고리즘 실행
        # 이러면 자손을 생성하는 과정 20번 함. 즉 20세대까지 만듦
        res = minimize(problem, method, termination=('n_gen', 20), save_history=True, verbose=True)
        
        # 이미 평가된 아키텍처는 제외
        not_duplicate = np.logical_not(
            [x in [x[0] for x in archive] for x in [self.search_space.decode(x) for x in res.pop.get("X")]])
        print(f'not_duplicate : {sum(not_duplicate)}')

        # 5. 후보군 중 최종 K개 선택 (Subset Selection)
        # 생성된 많은 후보들 중에서 다양성을 고려하여 실제 평가를 수행할 K개를 선택
        indices = self._subset_selection(res.pop[not_duplicate], F[front, 1:], K, self.subset_pop_size)
        pop = res.pop[not_duplicate][indices]

        candidates = []
        for x in pop.get("X"):
            candidates.append(self.search_space.decode(x))

        # 선택된 후보들과 그 예측 성능을 반환
        return candidates, predictor.predict(self.search_space.decode_encode_predictor(pop.get("X")))

    def _subset_selection(self, pop, nd_F, K, pop_size):
        """
        생성된 후보군(pop) 중에서 파레토 전선의 다양성을 최대화하는 K개의 부분집합을 선택.
        """
        # 후보들의 복잡도 목표값들과, 기존 파레토 전선(nd_F)을 입력으로 받음
        problem = SubsetProblem(pop.get("F")[:, 1:], nd_F, K, len(self.comp_obj))
        algorithm = GA(
            pop_size=pop_size, sampling=MySampling(), crossover=BinaryCrossover(),
            mutation=MyMutation(), eliminate_duplicates=True)

        res = minimize(
            problem, algorithm, ('n_gen', 60), verbose=False)

        return res.X

    @staticmethod
    def _calc_hv(ref_pt, F, normalized=True):
        """
        파레토 전선(F)과 참조점(ref_pt)을 기반으로 하이퍼볼륨(Hypervolume)을 계산.
        하이퍼볼륨은 다중 목표 최적화에서 해 집합의 품질을 나타내는 지표.
        """
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        ref_point = 1.01 * ref_pt
        hv = Hypervolume(ref_point=ref_point).do(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv


class AuxiliarySingleLevelProblem(Problem):
    """ 
    유전 알고리즘을 위한 최적화 문제를 정의하는 클래스.
    대리 모델(predictor)을 목적 함수 중 하나로 사용하여, (예측 성능, 복잡도1, 복잡도2, ...)를 최적화.
    """

    def __init__(self, search_space, predictor, config, comp_obj, comp_obj_max, comp_obj_min, group_size):
        n_block, n_linear = search_space.n_block, search_space.n_linear
        n_comp_obj = len(search_space.comp_obj)
        # 문제 정의: 변수 개수, 목적 함수 개수, 제약 조건 개수 등
        super().__init__(n_var=n_block * (n_linear + 2), n_obj=n_comp_obj + 1, n_constr=2 * n_comp_obj, type_var=int)

        self.ss = search_space
        self.predictor = predictor
        self.comp_obj = comp_obj
        self.comp_obj_max = comp_obj_max
        self.comp_obj_min = comp_obj_min
        self.config = config
        self.group_size = group_size
        # 변수의 하한(xl)과 상한(xu) 설정
        self.xl = np.zeros((n_linear + 2, n_block))
        self.xu = np.ones((n_linear + 2, n_block))
        
        for linear_idx, linear in enumerate(config['linear']):
            self.xu[linear_idx] = len(getattr(search_space, f"{linear.split('.')[-1]}_option")) - 1
        self.xu[n_linear] = len(search_space.k_option) - 1
        self.xu[n_linear + 1] = len(search_space.v_option) - 1
        
        for pass_w_linear in search_space.pass_module['w']:
            blk, linear = pass_w_linear.split('.', 1)
            linear_idx =  config['linear'].index(linear)
            self.xl[linear_idx, int(blk)] = len(getattr(search_space, f"{linear.split('.')[-1]}_option")) - 1
        for pass_k_layer in search_space.pass_module['k']:
            self.xl[n_linear, pass_k_layer] = len(search_space.k_option) - 1
        for pass_v_layer in search_space.pass_module['v']:
            self.xl[n_linear + 1, pass_v_layer] = len(search_space.v_option) - 1

        self.xl = self.xl.flatten()
        self.xu = self.xu.flatten()

    def _evaluate(self, x, out, *args, **kwargs):
        """ 아키텍처(x)를 받아 목적 함수 값(F)과 제약 조건(G)을 계산 """
        f = np.full((x.shape[0], self.n_obj), np.nan)
        g = np.full((x.shape[0], self.n_constr), np.nan)

        # 1. 대리 모델로 성능 예측
        metrics = self.predictor.predict(self.ss.decode_encode_predictor(x))[:, 0]

        for i, (_x, metric) in enumerate(zip(x, metrics)):
            arch = self.ss.decode(_x)
            info = get_net_info(arch, self.config, self.group_size)
            
            # 2. 목적 함수 값 설정
            f[i, 0] = metric  # 첫 번째 목표: 예측 성능
            for j in range(len(self.comp_obj)):
                f[i, 1 + j] = info[self.comp_obj[j]] # 두 번째, 세 번째, ... 목표: 복잡도
                
                # 3. 제약 조건 설정 (복잡도가 min/max 범위를 벗어나지 않도록)
                g[i, 2 * j] = 1 - info[self.comp_obj[j]] / self.comp_obj_min[j]
                g[i, 2 * j + 1] = info[self.comp_obj[j]] / self.comp_obj_max[j] - 1

        out["F"] = f
        out["G"] = g

class SubsetProblem(Problem):
    """ 
    후보군 중에서 K개의 부분집합을 선택하는 문제를 정의.
    선택된 후보들을 포함했을 때 파레토 전선의 다양성(분산)을 최대화하는 것을 목표로 함.
    """
    def __init__(self, candidates, archive, K, n_obj):
        super().__init__(n_var=len(candidates), n_obj=1,
                         n_constr=1, xl=0, xu=1, type_var=bool)
        self.archive = archive
        self.candidates = candidates
        self.n_max = K

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)

        for i, _x in enumerate(x):
            # 선택된 후보(_x가 true인 것들)를 기존 archive에 추가
            tmp = np.sort(np.concatenate((self.archive, self.candidates[_x])), axis=0)
            
            # 목적 함수: 합쳐진 집합의 점들 간 거리의 표준편차를 계산 (분산 최대화)
            f[i, 0] = np.std(np.diff(tmp, axis=0))

            # 제약 조건: 정확히 K개의 후보가 선택되도록 함
            g[i, 0] = (self.n_max - np.sum(_x)) ** 2

        out["F"] = f
        out["G"] = g

def main(args):
    """ 메인 실행 함수 """
    set_seed(args.seed)

    # 설정 파일 로드 및 가속기 초기화
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)
    
    # Search 엔진 클래스 인스턴스화 및 탐색 시작
    engine = Search(config=config, accelerator=accelerator, device_map=device_map, kwargs=vars(args))
    engine.search(accelerator)
    return


if __name__ == '__main__':
    # 커맨드 라인 인자 파서 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='save',
                        help='결과를 저장할 디렉토리 위치')
    parser.add_argument('--resume', type=str, default=None,
                        help='탐색을 재개할 체크포인트 파일')
    parser.add_argument('--iterations', type=int, default=50,
                        help='탐색 반복 횟수')
    parser.add_argument('--n_doe', type=int, default=100,
                        help='초기 실험설계(DOE)를 위한 샘플 크기')
    parser.add_argument('--n_iter', type=int, default=8,
                        help='각 반복에서 평가할 아키텍처 수')
    parser.add_argument('--predictor', type=str, default='rbf',
                        help='사용할 정확도 예측 모델 (rbf/gp/cart/mlp/as)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='사용할 GPU ID')
    parser.add_argument('--model_path', type=str, default='',
                        help='모델 가중치가 있는 경로')
    parser.add_argument('--model_name', type=str, default='',
                        help='모델 이름')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='미리 양자화된 모델들의 경로')
    parser.add_argument('--w_bits', type=int, nargs='+', default=[], 
                        help='가중치(W)에 대한 비트 후보')
    parser.add_argument('--k_bits', type=int, nargs='+', default=[2, 4], 
                        help='Key에 대한 비트 후보')
    parser.add_argument('--v_bits', type=int, nargs='+', default=[2, 4], 
                        help='Value에 대한 비트 후보')

    parser.add_argument('--w_group_size', type=int, default=128,
                        help='가중치 양자화 그룹 크기')
    parser.add_argument('--k_group_size', type=int, default=128,
                        help='Key 양자화 그룹 크기')
    parser.add_argument('--v_group_size', type=int, default=128,
                        help='Value 양자화 그룹 크기')
    
    parser.add_argument('--residual_length', type=int, default=128, 
                        help='KV 캐시에서 이전 스텝 정보를 얼마나 유지할지 길이')
    parser.add_argument('--use_flash', action='store_true', help='Flash Attention 사용 여부')

    parser.add_argument('--quant_kv_output', action='store_true', help='KV 캐시의 최종 출력을 양자화할지 여부')
    parser.add_argument('--k_quant_per', type=str, choices=['channel', 'token'], 
                        help='Key를 채널 단위 또는 토큰 단위로 양자화할지 선택')
    parser.add_argument('--v_quant_per', type=str, choices=['channel', 'token'], 
                        help='Value를 채널 단위 또는 토큰 단위로 양자화할지 선택')
    
    parser.add_argument('--comp_obj', type=str, nargs='+', default=['wbits', 'kvbits'], choices=['wbits', 'kvbits'], 
                        help='동시에 최적화할 복잡도 목표')
    parser.add_argument('--comp_obj_min', type=float, nargs='+', default=[2, 2], 
                        help='복잡도 목표의 최솟값 범위')
    parser.add_argument('--comp_obj_max', type=float, nargs='+', default=[4, 4], 
                        help='복잡도 목표의 최댓값 범위')
    
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='데이터셋 이름')
    parser.add_argument('--seed', type=int, default=0,
                        help='보정(calibration) 데이터셋 선택 등을 위한 시드')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='보정 데이터셋의 샘플 수')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='보정(학습) 데이터셋의 시퀀스 길이')
    parser.add_argument('--metric', type=str, default='ppl',
                        help='평가에 사용할 메트릭 (ppl/loss)')
    
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='모델 메타데이터를 읽어올 설정 파일')
    parser.add_argument('--ga_pop_size', type=int, default=40,
                        help='NSGA 단계의 인구 크기')
    parser.add_argument('--subset_pop_size', type=int, default=100,
                        help='부분집합 선택 단계의 인구 크기')
    parser.add_argument('--debug', action='store_true', help='각 반복 결과 시각화 여부')
    parser.add_argument('--result_file', type=str, default='results.txt',
                        help='결과 파일 이름')
    parser.add_argument('--ga_algorithm', type=str, default='nsga2',
                        help='사용할 유전 알고리즘')
    parser.add_argument('--method', type=str, nargs='+', default=[],
                        help='양자화 방법')
    parser.add_argument('--max_value', type=float, default=50,
                        help='PPL 등 메트릭의 상한값')
    parser.add_argument('--crossover_prob', type=float, default=0.9,
                        help='교배(Crossover) 확률')
    parser.add_argument('--mut_prob', type=float, default=0.1,
                        help='변이(Mutation) 확률')
    parser.add_argument('--loss_func', type=str, default='cross_entropy',
                        help='사용할 손실 함수')
    parser.add_argument('--base_outlier_bits', type=int, nargs='+', default=[], 
                        help='Outlier가 아닌 채널에 적용할 기본 비트 후보')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='Outlier 인덱스 정보가 담긴 파일 경로')
    parser.add_argument('--n_outlier', type=int, default=0, 
                        help='Outlier 채널의 개수')
    parser.add_argument('--only_outlier_bits', action='store_true', help='Outlier 비트만 탐색할지 여부')
    parser.add_argument('--sensitivity_result_path', type=str, default='',
                        help='민감도 분석 결과 경로')
    parser.add_argument('--save_iter', type=int, default=1, 
                        help='몇 번의 반복마다 중간 결과를 저장할지 설정')
        
    
    cfgs = parser.parse_args()
    main(cfgs)

