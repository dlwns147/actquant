# 2차 탐색 top-1의 LongBench-E / RULER 개별 regret 분석

작성일: 2026-08-11

## 1. 질문과 평가 정의

이 분석에서는 LongBench-E와 RULER를 합친 balanced/minimax utility를 사용하지
않는다. 두 endpoint를 각각 독립적으로 최적화한다.

- LongBench-E regret: 같은 test memory window의 최고 LongBench-E 점수에서
  선택한 구조의 점수를 뺀 값. 단위는 LongBench-E point이다.
- RULER regret: 같은 window의 최고 RULER 평균에서 선택한 구조의 평균을 뺀
  값. 표에서는 읽기 쉽게 percentage point(pp)로 보고한다.
- baseline: memory window 안에서 실측 `wt2_jsd_pp128_s32`가 가장 낮은 구조.
- unrestricted correction: loss와 W/KV allocation feature로 endpoint를 예측해
  제한 없이 top-1을 고른다.
- guarded correction: `loss <= best + 0.001`인 near-tie만 허용하고, ridge,
  kNN, ExtraTrees, RandomForest, histogram GBDT 중 4/5가 같은 구조를 고를
  때만 baseline을 변경한다. 합의하지 못하면 loss top-1로 복귀한다.

사용 데이터는 benchmark가 모두 측정된 200개 구조이다. 100/100과 150/50
train/test split 모두 memory 10-quantile 층화 후 50개 seed로 반복했다. 모델은
train row만 보며, regret oracle과 선택은 test row만 사용한다. Test 후보 pool은
고정 ±5% memory window이고 후보가 5개 이상인 경우만 집계했다.

## 2. Loss와 endpoint의 상관관계

낮은 loss가 높은 benchmark 점수와 대응하므로 상관계수는 음수이다.

| 범위 | n | LB-E Pearson / Spearman | RULER Pearson / Spearman |
|---|---:|---:|---:|
| 전체 | 200 | -0.988 / -0.995 | -0.906 / -0.957 |
| target ±10% | 75 | -0.966 / -0.975 | -0.687 / -0.741 |
| target ±5% | 40 | -0.925 / -0.940 | -0.465 / -0.591 |
| target ±3% | 20 | -0.761 / -0.737 | -0.015 / -0.202 |
| target ±2% | 12 | -0.515 / -0.531 | -0.099 / -0.252 |

전역 상관은 두 endpoint 모두 높지만, top-1을 실제로 고르는 좁은 memory
구간에서는 특히 RULER 상관이 무너진다. 따라서 전체 200개 상관계수만으로
"loss top-1이 RULER top-1"이라고 결론 내릴 수 없다. 반대로 ±2~3% 결과는
n=12~20에 불과하므로 "상관이 없다"는 강한 결론에도 부족하다.

### Task별 이질성

| Endpoint / 범위 | task별 Pearson 중앙값 | task별 범위 |
|---|---:|---:|
| LongBench-E 전체 | -0.955 | [-0.975, -0.830] |
| LongBench-E target ±5% | -0.609 | [-0.891, +0.046] |
| RULER 전체 | -0.871 | [-0.970, -0.643] |
| RULER target ±5% | -0.445 | [-0.853, +0.086] |

LongBench-E target 구간에서 `qasper`, `trec`, `multi_news`, `samsum` 등은
loss가 task 순위를 거의 설명하지 못한다. RULER도 task별 차이가 크다.
예를 들어 target ±5%의 `niah_single_1`은 평균 0.968, 표준편차 0.039로
포화되어 Pearson이 +0.086이다. 반면 `ruler_vt`는 -0.853이다. RULER 평균
하나가 모든 RULER 능력을 대표한다고 해석해서는 안 된다.

## 3. 구조 보정이 실제로 추가 정보를 주는가

Loss-only ridge와 loss+11개 aggregate W/KV feature ridge의 독립 test R²를
비교했다.

| Train / test | LB-E loss-only → allocation | RULER loss-only → allocation |
|---|---:|---:|
| 100 / 100 | 0.9906 → 0.9906 | 0.8984 → 0.9595 |
| 150 / 50 | 0.9910 → 0.9913 | 0.8971 → 0.9603 |

LongBench-E에서는 allocation feature의 추가 설명력이 사실상 0이다. 반면
RULER에서는 R²가 약 0.061~0.063 증가한다. 즉 구조 보정의 통계적 근거는
LongBench-E보다 RULER에서 훨씬 강하다.

Raw per-layer profile 모델은 사용하지 않았다. Random split에서는 좋아졌지만
structural-cluster holdout에서 regret가 크게 악화되어, 200개 label로는 고차원
구조 효과를 안정적으로 식별하지 못했다.

## 4. Endpoint별 독립 top-1 regret

### 4.1 LongBench-E만 최적화

| Train / test | 선택법 | LB-E regret | baseline 대비 변화와 95% CI |
|---|---|---:|---:|
| 100 / 100 | loss baseline | 0.4440 | -- |
| 100 / 100 | unrestricted | 0.3674 | -0.0767 `[-0.1121,-0.0412]` |
| 100 / 100 | guarded 4/5 | **0.4259** | -0.0181 `[-0.0280,-0.0083]` |
| 150 / 50 | loss baseline | 0.3231 | -- |
| 150 / 50 | unrestricted | 0.2877 | -0.0354 `[-0.0744,+0.0036]` |
| 150 / 50 | guarded 4/5 | **0.3023** | -0.0208 `[-0.0351,-0.0066]` |

Guarded correction은 두 split 모두 작지만 유의한 개선을 보였다. 그러나
loss-only R²가 이미 약 0.991이고, 150/50 unrestricted CI는 0을 포함한다.
따라서 LongBench-E에 대해 강한 구조 보정이나 큰 loss 희생을 정당화할 근거는
약하다. Near-tie 안의 보수적 tie-break 정도가 데이터가 지지하는 범위이다.

### 4.2 RULER만 최적화

| Train / test | 선택법 | RULER regret (pp) | baseline 대비 변화와 95% CI |
|---|---|---:|---:|
| 100 / 100 | loss baseline | 5.937 | -- |
| 100 / 100 | unrestricted | 1.728 | -4.209 `[-4.673,-3.746]` |
| 100 / 100 | guarded 4/5 | **5.575** | -0.363 `[-0.500,-0.225]` |
| 150 / 50 | loss baseline | 5.726 | -- |
| 150 / 50 | unrestricted | 1.243 | -4.483 `[-4.982,-3.984]` |
| 150 / 50 | guarded 4/5 | **5.463** | -0.263 `[-0.410,-0.116]` |

RULER는 allocation correction의 이득이 훨씬 크다. 하지만 unrestricted
RULER selector는 다른 endpoint에 큰 비용을 줄 수 있다. 100/100에서 선택한
구조의 LB-E regret는 0.444에서 0.912로, 150/50에서는 0.323에서 0.806으로
악화되었다. RULER만 정말 유일한 목적일 때에만 이 선택을 정당화할 수 있다.

Guarded RULER selector는 변경 빈도가 100/100에서 약 8.1%, 150/50에서 약
3.2%에 불과했다. 이 경우 LB-E regret도 각각 0.420, 0.298로 baseline보다
나빠지지 않았다. 현재 증거에서는 unrestricted보다 guarded 방식이 더
방어 가능하다.

### 4.3 Target ±5% 구간

| Train / test | Endpoint | loss baseline | guarded selector |
|---|---|---:|---:|
| 100 / 100 | LB-E regret | 0.1750 | 0.1606 |
| 100 / 100 | RULER regret (pp) | 0.712 | 0.611 |
| 150 / 50 | LB-E regret | 0.1125 | 0.1090 |
| 150 / 50 | RULER regret (pp) | 1.000 | 0.975 |

150/50 target window에는 평균 9.68개 test 후보만 있어 개선 추정치가 작고
불안정하다. 실제 ±0.5% 구간은 label이 전체 200개 중 네 개뿐이므로 train/test로
나눈 endpoint regret 주장을 할 수 없다.

## 5. 현재 2차 탐색 archive에 적용

Residual-128을 포함한 대칭 ±0.5% physical-memory band에는 108개 구조가 있다.
실측 loss top-1은 archive index 4041이고 JSD는 0.0549927이다. Index 4047은
JSD 0.0557251로 두 번째이다.

Unrestricted model의 추천은 endpoint별로 다음과 같이 분산되었다.

- LongBench-E: 각 모델이 loss rank 6, 45, 35, 1, 44를 선택했다.
- RULER: 각 모델이 loss rank 66, 45, 35, 41, 9를 선택했다.

두 endpoint 모두 동일 구조에 동의한 모델이 최대 1/5이다. 평균 holdout
성능이 좋더라도 현재 archive에 대한 unrestricted 추천은 심한 model
disagreement 때문에 신뢰할 수 없다.

`best + 0.001` guard를 적용하면 후보는 4041과 4047뿐이다. LongBench-E와
RULER 각각에서 투표 결과가 동일하다.

- 4041: kNN, ExtraTrees, RandomForest의 3표
- 4047: ridge, histogram GBDT의 2표

4/5 합의에 도달하지 못하므로 두 endpoint selector 모두 abstain하고 4041을
반환한다. 이것은 4041이 실제 LB-E/RULER oracle임을 입증한 결과가 아니라,
현재 200개 label로 다른 구조를 선택할 충분한 근거가 없다는 뜻이다.

대칭 band는 hard cap이 아니다. 4041의 physical memory는 5,341,501,184
bytes로 중심 5,315,764,224 bytes보다 0.484% 높다. 중심이 배포 상한이면
4041은 infeasible하며, `[center*(1-0.005), center]`의 loss winner인 index
4035(JSD 0.0602417)를 기준으로 endpoint 분석을 다시 해야 한다.

## 6. 객관적 결론

1. **LongBench-E:** 현재 joint loss가 이미 매우 강한 proxy다. 구조 보정은
   near-tie tie-break 수준에서만 정당화된다.
2. **RULER:** loss만으로는 iso-memory ranking이 부족하며 W/KV allocation이
   유의한 추가 정보를 준다. 다만 unrestricted correction은 현재 archive에서
   외삽과 모델 불일치가 심하다.
3. **현재 top-1:** LongBench-E 전용과 RULER 전용 guarded selector 모두
   4041에서 abstain/fallback한다. 별도 benchmark를 찍지 않는 조건에서는
   다른 구조가 낫다고 주장할 수 없다.
4. **두 endpoint를 동시에 만족하는 단일 top-1:** balanced/minimax 같은
   preference를 쓰지 않으면 원칙적으로 정의되지 않는다. 향후 두 selector가
   서로 다른 구조를 고르면 어느 것이 "정답"인지는 deployment objective가
   결정해야 한다.
5. **가장 큰 추가 불확실성:** target band의 best loss가 iteration 15 마지막에
   0.05859에서 0.05499로 갱신되었다. Post-search correction 이전에 search
   convergence가 충분한지도 확인해야 한다.

## 7. 주장 가능한 범위

현재 결과로 주장 가능한 것은 "독립 holdout에서 guarded endpoint correction이
평균 regret를 소폭 줄였고, RULER에서는 allocation 정보가 특히 유용했다"이다.
"4041이 실제 LongBench-E/RULER 최적" 또는 "unrestricted model의 추천이 실제
최적"이라고 주장할 수는 없다. 200개 labelled set과 현재 second-stage archive는
exact full-architecture overlap이 0개이고, target ±0.5% labelled row도 4개뿐이다.
