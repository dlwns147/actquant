# 1단계는 순서를 제공하고, 2단계는 값을 제공한다

## 가중치와 KV 캐시 압축을 위한 효율적인 결합 탐색

**익명 저자**

> **가칭.** 본 방법은 저장소 이름에 맞추어 임시로 **ActQuant**라
> 부른다. 최종 논문 제목과 시스템 이름이 결정되면 변경할 예정이다.

## 초록

제한된 메모리에서 거대 언어 모델을 배포하려면 모델 가중치와
키-값(key--value, KV) 캐시 사이에 정밀도를 함께 할당해야 한다. 이때
발생하는 레이어 단위 탐색 공간은 직접 블랙박스 최적화를 수행하기에
지나치게 크며, 활성값 인지(activation-aware) 가중치 양자화기는 새로운
가중치 할당을 평가할 때마다 비싼 빌드 과정을 요구한다. 본 논문은
*어디를 탐색할 것인가*와 *무엇을 선택할 것인가*를 분리하는 2단계 탐색
프레임워크 **ActQuant**를 제안한다. 1단계에서는 저비용 양자화 프록시를
이용해 가중치 축과 유효 KV 축을 독립적으로 탐색한다. 2단계에서는 두
축의 $\epsilon$-Pareto 집합의 곱 공간을 구조 인지 돌연변이와 함께
탐색하고, 실제 배포 양자화기로 목적값을 측정한다. 본 논문의
front-product coverage 정리는 상대 축에 의해 발생하는 순서 위반이 작을
경우 이 제한된 공간이 결합 공간의 Pareto 집합을 근사적으로 포함함을
보인다. 이 결과는 손실의 가산성을 요구하지 않는다. 또한 양자화
프록시를 안전하게 사용할 수 있는 범위를 규명한다. 실제 배포 Pareto
점이 프록시 목적에서 지배되면 그 점은 프록시 frontier에서 영구
제외되며, 프록시 평가 예산을 무한히 늘려도 이 목적 불일치는 사라지지
않는다. 또한 실제 KV 캐시의 누적 경로를
따라 prefix를 prefill하고 answer suffix를 여러 chunk로 진행하면서 출력
분포 발산을 측정하는 **Strided JSD**를 제안한다. 200개 결합 구성에서
Strided JSD는 기존 단일 forward JSD보다 LongBench-E와 RULER 성능에 대한
순위 상관관계가 높았다. 비싼 2단계 탐색의 표본 효율을 높이기 위해
프록시 지도 partial least squares(PLS) 임베딩으로 입력 차원을 축소하고,
한 번의 활성값 인지 가중치 빌드에서 여러 KV 구성을 평가한다.
Llama-3.1-8B에 대한 예비 분석에서 $20\times20$ 대응 격자의
축별 순서는 대부분 보존되었지만, 동일 예산의 세밀한 구성 사이에서는
프록시 순위가 크게 무너졌다. 이 결과는 프록시로 탐색 공간을 구성하고
표현하되, 결합 공간 내부의 최종 판단에는 실제 배포 양자화기를 사용해야
한다는 역할 분담을 뒷받침한다.

## 1. 서론

거대 언어 모델(large language model, LLM) 추론은 서로 다른 두 종류의
메모리에 의해 제약된다. 모델 가중치는 크지만 고정된 메모리를 차지하는
반면, 키-값(KV) 캐시는 배치 크기와 문맥 길이에 따라 증가한다. 가중치
전용 양자화는 전자의 크기를 줄이고 메모리 병목인 선형 계층을 가속할
수 있다 [1, 2]. KV 캐시 양자화와 채널 가지치기는 후자의 크기를 줄이며,
특히 긴 문맥을 처리하는 서빙 환경에서 중요하다 [5, 6]. 그러나 두
요소를 독립적으로 최적화하면 중요한 배포 문제가 남는다. 하나의 종단간
메모리 예산이 주어졌을 때 가중치 모듈, KV 레이어, K/V 양자화 파라미터,
그리고 보존할 KV 채널 사이에 정밀도를 어떻게 배분해야 하는가?

이는 조합적 다목적 최적화 문제다. 현재 Llama-3.1-8B 탐색 공간에서는
224개 가중치 행렬이 각각 2, 3, 4비트 형식 중 하나를 선택한다. 각
레이어는 추가로 K와 V의 비트 수 및 그룹 크기 조합과 K/V 보존 차원을
선택한다. 전체 탐색 공간의 크기는 대략

$$
3^{224}7^{64}5^{64}\approx 5\times10^{205}
$$

이다. 이 중 극히 일부조차 직접 평가하기 어렵다. 더 중요한 문제는
구성별 평가 비용이 균일하지 않다는 점이다. HQQ와 같은 활성값
비의존(activation-independent) 양자화기는 미리 계산한 가중치 뱅크에서
혼합 정밀도 모델을 조립할 수 있다 [3]. 반면 AWQ와 같은 활성값 인지
방법은 보정 활성값에 의존하므로 레이어별 비트 할당이 바뀔 때 새로운
양자화 가중치를 빌드해야 한다 [2]. 현재 구현에서 Llama-3.1-8B의 AWQ
빌드와 평가는 GPU 한 장에서 약 8분이 소요된다. 따라서 초기 설계의
크기와 그에 따른 surrogate 품질은 탐색 성능을 결정하는 핵심 요인이다.

AMQ는 탐색 공간 가지치기, HQQ 양자화 프록시, 품질 예측기, 반복적
다목적 탐색을 결합하여 레이어 단위 혼합 정밀도 *가중치 전용* 탐색을
실용화하였다 [4]. AMQ의 프록시 정리는 강한 조건을 사용한다. 프록시와
배포 양자화기가 모든 구성에 대해 동일한 전체 순서를 만들면 두 Pareto
frontier가 일치한다. 가중치와 KV를 함께 탐색하면 이 조건의 적용 범위가
분명해진다. KV 캐시를 고정했을 때 프록시가 가중치 구성의 거친 순서를
보존하더라도, KV 구성이 달라진 뒤 어느 레이어에 가중치 비트를
배치해야 하는지에 대해서는 배포 방법과 다른 판단을 내릴 수 있다.
따라서 “프록시와 배포 방법의 상관관계가 높다”는 사실만으로는 충분하지
않다. 탐색 공간을 제한하는 데 유용한 **예산 간 순서**와 최종 구조를
결정하는 **동일 예산 내 순서**를 구분해야 한다.

본 논문의 핵심 관찰은 결합 문제가 다음 두 질문으로 자연스럽게
분리된다는 것이다.

1. **어디를 탐색해야 하는가?** 상대 축이 변해도 축별 순서가 안정적이면
   축별 Pareto 집합의 Cartesian product가 결합 공간에서 유용한 영역을
   포함한다.
2. **어떤 구성을 선택해야 하는가?** 축별 관측만으로는 곱 공간 내부의
   실제 목적값, 가중치--KV 교환율, 세밀한 순서를 알 수 없다. 이러한
   정보는 배포 양자화기로 측정한 결합 표본에서 학습해야 한다.

이러한 순서--값 분리가 ActQuant의 출발점이다. 1단계에서는 상대 축을
고정밀 기준 구성으로 고정하고 (i) 레이어별 가중치 정밀도와 (ii) K/V
비트 수, 그룹 크기, 채널 가지치기를 결합한 유효 KV 정밀도를 독립적으로
탐색한다. 이 단계는 HQQ 가중치 뱅크, 출력 분포 발산, surrogate 기반
반복 다목적 탐색을 사용한다. 2단계에서는 1단계가 반환한
$\epsilon$-Pareto 블록으로 후보를 구성하고, 논쟁적인 레이어 및 모듈
위치에 제한된 돌연변이를 적용하며, AWQ 측정값으로 결합 공간을 탐색한다.
따라서 비싼 활성값 인지 빌드는 훨씬 작고 구조화된 영역에만 사용된다.

이 탐색 공간 제한은 간단하지만 핵심적인 정리로 뒷받침된다. 각 축에
대해 1단계가 더 좋다고 판단한 구성으로 교체했을 때 배포 손실이 증가할
수 있는 최대량을 직접 정의한다. 이 순서 위반 마진이 각각 $V_W$와
$V_{KV}$라면, 모든 결합 구성에 대해 비용은 더 작거나 같고 손실은 최대
$V_W+V_{KV}$만 증가하는 구성이 두 축 frontier의 곱 안에 존재한다. 이
정리는 비용 분리성과 충분한 축별 frontier를 요구하지만 손실의 가산성을
가정하지 않는다. 제한된 돌연변이는 후보 집합을 확장하므로 이 존재성
보장을 훼손하지 않는다.

프록시를 사용할 수 있는 범위도 별도의 불일치 정리로 명확히 한다. 실제
배포 Pareto 점을 비용과 프록시 손실에서 지배하는 구성이 하나라도
존재하면, 해당 점은 프록시 frontier에서 제거된다. 따라서 프록시
frontier로 수렴하는 탐색은 평가 횟수와 무관하게 실제 frontier 전체로
수렴할 수 없다. 또한 주어진 예산에서 모든 프록시 최적점이 실제
최적점보다 나쁘다면 양의 배포 regret 하한이 남는다. 예비 측정은 두
개별 축에서는 HQQ와 AWQ의 Spearman 상관계수가 1.00인 반면, 동일 예산의
세밀한 결합 후보에서는 약 0.40으로 감소함을 보인다. 따라서 프록시
정보는 1단계 후보 공급과 특징 학습에는 적합하지만 2단계의 최종 label로
사용하기에는 충분하지 않다.

AWQ label을 효율적으로 얻기 위해 두 가지 방법을 사용한다. 첫째,
가중치와 KV의 one-hot genome에 프록시 지도 PLS 투영을 각각 학습한다.
현재의 고차원 이산 표현은 16개 잠재 좌표와 정확한 비용 좌표 2개로
축소되며, AWQ surrogate는 이 18차원 공간에서 학습된다. 둘째, AWQ
빌드는 가중치 할당에만 의존한다. 선택된 가중치 anchor마다 가중치를 한
번만 빌드한 후 KV 구성만 교체하여 다양한 companion을 평가함으로써 한
번의 가중치 빌드에서 여러 결합 label을 얻는다.

본 논문의 기여는 다음과 같다.

- 가중치--KV 결합 압축을 2단계 3목적 탐색으로 정식화하고, 축별 탐색의
  순서적 역할과 결합 측정의 값 추정 역할을 분리한다.
- 단일 full-sequence forward의 JSD 대신 실제 KV 캐시 갱신 경로와
  answer phase를 모사하는 Strided JSD를 제안하고, 장문 과제와의 전역 및
  동일 예산 내 상관관계가 더 높음을 실증한다.
- 정확한 Pareto 집합과 $\epsilon$-Pareto 집합 모두에 적용할 수 있는
  검증 가능한 front-product coverage 정리를 제시한다. 또한 더 강하지만
  해석하기 쉬운 충분조건과 현재 실증 범위를 명시한다.
- 실제 배포 Pareto 점에 대한 프록시 지배가 영구적인 frontier 누락을
  만들고, 예산별 프록시 최적화에는 배포 regret 하한이 남는다는 프록시
  비일관성 정리를 제시한다. 이를 통해 HQQ는 1단계를 안내할 수 있지만
  2단계에는 AWQ 정보가 필요한 이유를 설명한다.
- 비싼 AWQ 빌드당 유용한 label 수를 늘리기 위해 프록시 지도 PLS 차원
  축소와 가중치 빌드 공유형 multi-KV 평가를 제안한다.
- 전역 상관관계만 보고하는 대신, 이론의 가정을 직접 측정하는 대응 격자
  및 동일 예산 분석 절차를 제시한다.

현재 초안의 주장은 실제로 분석한 구성 범위로 제한한다. 현재
$20\times20$ 분석은 발견된 frontier에서 표본화한 블록을 사용한다.
따라서 off-front 돌연변이, 1단계 frontier 오차, 다른 모델로의 일반화,
최적 KV companion 수는 숨겨진 가정이 아니라 향후 검증할 실험 문제로
남겨 둔다.

## 2. 관련 연구

### 2.1 가중치 전용 양자화

GPTQ는 정확한 사후 학습 가중치 양자화를 위해 근사 2차 정보를 사용한다
[1]. AWQ는 활성값 통계로 중요한 가중치 채널을 찾고 채널별 scaling을
탐색하여 낮은 비트에서도 높은 정확도와 하드웨어 친화적인 추론 kernel을
제공한다 [2]. 혼합 정밀도 탐색의 관점에서 두 방법은 모두 보정 과정에
의존한다. 즉, 레이어별 할당을 바꾸면 비싼 재구성 또는 보정 과정이 다시
필요할 수 있다. HQQ는 half-quadratic splitting을 이용해 활성값 보정 없이
가중치를 양자화한다 [3]. 이 특성 덕분에 비트별 가중치 뱅크를 미리
계산하고 임의의 레이어별 구성을 저비용으로 조립할 수 있다. 본 연구는
이러한 이유로 HQQ를 최종 배포 목적이 아니라 탐색 프록시로 사용한다.

혼합 정밀도 가중치 기법은 민감한 레이어, 채널 또는 이상치에 더 높은
정밀도를 할당한다. 이를 통해 소수점 평균 비트에서 높은 품질을 얻을 수
있지만, 지나치게 세밀한 형식은 불규칙한 메모리 접근이나 전용 kernel을
요구할 수 있다. 본 연구는 AMQ가 강조한 하드웨어 호환 granularity에
따라 선형 계층별로 하나의 정밀도를 사용한다 [4]. 본 연구의 기여는
새로운 스칼라 가중치 양자화기가 아니라, 배포 가중치 양자화와 문맥에
따라 증가하는 KV 캐시에 정밀도를 함께 할당하는 방법이다.

### 2.2 자동 혼합 정밀도 탐색

AMQ는 가중치 축에서 본 연구와 가장 가까운 선행연구다 [4]. AMQ는
민감도에 따라 레이어별 탐색 공간을 가지치기하고, HQQ를 AWQ/GPTQ의
프록시로 사용하며, RBF 모델로 Jensen--Shannon divergence를 예측하고,
NSGA-II archive를 반복적으로 갱신한다. 또한 프록시와 배포 점수 사이의
전역 순서 동치가 동일한 Pareto frontier를 만든다는 것을 증명한다. 본
연구는 이러한 축별 탐색 방법을 활용하지만, 결합 공간 내부에서 전역
순서 동치가 성립하지 않는 문제를 다룬다. 따라서 *coverage*에 필요한
조건을 작은 조건부 순서 위반으로 완화하고, 프록시가 구분할 수 없는
결정에만 배포 방법의 측정값을 사용한다.

Surrogate-assisted neural architecture search는 많은 고비용 평가를 학습된
성능 예측기로 대체한다 [11, 12]. NSGA-II와 NSGA-III 같은 다목적 진화
알고리즘은 하나의 고정된 예산을 최적화하는 대신 정확도--효율 간의 여러
절충점을 유지한다 [9, 10]. 본 연구는 수백 개 이산 레이어 변수를 그대로
사용해 예측기를 학습하는 대신, 훨씬 큰 프록시 archive에서 지도 차원
축소를 학습한다. PLS는 입력 분산만을 보존하는 것이 아니라 target과의
공분산에 따라 잠재 방향을 선택하므로 이 목적에 적합하다 [13]. 최종
surrogate의 regression head는 여전히 배포 label로 학습되므로 프록시
값을 실제 배포 정답으로 취급하지 않는다.

### 2.3 KV 캐시 압축

KIVI는 key와 value의 분포가 다름을 관찰하고, key에는 채널별 양자화를,
value에는 토큰별 양자화를 적용하며, 일부 최근 토큰은 고정밀 residual
window로 유지한다 [5]. KVQuant 역시 이상치, 민감도 인지 datatype,
저비트 KV 캐시 표현을 연구한다 [7]. ThinK는 query-driven pruning으로
채널 차원을 줄이며 KIVI와 결합할 수 있다 [6]. 이 방법들은 본 연구의
유효 KV 축을 구성하는 양자화 및 가지치기 연산을 제공한다. 그러나 공유
메모리 예산을 레이어별 가중치 정밀도와 어떻게 교환할지는 결정하지
않는다.

KVTuner는 민감도 기반 가지치기와 clustering 이후 하드웨어 친화적인
레이어별 K/V 정밀도 쌍을 탐색한다 [8]. 이는 본 연구의 1단계 유효 KV
탐색과 밀접하다. 본 연구는 K/V 비트 수와 그룹 크기에 보존 채널 차원을
추가하고, 이 축을 레이어별 가중치 할당과 결합한다. 그 결과 두 가지
새로운 문제가 발생한다. 두 축의 Cartesian product가 매우 크고, 그
공간의 품질 순서는 가중치 양자화기가 활성값 비의존인지 활성값 인지인지에
따라 달라진다. 본 연구의 front-product 정리와 프록시 적용 범위 분석은
이러한 축 간 문제를 대상으로 한다.

### 2.4 본 연구의 위치

기존 연구는 주로 하나의 양자화 축 *내부*에서 민감도 또는 Pareto
가지치기를 사용한다. 본 연구는 이와 다른 중간 영역을 다룬다. 가중치와
KV의 자원 비용은 분리되지만 품질 목적함수의 분리성은 가정하지 않는다.
Product-front 제한은 가중치 손실과 KV 손실이 더해진다는 가정이 아니라,
측정된 조건부 순서 안정성에 의해 정당화된다. 이는 실제 joint loss가
배포 가능한 영역에서는 거의 가산적이지만, 낮은 가중치와 낮은 KV가
결합된 공격적인 영역에서는 강한 포화 현상을 보인다는 점에서 중요하다.

## 3. 방법

### 3.1 문제 정의

$\mathcal{X}=\mathcal{X}_W\times\mathcal{X}_{KV}$를 유한한 구성 공간이라
하자. 가중치 구성 $w\in\mathcal{X}_W$는 모든 선형 모듈에 비트 수를
할당한다. 캐시 구성 $k\in\mathcal{X}_{KV}$는 각 레이어의 K/V 비트 수,
그룹 크기, K/V 채널 가지치기 차원을 할당한다. 결합 구조는 $a=(w,k)$로
표기한다.

양자화 방법 $M$에서 품질 손실은 full-precision teacher와 압축 모델의
출력 분포 사이 Jensen--Shannon divergence로 정의한다.

$$
y_M(a)=\frac{1}{|\mathcal{D}|}
\sum_{(x,t)\in\mathcal{D}}
\operatorname{JSD}\!\left(
p_{\mathrm{FP}}(\cdot\mid x,t),
p_{M,a}(\cdot\mid x,t)
\right).
\tag{1}
$$

작을수록 좋은 값이다. 모든 비교에서 보정 data와 teacher를 고정한다.
다만 동일한 pointwise JSD를 사용하더라도 compressed model의 forward
경로에 따라 KV 압축 오차가 드러나는 정도가 달라질 수 있다. 3.2절에서는
기존 단일 forward JSD를 실제 장문 생성 경로에 더 가깝게 바꾼다.

두 비용 좌표는 다음과 같다.

$$
c(a)=\big(c_W(w),c_{KV}(k;T)\big).
\tag{2}
$$

$c_W$는 파라미터 수로 가중한 평균 가중치 정밀도다. $c_{KV}$는 보고
목적에 따라 유효 KV 정밀도 또는 문맥 길이 $T$에서의 정확한 캐시
메모리다. 정확한 메모리 계산에는 양자화 metadata, 보존 채널 차원,
고정밀 residual window, 고정 attention-sink window가 포함된다. 구성상
$c_W$는 $w$에만, $c_{KV}$는 $k$에만 의존한다. 이 정확한 비용 분리성은
이론에서 중요하지만, $y_M$에 대해서는 같은 분리성을 가정하지 않는다.

배포 목적은 다음 3목적 최소화 문제다.

$$
\min_{(w,k)\in\mathcal{X}}
\left(y_{\mathrm{AWQ}}(w,k),c_W(w),c_{KV}(k;T)\right).
\tag{3}
$$

동일하게, 배포 시 두 비용 좌표의 상한을 만족하는 측정 구성 중
$y_{\mathrm{AWQ}}$가 최소인 구조를 선택할 수 있다.

### 3.2 Strided JSD: 장문 배포 경로를 모사하는 탐색 지표

#### 기존 JSD의 불일치

기존 JSD는 teacher forcing된 전체 길이 $L$의 sequence를 한 번의
forward로 처리하고, 모든 유효 token의 teacher--compressed 분포 발산을
평균한다. 이는 저비용이고 재현성이 높지만 두 가지 불일치가 있다. 첫째,
실제 생성에서는 prefix를 먼저 prefill한 뒤 이전 step에서 저장한 압축 KV
cache를 계속 읽고 갱신하지만, 단일 forward는 이 cache의 누적 경로를
그대로 실행하지 않는다. 둘째, 전체 token 평균은 긴 prefix의 쉬운
국소 예측에 의해 지배될 수 있어, 긴 문맥을 읽은 뒤 답을 생성할 때
발생하는 오차를 희석한다.

#### 정의

이를 해결하기 위해 **Strided JSD(S-JSD)**를 사용한다. 길이 $L_x$인
sequence $x$에 대해 마지막 $A$개 token을 answer 구간
$\mathcal{T}_x=\{L_x-A,\ldots,L_x-1\}$로 두고, 그 앞의 prefix를 한 번에
prefill한다. 이후 정답 token을 teacher forcing하되 크기 $s$인 chunk로
나누어 순서대로 입력한다. 각 chunk는 앞선 chunk가 만든 실제 압축 KV
cache를 `past_key_values`로 받아 갱신한다. Full-precision teacher 분포를
$q_t$라 하고 이 strided cache 경로에서 얻은 압축 모델 분포를
$\widetilde p^{a}_{t;A,s}$라 하면

$$
\operatorname{S\text{-}JSD}_{A,s}(a)
=\frac{1}{\sum_{x\in\mathcal{D}}|\mathcal{T}_x|}
\sum_{x\in\mathcal{D}}\sum_{t\in\mathcal{T}_x}
\operatorname{JSD}\!\left(q_t,\widetilde p^{a}_{t;A,s}\right)
\tag{S-JSD}
$$

로 정의한다. 이 지표는 정답 token을 입력하는 teacher-forced metric이므로
실제 생성 자체는 아니다. 그러나 압축 모델 쪽에서는 `use_cache=True`인
prefill--decode 경로를 실행하고, 오직 긴 prefix 이후의 answer token만
평가한다. $s$가 작을수록 token-by-token decode에 가까워지는 대신 answer
구간의 forward 횟수 $\lceil A/s\rceil$가 증가한다.

현재 탐색 기본값은 WikiText-2의 길이 2048 표본 128개에서
$A=512,s=128$을 사용하는 것이다. 즉 앞의 1536개 token을 prefill한 후
마지막 512개를 128-token chunk 네 개로 진행한다. 이 설정은 실제 cache
경로를 통과하면서도 탐색 중 반복 측정이 가능한 비용 절충점이다. 더
정확한 사후 검증에는 $s=32$를 사용할 수 있지만, 이 경우 answer 구간의
compressed-model forward 수가 4개에서 16개로 증가한다.

#### 장문 과제 상관관계

S-JSD가 단순히 구현상 더 그럴듯한 지표인지, 실제 장문 성능의 순서를 더
잘 예측하는지 확인하였다. 지정된 correlation archive의
Llama-3.1-8B-Instruct 결합 구성 200개를 사용하였다. 각 구성은 AWQ
가중치와 KIVI--ThinK KV 압축을 사용하며, 장문 성능은 LongBench-E의
39개 task--length cell과 16K RULER의 13개 task에서 측정하였다 [14, 15].
JSD는 낮을수록, task score는 높을수록 좋으므로 상관계수는 음수이며
아래 표는 비교가 쉽도록 절댓값 $|\rho|$를 보고한다. “예산 내” 값은
가중치 비트와 유효 KV 비트를 각각 사분위로 나눈 $4\times4$ cell에서
Spearman 상관계수를 계산한 뒤 표본 수로 가중 평균한 값이다.

| 보정 지표 | LongBench-E 평균 | RULER 평균 | LongBench-E 예산 내 | RULER 예산 내 |
|---|---:|---:|---:|---:|
| 기존 full-sequence JSD | 0.656 | 0.792 | 0.481 | 0.578 |
| 마지막 128 token JSD, 단일 forward | 0.676 | 0.808 | 0.496 | 0.579 |
| 전체 sequence strided JSD, $s=512$ | 0.820 | 0.919 | 0.731 | 0.769 |
| S-JSD, $A=512,s=128$ | **0.903** | **0.954** | **0.807** | **0.813** |
| S-JSD, $A=512,s=32$ | **0.995** | **0.969** | **0.907** | **0.854** |

마지막 token만 선택하고 여전히 단일 forward를 사용하면 개선 폭이 작지만,
전체 sequence를 cache를 유지하며 chunk로 진행하는 것만으로도 상관이 크게
높아진다. 여기에 prefix--answer 분리까지 적용한 S-JSD가 다시 개선되므로,
관찰된 이득을 answer masking만의 효과로 설명하기 어렵다. 다만 이 표는
완전한 요인 실험은 아니며, $A$, $s$, calibration dataset을 각각 독립적으로
통제하는 ablation이 필요하다.

현재 기본 설정인 $A=512,s=128$은 기존 JSD보다 39개 LongBench-E cell과
13개 RULER task 전부에서 더 큰 $|\rho|$를 보였다. Aggregate score에
대한 Pearson 상관계수도 LongBench-E에서 $|r|=0.590$에서 0.889로,
RULER에서 0.783에서 0.934로 증가하였다. 또한 $4\times4$ 예산 cell
안에서도 차이가 유지되므로, 전역 결과를 단순히 저비트 구조와 고비트
구조의 차이만으로 설명하기는 어렵다. 더 작은 stride 32는 평균 상관이
더 높아 고정밀 re-ranking metric으로 유용하지만, 증가한 평가 비용 때문에
현재 search objective는 stride 128을 사용한다.

이 결과의 범위는 명확히 제한한다. 동일한 200개 구조에서 여러 metric을
비교했으므로 metric 간 비교는 대응되어 있지만, 구조 표본은 발견된 축별
frontier의 곱에서 얻었고 모델, seed, compression family가 각각 하나다.
또한 상관관계는 선택 regret이나 Pareto frontier 보존을 직접 보장하지
않는다. 최종 주장을 위해서는 다른 모델과 문맥 길이에서의 반복, 독립적인
configuration split, 동일 예산 top-$k$ regret, metric 측정 시간 대비
효율을 함께 보고해야 한다.

### 3.3 2단계 탐색 개요

식 (3)을 직접 최적화하는 것은 조합 공간의 크기와 AWQ 빌드 비용 모두의
제약을 받는다. ActQuant는 이를 다음 두 단계로 분해한다.

#### 1단계: 축별 순서 탐색

고정밀 기준 상대 구성 $r_W$와 $r_{KV}$를 선택한다. 저비용 프록시
$P=\mathrm{HQQ}$에 대해 다음 축별 점수를 정의한다.

$$
z_W(w)=y_P(w,r_{KV}),\qquad
z_{KV}(k)=y_P(r_W,k).
\tag{4}
$$

초기 설계, 반복적으로 재학습되는 surrogate, NSGA-II 후보 생성을 이용해
$(z_W,c_W)$와 $(z_{KV},c_{KV})$를 독립적으로 탐색한다. 각 탐색은 비용별
단일 점이 아니라 archive와 $\eta_i$-Pareto band를 반환한다. 거의 같은
성능의 할당이 하나의 레이어 패턴으로 축소되지 않도록 band 안의 구조적
다양성을 유지한다.

#### 2단계: 결합 공간의 실제 값 탐색

두 band에 포함된 가중치 블록과 KV 블록의 Cartesian product로 후보
공간을 만든다. 축 블록 단위 crossover는 1단계에서 발견한 일관된 구조를
보존한다. 돌연변이는 1단계 frontier 블록들이 서로 다른 값을 갖는 위치를
우선적으로 다시 표본화한다. 블록의 95% 이상이 같은 값을 갖는 위치는
고정할 수 있다. 이 product-plus-mutation 공간에서
$(y_{\mathrm{AWQ}},c_W,c_{KV})$를 NSGA-III로 최적화한다. Surrogate는
후보를 제안하지만, 최종 archive와 예산별 선택에는 실제로 측정한 AWQ
값만 사용한다.

각 2단계 반복은 다음 과정을 수행한다.

1. 현재 archive와 3.6절의 프록시 지도 임베딩으로 AWQ-loss surrogate를
   학습한다.
2. NSGA-III로 product, crossover, mutation 후보를 생성한다.
3. 현재 $(c_W,c_{KV})$ Pareto 기하를 잘 덮는 가중치 anchor를 선택하고,
   각 anchor에 서로 다른 $K-1$개의 KV companion을 연결한다.
4. 서로 다른 가중치 anchor마다 AWQ 가중치를 한 번 빌드하고, 연결된 모든
   KV companion을 평가한 뒤 성공한 측정값을 archive에 추가한다.

이 구조에서 1단계는 유용한 블록을 식별하는 **순서적 역할**을 하고,
2단계는 loss surface를 학습하여 결합 예산에서 구성을 결정하는 **값
추정 역할**을 한다.

### 3.4 축별 frontier의 곱으로 충분한 이유

먼저 탐색 공간 제한에 필요한 가장 약한 양을 정의한다. 이 정의는
양자화 방법을 구분한다. 1단계 순서는 HQQ에서 얻을 수 있지만 정리의
결론에 사용되는 결합 손실은 AWQ로 측정한다.

**정의 1 ($\eta$-front coverage).** 집합
$\widehat{\mathcal{P}}_i\subseteq\mathcal{X}_i$가 모든
$x_i\in\mathcal{X}_i$에 대해 다음을 만족하는
$p_i\in\widehat{\mathcal{P}}_i$를 포함하면, 이 집합을 축 $i$의
$\eta_i$-cover라 한다.

$$
c_i(p_i)\le c_i(x_i),\qquad
z_i(p_i)\le z_i(x_i)+\eta_i.
\tag{5}
$$

유한 공간의 정확한 Pareto 집합에서는 $\eta_i=0$이다. 실제 탐색으로
발견한 archive에서는 식 (5)가 실증적으로 검증해야 할 충실도 조건이다.

축 $i$의 구성을 교체할 때 발생하는 조건부 순서 위반 마진을 다음과
같이 정의한다.

$$
V_i(\eta_i)=
\sup\left\{
\left[y_D(u,x_{-i})-y_D(u',x_{-i})\right]_+:
\begin{array}{l}
c_i(u)\le c_i(u'),\\
z_i(u)\le z_i(u')+\eta_i,\\
x_{-i}\in\mathcal{X}_{-i}
\end{array}
\right\}.
\tag{6}
$$

여기서 $D=\mathrm{AWQ}$이고 $[q]_+=\max(q,0)$이다. 이 값은 1단계가 더
나쁘지 않다고 판단한 구성으로 교체했음에도 임의의 상대 축 구성에서
배포 품질이 나빠질 수 있는 최대량이다.

**정리 1 (Front-product coverage).** 비용 함수가 식 (2)와 같이
분리되고 각 $\widehat{\mathcal{P}}_i$가 $\eta_i$-cover라고 하자. 그러면
모든 $a\in\mathcal{X}$에 대해

$$
b\in\mathcal{C}
=\widehat{\mathcal{P}}_W\times\widehat{\mathcal{P}}_{KV}
$$

이면서 다음을 만족하는 $b$가 존재한다.

$$
c(b)\le c(a)\quad\text{성분별로},\qquad
y_D(b)\le y_D(a)+V_W(\eta_W)+V_{KV}(\eta_{KV}).
\tag{7}
$$

따라서 product 집합은 결합 Pareto 집합에 대해 비용 관점에서 약하게
지배하고, $\epsilon=V_W+V_{KV}$인 품질 근사 지배를 제공한다.

*증명.* 식 (5)의 cover 점을 $p_W,p_{KV}$라 하고
$b=(p_W,p_{KV})$로 둔다. 비용 분리성에 의해 $c(b)\le c(a)$이다. 한
번에 한 축씩 교체하면 식 (6)에 의해

$$
\begin{aligned}
y_D(p_W,p_{KV})
&\le y_D(a_W,p_{KV})+V_W(\eta_W)\\
&\le y_D(a_W,a_{KV})+V_W(\eta_W)+V_{KV}(\eta_{KV})
\end{aligned}
$$

이므로 결론을 얻는다. $\square$

이 정리는 가중치 손실과 KV 손실이 가산적이라거나 두 축의 상호작용이
작다고 주장하지 않는다. 1단계 순서를 뒤집는 상호작용 성분에 대해서만
오차를 지불한다. 따라서 포화 현상이 절대 목적값을 크게 바꾸더라도
$V_i$는 작을 수 있다. 또한 실제 탐색 공간이 $\mathcal{C}$를 계속
포함한다면 돌연변이는 정리를 무효화하지 않는다. 돌연변이는 후보 집합을
확장하므로 얻을 수 있는 최선의 값은 나빠지지 않는다.

다음은 더 강하지만 해석하기 쉬운 충분조건이다.

**따름정리 1 (단조 충분 요약).** 모든 $(w,k)\in\mathcal{X}$에 대해

$$
\left|y_D(w,k)-F(z_W(w),z_{KV}(k))\right|\le\epsilon_0
\tag{8}
$$

를 만족하는 성분별 비감소 함수 $F$가 존재하고, $F$가 각 좌표에 대해
$L_W,L_{KV}$의 Lipschitz 상수를 갖는다고 하자. 그러면 정리 1의 점 $b$는

$$
y_D(b)\le y_D(a)+2\epsilon_0+L_W\eta_W+L_{KV}\eta_{KV}
\tag{9}
$$

를 만족한다.

*증명.* 식 (8)을 $a$와 $b$에 적용하고, cover 관계에는 단조성을,
$\eta_i$ 오차에는 Lipschitz bound를 적용한다. 잔차는 두 끝점에서만
발생하므로 축마다 한 번이 아니라 총 $2\epsilon_0$이 된다. $\square$

#### 실증 분석

1단계 frontier에서 가중치 블록 20개와 유효 KV 블록 20개를 비용에 따라
층화 표본화하고, **동일한 20개 블록이 모든 상대 축과 대응되도록** 400개
Cartesian product 전체를 AWQ로 측정하였다. 독립적인 무작위 400개를
사용하면 특정 가중치 블록에서 본 KV 순위와 다른 가중치 블록에서 본 KV
순위를 직접 비교할 수 없다. 대응 격자는 이 교란을 제거한다.

각 축에 대해 두 종류의 통계를 계산하였다. 먼저 상대 축의 두 partner가
만드는 20개 구성의 순위를 비교한다. partner 쌍은 $\binom{20}{2}=190$개다.
다음으로 프록시 축 점수가 더 좋다고 한 모든 블록 쌍을 20개 partner에서
직접 대조한다. 축마다 총 $190\times20=3{,}800$개의 교체 비교가 생기며,
여기서 배포 손실이 증가한 경우만 식 (6)의 유해 위반으로 센다.

| 측정 프로토콜 | 교체 축 | partner 간 Kendall $\tau$ 중앙값 / 최솟값 | 유해 위반 | 위반 마진 중앙값 / 최댓값 |
|---|---:|---:|---:|---:|
| stride-128 | 가중치 | 0.989 / 0.989 | 8 / 3,800 (0.21%) | 0.00269 / 0.01025 |
| stride-128 | KV | 0.989 / 0.942 | 12 / 3,800 (0.32%) | 0.00090 / 0.00342 |
| stride-32 | 가중치 | 1.000 / 1.000 | 0 / 3,800 (0.00%) | -- |
| stride-32 | KV | 0.979 / 0.923 | 23 / 3,800 (0.61%) | 0.00073 / 0.00195 |

주 분석인 stride-128에서는 7,600개 비교 중 20개(0.26%)만 순서를
위반하였다. 최악의 경우도 넓은 영역의 체계적 교차가 아니었다. 가중치
2.25비트와 2.40비트 partner가 만든 KV 순위를 비교했을 때 190개 KV 쌍
중 5개가 불일치했고, 해당 AWQ JSD 차이의 중앙값과 최댓값은 각각
0.0010과 0.0020이었다. 즉 최소 $\tau=0.942$는 주로 공격적인 저비트
corner의 근접 동률에서 발생한다.

정리 1에 직접 들어가는 측정 순서 위반 마진은

$$
\widehat V_W=0.0103,\qquad
\widehat V_{KV}=0.0034
$$

이며, 합은 0.0137로 관찰된 stride-128 JSD 범위
$0.5391-0.0138=0.5253$의 2.6%다. 식 (7)은 평균적인 순서가 아니라 가장
큰 유해 순위 역전에 의존하므로, 이 마진은 상관계수만 보고하는 것보다
정리에 직접 연결되는 근거다.

별도의 two-way additive 분해에서는 가중치 주효과가 전체 변동의 74.5%,
KV 주효과가 23.7%, interaction과 측정 잔차를 합친 항이 1.73%를
차지하였다. 반복 측정이 없는 격자이므로 마지막 항을 순수한 interaction
분산으로 해석할 수는 없다. 다만 잔차의 인접 cell 자기상관이 0.96이고
stride-32 잔차와도 0.62의 상관을 보여 구조화된 상호작용이 존재한다.
따라서 이 결과는 “손실이 가산적이다”가 아니라, **상호작용은 존재하지만
검증한 frontier product 안에서는 축별 순서를 거의 뒤집지 않는다**는
정리 1의 더 약한 조건을 지지한다.

![대응된 20×20 격자에서 순서와 값의 예비 분석.](../tests/docs/fig/fig6_order_vs_values.png)

그러나 400개 cell은 frontier 위의 블록만 분석한다. 돌연변이가 만든
off-front 블록은 식 (6)에서 교체 대상과 상대 축의 역할을 모두 할 수
있으며, 이 경우는 아직 충분히 분석하지 않았다. 또한 한 번의 탐색으로
얻은 archive가 식 (5)를 만족하는지는 자동으로 보장되지 않는다.
따라서 현재 수치는 분석한 frontier product에 한정된 실증 결과로
해석한다. 더 넓은 주장을 위해 필요한 반증 실험은 3.8절에서 제안한다.

### 3.5 양자화 프록시는 어디까지 사용할 수 있는가?

동일한 비용 계산을 사용하는 프록시 HQQ와 배포 방법 AWQ의 손실을 각각
$y_P,y_D$라 하고, 각각의 Pareto 집합을 $\operatorname{PF}_P$와
$\operatorname{PF}_D$라 하자. AMQ의 전역 순서 동치 조건은 두 집합이
같아지는 충분조건이다 [4]. 여기서는 그 충분조건을 다시 제시하는 대신,
프록시 목적의 불일치가 탐색 예산으로 해결되지 않는 조건을 보인다.

**정리 2 (프록시 전용 탐색의 Pareto 비일관성).** 유한한
$\mathcal{X}$에서 어떤 $x^\star\in\operatorname{PF}_D$와
$z\in\mathcal{X}$가 다음을 만족한다고 하자.

$$
\begin{gathered}
c(z)\le c(x^\star),\qquad y_P(z)\le y_P(x^\star),\\
(c(z),y_P(z))\ne(c(x^\star),y_P(x^\star)),\qquad
y_D(z)>y_D(x^\star).
\end{gathered}
\tag{10}
$$

그러면 $x^\star\notin\operatorname{PF}_P$이다. 따라서 평가 횟수가
늘어날수록 프록시 Pareto 집합만 보존하는 탐색은
$\operatorname{PF}_D$ 전체로 수렴할 수 없다.

더 나아가 배포 예산 $B$의 가능 집합을
$\mathcal{X}_B=\{x:c(x)\le B\}$, 프록시 최적점 집합을
$S_P(B)=\arg\min_{x\in\mathcal{X}_B}y_P(x)$라 하자. 다음의 배포 gap이

$$
\delta_D(B)=
\min_{x\in S_P(B)}y_D(x)
-\min_{x\in\mathcal{X}_B}y_D(x)>0
\tag{11}
$$

이면, 프록시 regret이 0으로 수렴하는 모든 선택열 $x_n$은

$$
\liminf_{n\rightarrow\infty}
\left[y_D(x_n)-\min_{x\in\mathcal{X}_B}y_D(x)\right]
\ge\delta_D(B)>0
$$

를 만족한다. 즉 프록시 평가를 더 수행해도 이 배포 regret 하한은
사라지지 않는다.

*증명.* 식 (10)의 앞 세 조건에 의해 $z$가 프록시 목적과 비용에서
$x^\star$를 지배하므로 $x^\star$는 $\operatorname{PF}_P$에 속할 수
없다. 그러나 마지막 부등식 때문에 이 지배 관계는 배포 목적에서
보존되지 않는다. 따라서 프록시 frontier에 수렴하는 archive는
$x^\star$를 영구적으로 제외한다. 두 번째 주장에서는
$\mathcal{X}_B\setminus S_P(B)$가 비어 있지 않다면 유한성에 의해 이
집합과 프록시 최솟값 사이에 양의 gap이 존재한다. 따라서 프록시
regret이 0으로 수렴하면 충분히 큰 $n$에서 $x_n\in S_P(B)$이다. 차집합이
비어 있으면 식 (11)의 양의 gap 자체가 성립할 수 없다. 어느 경우든 식
(11)이 배포 regret의 하한을 준다. $\square$

정리의 전제는 단순히 “순위 역전이 하나 있다”보다 강하다. 역전된 점이
실제 Pareto 점을 프록시 목적과 비용에서 지배해야 frontier 누락의
증명서가 된다. 반대로 높은 전역 상관관계는 이 증명서가 없음을 보장하지
않는다. 예산 간 큰 비트 차이가 전역 상관관계를 지배하는 반면, 최종
선택은 거의 같은 비용을 서로 다른 레이어에 배치한 후보의 순위로
결정되기 때문이다.

현재 측정은 후보 공급과 최종 선택에서 이 차이를 직접 보여 준다.
가중치 축과 KV 축의 1단계 frontier 블록에서는 HQQ--AWQ Spearman
$\rho$가 각각 1.000이다. 더 큰 4,365개 paired production archive에서도
HQQ의 10% loss band는 후보의 12.5%만 남기면서 예산별 AWQ 최적점의
95.2%를 포함했다. 즉 HQQ는 1단계에서 유망 후보를 **공급**하는 데
유용하다.

반면 평균 가중치 비용과 유효 KV 비용을 맞춘 183개 cell의 세밀한
레이어 할당에서는 HQQ--AWQ Kendall $\tau$ 중앙값이 0.295, Spearman
$\rho$ 중앙값이 0.398로 감소했다. HQQ의 최상 구성이 실제 AWQ 최상
구성과 일치한 cell은 21.9%뿐이었다. HQQ 최상점을 배포했을 때의 상대
AWQ regret은 중앙값 1.69%, 90분위 6.90%, 최댓값 12.87%였다. 이는 식
(11)의 gap이 실제 선택 해상도에서 자주 양수임을 보여 준다.

![1단계 후보 공급과 2단계 최종 선택에서 요구되는 프록시 충실도의 차이.](../visualize/hqq_awq/fig/narr2_two_requirements.png)

Pareto 집합 자체도 같은 후보 pool에서 직접 비교하였다. 정확한 지배를
사용하면 AWQ target front 2,304개 중 1,656개만 HQQ proxy front와
겹쳤다. Target-front recall은 71.9%, Jaccard overlap은 0.555이며,
제외된 648개 모두가 식 (10)의 엄격한 프록시 지배 증명서를 측정 archive
안에서 갖는다. JSD 차이 $10^{-3}$을 동률로 처리해도 target front
1,654개와 proxy front 1,677개의 교집합은 1,105개에 그쳤다(recall 66.8%,
Jaccard 0.496). 동일한 AWQ loss에 $[-10^{-3},10^{-3}]$ 잡음만 가한 5회
null의 Jaccard는 $0.825\pm0.007$이었다. 관찰된 front 차이는 단순 측정
해상도에서 예상되는 membership churn보다 훨씬 크다. 다만 이는 측정된
production archive에 대한 실증 증명서이지, 아직 평가하지 않은 전체
조합 공간의 front 크기나 recall을 추정한 것은 아니다.

이에 따라 HQQ와 AWQ의 역할을 명확히 구분한다. HQQ는 (i) 두 축 탐색,
(ii) frontier 블록 선택, (iii) 저차원 표현 학습에 사용한다. AWQ label은
결합 surrogate 학습, 2단계 archive 갱신, 최종 구조 선택에 사용한다.
그렇다고 모든 결합 후보마다 별도의 AWQ 빌드가 필요한 것은 아니다.
소수의 AWQ 정보로 보정 모델을 학습할 수 있고, 3.7절의 방법은 한 번의
빌드를 여러 후보에 분산한다. 따라서 본 논문의 정확한 주장은 결합 공간
내부에서 **배포 방법의 정보**가 필요하다는 것이며, 배포 방법을 이용한
전수 탐색이 필요하다는 것이 아니다.

### 3.6 프록시 지도 surrogate 임베딩

현재 이산 genome은 352개 cell로 구성된다. 이 중 224개는 가중치 cell,
128개는 KV 비트/그룹 및 가지치기 cell이다. 정수형 ordinal encoding은
인위적인 선형 기하를 가정한다. 예를 들어 3비트에서 2비트로 낮출 때의
오차 증가가 4비트에서 3비트로 낮출 때와 같을 이유는 없다. One-hot
encoding은 이 문제를 제거하지만 가중치와 KV 표현을 각각 672차원과
896차원으로 확장한다. 두 방식 모두 약 100개 AWQ 빌드로 구성된 초기
설계에 비해 지나치게 고차원이다.

대규모 1단계 HQQ archive를 이용해 지도 표현을 학습한다. $h_W(w)$와
$h_{KV}(k)$를 cell별 one-hot encoding이라 하자. 각 축에서 PLS는 1단계
JSD의 제곱근과 공분산이 큰 좌표를 찾는다.

$$
\begin{aligned}
R_W &= \operatorname{PLS}_8\!\left(h_W(w),\sqrt{z_W(w)}\right),\\
R_{KV} &= \operatorname{PLS}_8\!\left(h_{KV}(k),\sqrt{z_{KV}(k)}\right)
\end{aligned}
\tag{12}
$$

결합 surrogate의 입력은 다음과 같다.

$$
\phi(w,k)=\left[
R_W^\top h_W(w),
R_{KV}^\top h_{KV}(k),
c_W(w),c_{KV}(k)
\right]\in\mathbb{R}^{18}.
\tag{13}
$$

Matérn-$3/2$ kernel을 사용하는 ARD Gaussian process가 $\phi$에서
$\sqrt{y_D}$를 예측한다. HQQ는 표현을 학습하는 감독 정보를 제공하지만,
regression head와 최종 선택에 사용되는 모든 목적값은 AWQ label에서
학습한다.

현재 기본값은 축별 8개 성분이다. 1단계 hold-out data에서 프록시 target
재구성 $R^2$는 8개 성분일 때 가중치 0.983, KV 0.992다. KV 점수는
12--16개 성분에서 조금 감소한다. 현재 AWQ archive에서 18차원 production
pipeline의 cell 내부 평균 Spearman 상관계수는 학습 표본 100개에서
$0.640\pm0.054$, 430개에서 $0.700\pm0.008$이며, 예산 band 상관계수는
각각 0.955와 0.970이다. 다만 이는 sanity check이지 PLS가 표본 효율을
인과적으로 높인다는 완전한 증거는 아니다. 최종 비교에서는 test set,
학습 표본 수, surrogate head를 모두 고정해야 한다.

따라서 향후 ablation은 같은 head와 고정된 hold-out 후보에서 raw ordinal,
one-hot, PCA, $d\in\{2,4,8,16\}$인 PLS, AWQ archive만으로 학습한 self-PLS를
비교한다. 학습 크기는 $N\in\{25,50,100,200,400\}$으로 설정한다. 하나의
AWQ 빌드를 공유하는 KV companion이 train과 test에 동시에 들어가 정보가
누출되지 않도록 가중치 할당 단위로 split한다. 전역 및 동일 예산
Spearman 상관계수, RMSE, 예산별 top-1 regret, Pareto hypervolume을
보고한다. 이 설계는 단순한 PLS 공간 재구성 품질이 아니라, **AWQ label
하나당 더 유용한 의사결정을 만드는가**라는 실제 주장을 검증한다.

### 3.7 Multi-KV 구성 평가

AWQ 보정과 가중치 생성은 $w$에 의존하지만 선택한 KV 캐시 구성에는
의존하지 않는다. 따라서 하나의 가중치 anchor를 AWQ로 한 번 빌드한 뒤
캐시 구현만 교체하여 $K$개의 KV companion을 가중치 재빌드 없이 평가할
수 있다. $K$가 커질수록 같은 가중치 빌드 비용에서 얻는 결합 label 수가
늘어나며, 절감 폭은 가중치 빌드가 KV 교체 및 평가보다 비쌀수록 커진다.
현재 구현의 GPU 정확성 시험에서 KV를 교체한 모델과 독립적으로 다시
빌드한 모델의 JSD가 정확히 일치하였다. 첫 구성의 빌드 및 평가는 약
492초가 걸렸고, 재사용 구성은 현재 로그의 1초 미만 해상도 안에서
완료되었다.

선택된 각 가중치 anchor에는 1단계 KV pool의 companion을 연결한다.
먼저 예측된 3목적 Pareto 집합으로 후보를 필터링한 뒤, 측정 archive를
기준으로 $(c_W,c_{KV})$ 평면을 넓게 덮도록 companion을 선택한다.
Surrogate의 tail이 평평하면 높은 용량의 끝점이 지배되어 제거될 수
있으므로, 필요한 경우 낮은 KV와 높은 KV 끝점을 강제로 포함한다. 이를
통해 예산 경계에서 surrogate가 보정되지 않은 상태로 남는 것을 막는다.

Multi-KV 평가는 한 번의 빌드에서 label 수를 늘리지만 군집된 data를
만든다. 큰 $K$는 소수 가중치 할당의 KV 반응을 조밀하게 측정한다. 반면
총 label 수나 총 시간이 고정되면 작은 $K$가 더 다양한 가중치 할당을
탐색할 수 있다. 따라서 “평가한 architecture 수”만 보고하면 실질적인
독립 표본 수를 과장하게 된다. 본 논문에서는 $K$를 **가중치 빌드당 전체
구성 수**로 정의한다. 즉 $K=1$은 companion이 없는 경우다. 다음 세 가지
관점에서 ablation을 수행한다.

1. **고정 빌드 예산:** AWQ 가중치 anchor 수 $B$를 고정하고
   $K\in\{1,5,10,20,40\}$을 비교한다. 비싼 빌드 하나당 저비용 KV
   label의 한계 가치를 측정한다.
2. **고정 label 예산:** 전체 결합 label 수를 고정한다. 독립적인 가중치
   coverage와 한 가중치 내부의 KV 해상도 사이 trade-off를 확인한다.
3. **고정 실행 시간:** 실제로 측정한 빌드, KV 교체, 평가 시간을 이용해
   같은 GPU-hour를 할당한다. 긴 문맥에서 KV 평가 비용을 무시할 수 없을
   때 실제로 중요한 비교다.

가중치 빌드가 25, 50, 100, 200개에 도달한 시점마다 전체 run의 측정값을
합친 reference frontier에 대한 hypervolume과 inverted generational
distance, $8\times8$ 예산 격자 coverage, cell 내부 surrogate 순위
상관관계, 최종 예산별 top-1 regret, 서로 다른 가중치 family 수, family별
최대 KV gap을 측정한다. 모든 지표는 최소 3개 탐색 seed에 대해 평균하고,
예측기 검증은 가중치 family 단위로 묶는다. 현재 구현의 설정은 anchor
하나와 companion 10개이므로 이 정의에서 $K=11$이다. Ablation이 완료되기
전까지 이를 최적값이 아닌 동작 기본값으로 취급한다.

### 3.8 가정 검증 및 반증 프로토콜

앞의 이론은 조건부 주장이다. 따라서 실험은 전역 상관계수를 나열하기보다
각 이론 가정을 직접 검증하는 형태로 구성해야 한다. 최소한 다음 분석이
필요하다.

#### 대응된 $20\times20$ 순위 격자

각 축의 $\eta$-front에서 비용에 따라 블록 20개씩을 표본화하고, 400개
모든 곱을 HQQ와 AWQ로 평가한다. 상대 축별 Kendall $\tau$, Spearman
$\rho$, 쌍별 불일치율, 최악의 위반 마진
$\widehat V_W,\widehat V_{KV}$를 보고한다. 정리 1에 직접 연결되는 통계는
평균 $\tau$가 아니라 최대 위반 마진이다. 높은 $\tau$도 하나의 치명적인
순위 역전을 숨길 수 있다.

#### Off-front 교체 경로

최소 30개의 off-front 또는 돌연변이 결합 구조 $a$를 표본화한다. 각
구조의 축별 frontier 투영을 구하고, 순차 교체 증명에 필요한
$(a_W,a_{KV})$, $(a_W,p_{KV})$, $(p_W,p_{KV})$를 측정한다. 투영된 끝점을
재사용하면 약 60회의 새 AWQ 평가가 필요하다. 이 실험은 기존 on-front
격자가 포함하지 않는 교체 대상 및 상대 축 역할을 직접 검증한다.

#### 1단계 frontier 충실도

각 축 탐색을 서로 다른 seed로 반복하고, 상호 지배 관계, hypervolume
차이, 다른 run의 frontier가 $\eta$-cover되는 비율을 측정한다. 이는 식
(5)에 숨겨진 1단계 오차를 추정한다. 이 분석이 없으면 정리 1은 참
frontier에는 적용되지만 실제 알고리즘이 찾은 frontier에 적용된다고
주장할 수 없다.

#### 프록시 적용 범위

HQQ--AWQ 일치도를 축별, 예산 간, 고정 $(c_W,c_{KV})$ cell 내부, 동일
cell에서 구조적으로 먼 쌍이라는 네 해상도에서 보고한다. 예산별로 HQQ
최상 구조의 AWQ regret과 HQQ top-$q$ shortlist 안에 AWQ 최상 구조가
포함되는 비율을 측정한다. 상관관계와 함께 Pareto overlap 및 지배
hypervolume, 식 (10)을 만족하는 target-front 제외점의 수, 식 (11)의
예산별 regret gap을 제시해야 한다. 높은 전역 $R^2$나 시각적으로 비슷한
frontier만으로 프록시 지배 증명서가 없다고 결론 내릴 수 없다.

#### 일반화

전체 분석을 Qwen2.5-7B처럼 attention 구조가 다른 모델 하나 이상과 긴
문맥 dataset에서 반복해야 한다. 탐색 효율을 위한 주 지표는 answer-phase
JSD로 유지할 수 있지만, 최종 구조는 perplexity 및 LongBench, RULER와
같은 장문 과제에서 평가해야 한다 [14, 15]. 탐색 지표의 Pareto 개선만으로
실제 배포 품질이 향상되었다고 결론 내릴 수 없다.

### 3.9 현재 이론적 보장의 한계

정리 1은 존재성 정리이며 NSGA-III나 surrogate의 수렴 정리가 아니다.
종단간 regret에는 추가로 (i) 1단계 frontier 오차, (ii) 측정한 순서 위반
마진, (iii) product 최적점 대비 2단계 탐색 오차, (iv) 거의 같은 구성
사이의 측정 잡음이 포함된다. 이 항들은 분리해서 보고해야 한다. 현재
근거는 하나의 모델과 하나의 compression family에 집중되어 있다.
정리 2 역시 모든 프록시가 항상 실패한다는 무조건적 불가능성 정리가
아니다. 식 (10)의 지배 증명서 또는 식 (11)의 양의 gap이 있을 때
프록시-only 수렴이 실패한다는 조건부 결과다. 현재 4,365개 archive는
실제 운영 후보에 대한 직접 증명서를 제공하지만 AWQ 탐색으로 수집된
집합이므로, 전체 조합 공간의 front 누락률을 추정하려면 독립적인 held-out
후보와 다른 모델에서 같은 감사를 반복해야 한다.
Strided JSD의 높은 상관관계도 독립적인 모델과 구성 pool에서 재현되어야
한다. 마지막으로 product 안의 더 저비용인 점이 양방향 메모리 band의
하한보다 낮아질 수
있다. 정리의 문자 그대로의 결론은 비용 상한 제약에 적용된다. 양방향
보고 구간에서는 더 비싸지 않은 근사 대체점이 존재함을 보장하지만, 그
대체점이 같은 구간 안에 남는다고 보장하지는 않는다.

## 참고문헌

1. Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. “GPTQ:
   Accurate Post-Training Quantization for Generative Pre-trained
   Transformers.” ICLR, 2023. <https://arxiv.org/abs/2210.17323>
2. Ji Lin, Jiaming Tang, Haotian Tang, et al. “AWQ: Activation-Aware Weight
   Quantization for On-Device LLM Compression and Acceleration.” MLSys, 2024.
   <https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html>
3. Hicham Badri and Appu Shaji. “Half-Quadratic Quantization of Large Machine
   Learning Models.” 2023. <https://dropbox.github.io/hqq_blog/>
4. Sangjun Lee, Seung-taek Woo, Jungyu Jin, Changhun Lee, and Eunhyeok Park.
   “AMQ: Enabling AutoML for Mixed-Precision Weight-Only Quantization of Large
   Language Models.” 2025. <https://arxiv.org/abs/2509.12019>
5. Zirui Liu, Jiayi Yuan, Hongye Jin, et al. “KIVI: A Tuning-Free Asymmetric
   2bit Quantization for KV Cache.” ICML, 2024.
   <https://proceedings.mlr.press/v235/liu24bz.html>
6. Yuhui Xu, Zhanming Jie, Hanze Dong, et al. “ThinK: Thinner Key Cache by
   Query-Driven Pruning.” ICLR, 2025.
   <https://openreview.net/forum?id=n0OtGl6VGb>
7. Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, et al. “KVQuant: Towards 10
   Million Context Length LLM Inference with KV Cache Quantization.” NeurIPS,
   2024. <https://arxiv.org/abs/2401.18079>
8. Xing Li, Zeyu Xing, Yiming Li, et al. “KVTuner: Sensitivity-Aware Layer-Wise
   Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM
   Inference.” ICML, 2025. <https://arxiv.org/abs/2502.04420>
9. Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan. “A Fast and
   Elitist Multiobjective Genetic Algorithm: NSGA-II.” IEEE Transactions on
   Evolutionary Computation, 2002. <https://doi.org/10.1109/4235.996017>
10. Kalyanmoy Deb and Himanshu Jain. “An Evolutionary Many-Objective
    Optimization Algorithm Using Reference-Point-Based Nondominated Sorting
    Approach, Part I: Solving Problems with Box Constraints.” IEEE Transactions
    on Evolutionary Computation, 2014.
    <https://doi.org/10.1109/TEVC.2013.2281535>
11. Bowen Baker, Otkrist Gupta, Ramesh Raskar, and Nikhil Naik. “Accelerating
    Neural Architecture Search Using Performance Prediction.” 2017.
    <https://arxiv.org/abs/1705.10823>
12. Colin White, Arber Zela, Robin Ru, Yang Liu, and Frank Hutter. “How Powerful
    Are Performance Predictors in Neural Architecture Search?” NeurIPS, 2021.
13. Svante Wold, Michael Sjöström, and Lennart Eriksson. “PLS-Regression: A
    Basic Tool of Chemometrics.” Chemometrics and Intelligent Laboratory
    Systems, 2001. <https://doi.org/10.1016/S0169-7439(01)00155-1>
14. Yushi Bai, Xin Lv, Jiajie Zhang, et al. “LongBench: A Bilingual, Multitask
    Benchmark for Long Context Understanding.” ACL, 2024.
    <https://arxiv.org/abs/2308.14508>
15. Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, et al. “RULER: What's the Real
    Context Size of Your Long-Context Language Models?” COLM, 2024.
    <https://arxiv.org/abs/2404.06654>
