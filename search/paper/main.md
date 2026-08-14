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
최악 경우(minimax) 분석은 구조 없는 $d$차원 직접 탐색의 평가 수가 목표 오차에
대해 지수적으로 증가하는 반면, 축별 Pareto screening은 각 축 차원의
지수 비용을 합하는 형태로 바뀜을 보인다. 이어지는 front-product
coverage 정리는 상대 축에 의해 발생하는 순서 위반이 작을 경우 이
제한된 공간이 결합 공간의 Pareto 집합을 근사적으로 포함함을 보인다. 이
결과는 손실의 가산성을 요구하지 않는다. 나아가 그 축별 순위 자체가
파트너를 공유하는 매칭 관측에서만 식별됨을 보여, 축별 탐색을 순위를 얻는
최소 실험설계로 정당화한다. 또한 양자화
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
   포함한다. 그리고 그 축별 순서는 상대 축을 고정한 매칭 비교에서만
   식별되므로, 축별 탐색은 편의상의 단순화가 아니라 이 정보를 얻는
   관측 설계다.
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

표본 효율의 이점도 조건부로 정식화한다. Smoothness 이외의 구조가 없는
$d$차원 Lipschitz 목적은 오차 $\epsilon$에 대해 worst case
$\Omega(\epsilon^{-d})$개의 평가가 필요하다. 반면 각 축의 유효 차원이
$d_i$이고 축별 목적이 안정적이면, 축별 Pareto cover를 만드는 비용은
$O(\sum_i\epsilon^{-d_i})$로 분해된다. 이 결과는 2단계 AWQ 비용까지
자동으로 줄어든다는 주장이 아니다. 축별 cover가 실제 joint optimum을
버리지 않는다는 다음 조건이 함께 필요하다.

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
  검증 가능한 front-product coverage 정리를 제시한다. 구조 없는 joint
  탐색의 차원 하한과 축별 Pareto screening의 조건부 표본복잡도를 함께
  제시하여 공간 제한의 계산적 이점과 품질 조건을 분리한다.
- 실제 배포 Pareto 점에 대한 프록시 지배가 영구적인 frontier 누락을
  만들고, 예산별 프록시 최적화에는 배포 regret 하한이 남는다는 프록시
  비일관성 정리를 제시한다. 이를 통해 HQQ는 1단계를 안내할 수 있지만
  2단계에는 AWQ 정보가 필요한 이유를 설명한다.
- 비싼 AWQ 빌드당 유용한 label 수를 늘리기 위해 프록시 지도 PLS 차원
  축소와 가중치 빌드 공유형 multi-KV 평가를 제안한다. 비모수 최악 경우
  학습률을 통해 필요한 표본 수가 표현의 유효 차원과 smoothness에 어떻게
  의존하는지 명시하고, 고정된 최소 표본 수 대신 grouped learning curve로
  운영 임계점을 정한다.
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

### 2.3 고차원 블랙박스 최적화와 표본복잡도

구조를 가정하지 않은 $d$차원 Lipschitz 전역 최적화는 worst case에서
simple regret이 $N^{-1/d}$보다 빠르게 감소할 수 없고 [16], 잡음이 있는
$s$-smooth 비모수 회귀의 최적 MSE 수렴률도 $N^{-2s/(2s+d)}$이므로
차원은 surrogate 학습 속도에 직접 들어간다 [17]. 고차원 Bayesian
optimization은 additive 구조 [18], 낮은 유효 차원 [19], information
gain 기반 분석 [20, 21], 다목적 $\epsilon$-PAL [22]처럼 추가 구조
가정으로 이 장벽을 우회한다. 공통점은 필요한 표본 수가 입력 좌표 수
하나로 결정되지 않고 함수족, kernel, smoothness, 잡음, 유효 차원, 목표
오차에 의존한다는 것이다. 따라서 본 연구는 “18차원 표현에는 18개
label이면 충분하다”와 같은 고정 표본 주장을 하지 않고, 조건부 이론과
학습곡선을 함께 사용한다. 문헌 상세는 부록 E에 있다.

### 2.4 KV 캐시 압축

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

### 2.5 본 연구의 위치

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

즉 식 (S-JSD)는 **prefix를 한 번 읽은 뒤 cache를 실제처럼 누적하면서,
answer 위치에서 측정한 JSD를 평균한다**는 뜻이다. 이 지표는 정답 token을
입력하는 teacher-forced metric이므로
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

### 3.3 고차원 직접 탐색의 표본 비효율성

먼저 핵심 직관을 설명한다. 한 축을 오차 $\epsilon$ 수준으로 확인하는 데
대략 $1/\epsilon$개의 구간이 필요하다면, $d$개의 독립적인 축을 함께
확인할 때는 약 $(1/\epsilon)^d$개의 구간이 생긴다. 차원이 하나 늘 때마다
필요한 평가 수가 더해지는 것이 아니라 곱해지는 이유다.

**정리 1 (차원의 저주에 의한 평가 하한, 요약).** smoothness 이외의
구조가 없는 $d$차원 공간에서는 어떤 적응적 알고리즘을 쓰더라도 $N$회의
잡음 없는 평가 후 worst-case simple regret이 $\Omega(LaN^{-1/d})$ 아래로
내려갈 수 없다 [16]. 평가 수의 형태로 바꾸면, worst-case simple
regret을 $\epsilon$ 이하로 만들기 위해

$$
N_{\mathrm{direct}}(\epsilon)
=\Omega\!\left((La/\epsilon)^d\right)
$$

개의 평가가 필요하다. 즉 목표 오차를 절반으로 줄이거나 차원을 하나
늘리는 비용이 고차원에서는 빠르게 커진다. Packing number를 이용한
형식적 진술, 증명 개요, 유한 이산 공간에서의 해석 범위는 부록 A.1에
있다.

**해석.** 이 정리에 현재 genome의 352개 cell을 그대로 $d=352$로
대입해서는 안 되며, 반대로 공간이 유한하다는 사실만으로 탐색이 쉬워지는
것도 아니다. 정리 1은 **구조를 활용하지 않는 직접 joint 탐색의 worst
case**를 설명하며, 현재 문제가 이 비율을 따르는지는 부록 C.1의 차원별
실험으로 별도 검증한다.

### 3.4 2단계 탐색 개요

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

1. 현재 archive와 3.7절의 프록시 지도 임베딩으로 AWQ-loss surrogate를
   학습한다.
2. NSGA-III로 product, crossover, mutation 후보를 생성한다.
3. 현재 $(c_W,c_{KV})$ Pareto 기하를 잘 덮는 가중치 anchor를 선택하고,
   각 anchor에 서로 다른 $K-1$개의 KV companion을 연결한다.
4. 서로 다른 가중치 anchor마다 AWQ 가중치를 한 번 빌드하고, 연결된 모든
   KV companion을 평가한 뒤 성공한 측정값을 archive에 추가한다.

이 구조에서 1단계는 유용한 블록을 식별하는 **순서적 역할**을 하고,
2단계는 loss surface를 학습하여 결합 예산에서 구성을 결정하는 **값
추정 역할**을 한다.

### 3.5 축별 frontier를 먼저 찾는 이유

축별 탐색이 유리하려면 세 질문을 분리해야 한다.

1. **축별 순위라는 정보는 어떤 관측에서 얻어지는가?** joint 표본 하나의
   손실에는 두 축의 기여가 섞여 있으므로, 축별 순위가 흩어진 joint
   관측에서 복원 가능한 정보인지부터 확인해야 한다.
2. **축별 후보를 얼마나 적은 평가로 추릴 수 있는가?** 낮은 차원에서
   따로 탐색하면 전체 차원의 곱 대신 축별 비용의 합만 지불한다.
3. **그렇게 추려도 좋은 joint 후보를 버리지 않는가?** W나 KV를 축별로
   더 좋은 후보로 교체했을 때 AWQ 손실이 얼마나 나빠질 수 있는지를
   확인해야 한다.

첫 질문은 정리 2가, 두 번째 질문은 명제 1이, 세 번째 질문은 정리 3이
다룬다. 1단계의 축별 순서는 HQQ로 얻지만, 세 번째 질문의 실제 손실은
AWQ로 측정한다.

#### 축 순위는 매칭된 파트너 비교로만 식별된다

1단계가 생산하는 것은 축별 후보의 **순위**다. 그런데 이 순위는 joint
공간의 흩어진 표본에서 사후적으로 추출할 수 있는 정보가 아니다. joint
구성 하나를 평가하면 손실 숫자 하나가 나오고, 그 숫자에는 가중치 구성의
기여와 KV 구성의 기여가 섞여 있다. 서로 다른 KV 파트너에서 평가된 두
가중치 구성을 비교하는 것은 서로 다른 난이도의 시험을 본 두 학생의
점수로 실력을 비교하는 것과 같다. 손실이 정확히 가산적이라면 파트너
효과를 추정해 빼는 보정이 가능하다. 그러나 이 절 말미의 실증이 보이듯
실제 손실은 공격적 영역에서 포화하는 비가산 합성이므로, 파트너는 점수에
상수를 더하는 것이 아니라 구성 간 차이를 비선형으로 압축한다. 이 경우
보정은 원리적으로 실패한다. 다음 정리가 이를 형식화한다.

**정리 2 (축 순위의 매칭 설계 식별).** $y(w,k)=F(z_W(w),z_{KV}(k))$이고,
$F$는 각 인자에 대해 순증가하지만 그 외에는 알려져 있지 않으며
$z_W,z_{KV}$도 알려져 있지 않다고 하자.

(i) **비매칭 관측의 비식별성.** 관측 집합 $\{y(w_j,k_j)\}_{j=1}^N$에서
어떤 두 관측도 KV 파트너를 공유하지 않는다고 하자. 서로 다른 파트너와
함께 관측된 임의의 두 가중치 구성 $u\ne u'$에 대해, 모든 관측값을
재현하면서 $z_W(u)<z_W(u')$인 $(F,z_W,z_{KV})$와 그 반대 순서인
$(\tilde F,\tilde z_W,\tilde z_{KV})$가 모두 존재한다. 따라서 함수형
가정을 추가하지 않는 한 어떤 추정량도 비매칭 관측에서 축 순위를 식별할
수 없다.

(ii) **무작위 설계의 매칭 희소성.** KV 주변분포가
$\mathcal{X}_{KV}$에서 균일한 i.i.d. 표본 $N$개에서 파트너를 공유하는
쌍 수의 기댓값은 $\binom{N}{2}/|\mathcal{X}_{KV}|$이므로, 매칭쌍을
하나라도 얻으려면 $N=\Omega(\sqrt{|\mathcal{X}_{KV}|})$가 필요하다.

(iii) **축 슬라이스는 최소 매칭 설계다.** 설계
$\{(u,r_{KV}):u\in S\}$는 모든 관측쌍을 매칭시키므로, $|S|$번의 평가로
(i)의 함수족 아래에서 $S$의 전체 순위가 식별된다. 다중 파트너 격자
$\{(u,v):u\in S,v\in P\}$는 그 순위의 파트너 불변성, 즉 정리 3의 위반
마진 $V$까지 추가로 감사한다.

*증명 개요는 부록 A.2에 있다.*

**해석.** 이 정리는 joint 직접 탐색의 학습 신호가 어디서 오염되는지
설명한다. 진화 탐색이 표본 $(w,k)$를 낮은 손실 때문에 보존하면 $w$와
$k$ **모두**에게 공로를 배정한다. 평범한 가중치 구성이 관대한 KV 파트너
덕에 살아남고, 좋은 가중치 구성이 공격적인 파트너 탓에 버려진다. 평가
예산을 늘려도 비매칭 관측은 비매칭으로 남으므로 이 오염은 (ii)에 의해
줄지 않는다. 반면 적응적 알고리즘이 오염을 없애기 위해 의도적으로 매칭
비교를 구성한다면 그것은 정의상 축별 프로빙이며, 축 슬라이스는 그 설계
계열의 최소 인스턴스다. 실제로 본 연구의 상관 분석에 사용한 frontier 곱
표본 200개는 서로 다른 가중치 블록 183개를 포함해 순위·상호작용 질문에
답할 수 없었고, 3.5절의 대응 격자를 별도 설계로 구성해야 했다. 이것이
현재 조합 공간 규모에서 (ii)의 실물 사례다.

**한계.** 정리 2는 축 순위 **식별**의 결과이지 joint frontier
**발견**의 하한이 아니며, 가산 보정이 공격적 corner에서 실패한다는
측정과 결합될 때에만 힘을 갖는다. 상세와 매칭 대 비매칭 반증 실험
설계는 부록 A.2와 C.5에 있다.

다음으로, 남은 두 질문을 위해 “충분히 잘 추렸다”는 뜻을 정의한다. 축별 archive 밖의 임의의
후보를 하나 고르더라도, archive 안에 (i) 비용이 더 비싸지 않고
(ii) 프록시 손실도 최대 $\eta_i$만 나쁜 대체점이 있으면 충분하다고 본다.

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

식 (5)는 위 문장을 그대로 쓴 것이다. 유한 공간의 정확한 Pareto
집합에서는 $\eta_i=0$이고, 실제 archive에서는 $\eta_i$가 작을수록 1단계가
좋은 후보를 덜 놓쳤다는 뜻이다.

다음으로 **순서 위반 마진** $V_i$를 정의한다. 1단계는 $u$가 $u'$보다
비용과 프록시 손실에서 더 좋다고 판단했는데, 상대 축을 고정하고 실제
AWQ로 측정하니 오히려 손실이 증가할 수 있다. $V_i$는 이런 예상 밖
증가량 중 가장 큰 값이다.

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

여기서 $D=\mathrm{AWQ}$이고 $[q]_+=\max(q,0)$이다. 수식의 sup는 가능한
교체와 상대 축을 모두 확인해 최악의 경우 하나를 고른다는 뜻이다.

#### 축별 screening에 필요한 평가 수

명제 1의 핵심은 간단하다. W와 KV를 함께 촘촘히 덮으면 두 축의 격자
수가 **곱해지지만**, 따로 덮으면 두 격자 수를 **더하면** 된다. Covering
논증의 직접적 귀결이므로 정리가 아니라 조건부 명제로 분류한다.

**명제 1 (축별 Pareto screening의 조건부 차원 이점).** 축 $i$를 거리 $r$
간격으로 덮는 데 필요한 대표점 수가 최대 $(A_i/r)^{d_i}$이고, 축별
손실 $z_i$가 거리 변화에 따라 급격히 바뀌지 않는 $L_i$-Lipschitz
함수라고 하자. 그러면 식 (5)의 $\eta_i$-cover를 만드는 데 충분한
프록시 평가 수는

$$
N_{\mathrm{axis}}
=O\!\left(
\sum_i (A_iL_i/\eta_i)^{d_i}
\right).
\tag{Axis-SC}
$$

여기서 $A_i$와 $L_i$는 공간의 크기와 손실 변화율을 반영하는 상수이고,
중요한 항은 지수 $d_i$다. 반면 joint 공간의 차원이
$D=\sum_i d_i$이고 smoothness 이외의 구조가 없으면 정리 1에 의해
$N_{\mathrm{joint}}(\epsilon)=\Omega((L/\epsilon)^D)$가 필요하다. 특히
두 축의 차원이 모두 $d_0$이고 정확도 차수가 같다면 axis screening은
$O(\epsilon^{-d_0})$인 반면 직접 joint screening의 worst-case 하한은
$\Omega(\epsilon^{-2d_0})$다.

*증명 개요는 부록 A.3에 있다.* 이 명제는 **1단계 screening 비용**만
비교한다는 점에 주의해야 한다. 축 frontier product의 크기와
interaction이 남으므로, 총 비용 이점은
$\lambda_PN_{\mathrm{axis}}+\lambda_DN_{\mathrm{stage2}}
<\lambda_DN_{\mathrm{joint}}$인지로 검증해야 한다(비용 회계와 additive
BO [18]와의 관계는 부록 A.3).

#### 축별 후보를 결합해도 되는 조건

정리 3은 다음을 말한다. 임의의 joint 후보에서 W와 KV를 차례로 축별
frontier 점으로 바꾸면 비용은 늘지 않는다. 각 교체에서 AWQ 손실이
나빠질 수 있는 최대량이 각각 $V_W,V_{KV}$이므로, 두 번의 교체가 만드는
총 손실 증가는 그 합보다 클 수 없다.

**정리 3 (Front-product coverage).** 비용 함수가 식 (2)와 같이
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

쉽게 말해, 축별 frontier의 곱 안에는 임의의 원래 후보보다 비싸지 않으면서
AWQ 손실도 최대 $V_W+V_{KV}$만 나쁜 대체 후보가 존재한다.

*증명은 부록 A.4에 있다.*

정리 3은 개별 후보에 대한 점별 진술이다. 축별 탐색 설계가 실제로 필요로
하는 형태 — "$\epsilon$-frontier의 곱으로 제한한 뒤 joint 탐색을 해도
전체 joint 공간의 Pareto frontier를 놓치지 않는다" — 는 다음 따름정리다.

**따름정리 3.1 (전체 joint front의 $\epsilon$-지배).** 정리 3의 조건에서
$V=V_W(\eta_W)+V_{KV}(\eta_{KV})$라 하자. 전체 joint 공간의 Pareto front
$\operatorname{PF}_D(\mathcal{X})$의 모든 점 $a^{\star}$에 대해
$\mathcal{C}=\widehat{\mathcal{P}}_W\times\widehat{\mathcal{P}}_{KV}$ 안에
$c(b)\le c(a^{\star})$이고 $y_D(b)\le y_D(a^{\star})+V$인 $b$가 존재한다.
따라서 $\mathcal{C}$의 Pareto front는 손실 축을 $V$만큼 팽창시키는
additive $\epsilon$-dominance 의미에서 전체 공간의 **$V$-근사 Pareto
front**이며, 임의의 비용 상한 $\tau$에 대해

$$
\min_{b\in\mathcal{C},\,c(b)\le\tau} y_D(b)
\;\le\;
\min_{a\in\mathcal{X},\,c(a)\le\tau} y_D(a) + V
\tag{7'}
$$

이다. 1단계 점수가 오차 $|\hat z_i-z_i|\le\delta_z$로 관측되는 경우,
정확한 frontier 대신 $2\delta_z$-band
$\mathcal{B}_i=\{u:\hat z_i(u)\le\hat z_i^{\mathrm{front}}(c_i(u))+2\delta_z\}$의
곱을 쓰면 같은 결론이 slack
$V_W(\eta_W{+}2\delta_z)+V_{KV}(\eta_{KV}{+}2\delta_z)$로 복원된다.
*증명은 부록 A.4에 있다.*

이 따름정리를 읽을 때 세 가지를 정확히 해야 한다. 첫째, 보장은 **목적
공간의 $\epsilon$-지배**이지 결정 공간의 포함이 아니다. Joint-최적 구조
자체는 off-front 블록을 포함할 수 있어 $\mathcal{C}$ 밖에 있을 수 있고,
$\mathcal{C}$가 보장하는 것은 비용이 같거나 싸면서 손실이 최대 $V$ 나쁜
대체 구조의 존재다. 배포 목표가 손실--메모리 절충값이라면 이 약한 형태로
충분하지만, "최적 구조 자체를 찾는다"는 주장은 이 따름정리로 정당화되지
않는다. 둘째, 비용 방향은 단측이다. 양방향 메모리 band로 보고할 때
지배점이 band 하단 아래로 빠질 수 있다(3.10절). 셋째, slack
$V_W+V_{KV}$는 두 교체의 최악 경우를 각각 더한 union-bound형 상한이라
구조적으로 보수적이다. 두 최악이 같은 후보에서 동시에 실현될 필요가
없으므로 실현된 front 간 gap은 이 상한보다 작을 수 있으며, bound의 재료
$\widehat V$ 감사와 별도로 front-지배 자체를 직접 측정하는 프로토콜을
부록 C.10에 둔다.

**해석과 한계.** 이 정리는 가중치 손실과 KV 손실이 더해진다고 가정하지
않는다. 상호작용이 존재해도 축별 순서를 거의 뒤집지 않아 $V_i$가 작으면
충분하다. 하지만 이는 좋은 대체점이 product 안에 **존재한다**는 결과일
뿐, NSGA-III나 surrogate가 그 점을 반드시 찾는다는 수렴 정리는 아니다.
또한 돌연변이가 product를 포함한 채 후보를 추가하는 것은 안전하지만,
product 자체를 제거하는 hard restriction은 이 보장을 잃는다.

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
위반하였고, 최악의 경우도 넓은 영역의 체계적 교차가 아니라 공격적인
저비트 corner의 근접 동률이었다(최악 사례 세부는 부록 B.1).

정리 3에 직접 들어가는 측정 순서 위반 마진은

$$
\widehat V_W=0.0103,\qquad
\widehat V_{KV}=0.0034
$$

이며, 합은 0.0137로 관찰된 stride-128 JSD 범위
$0.5391-0.0138=0.5253$의 2.6%다. 식 (7)은 평균적인 순서가 아니라 가장
큰 유해 순위 역전에 의존하므로, 이 마진은 상관계수만 보고하는 것보다
정리에 직접 연결되는 근거다.

정리 3과 따름정리 3.1이 딛고 서는 근거는 위의 마진 감사이지 가산성이
아니다. 별도의 two-way additive 분해는 정리의 전제가 아니라 **왜 $V$가
작은지**에 대한 보조 증거로만 읽어야 한다: 주효과가 변동의
98.2%(가중치 74.5% + KV 23.7%)를 설명하지만, 잔차의 구조적 자기상관은
실재하는 상호작용을 가리킨다(세부는 부록 B.1). 따라서 이 결과는 “손실이
가산적이다”가 아니라, **상호작용은 존재하지만 검증한 frontier product
안에서는 축별 순서를 거의 뒤집지 않는다**는 정리 3의 더 약한 조건을
지지한다. 포화는 손실 차이를 압축하지만 순서를 재배열하지는 않는다는
것이 마진 감사의 요지이며, 순서를 뒤집는 성분만이 $V$에 들어간다.

![대응된 20×20 격자에서 순서와 값의 예비 분석.](../tests/docs/fig/fig6_order_vs_values.png)

그러나 400개 cell은 frontier 위의 블록만 분석하며, 한 번의 탐색으로
얻은 archive가 식 (5)를 만족하는지도 자동으로 보장되지 않는다. 따라서
현재 수치는 분석한 frontier product에 한정된 실증 결과로 해석하고, 더
넓은 주장에 필요한 off-front·frontier 충실도 반증 실험은 부록 C.6과
C.7에서 제안한다.

### 3.6 양자화 프록시는 어디까지 사용할 수 있는가?

동일한 비용 계산을 사용하는 프록시 HQQ와 배포 방법 AWQ의 손실을 각각
$y_P,y_D$라 하고, 각각의 Pareto 집합을 $\operatorname{PF}_P$와
$\operatorname{PF}_D$라 하자. AMQ의 전역 순서 동치 조건은 두 집합이
같아지는 충분조건이다 [4]. 여기서는 그 충분조건을 다시 제시하는 대신,
프록시 목적의 불일치가 탐색 예산으로 해결되지 않는 조건을 보인다.

핵심은 **평가 수 부족**과 **목적 불일치**를 구분하는 것이다. HQQ가 어떤
AWQ Pareto 후보 $x^\star$보다 다른 후보 $z$를 더 싸고 더 좋다고 판단하면
HQQ frontier는 $x^\star$를 버린다. 그런데 AWQ에서는 $x^\star$가 더
좋다면, HQQ 평가를 아무리 많이 추가해도 이 잘못된 판단 자체는 바뀌지
않는다.

**정리 4 (프록시 전용 탐색의 Pareto 비일관성).** 유한한
$\mathcal{X}$에서 어떤 $x^\star\in\operatorname{PF}_D$와
$z\in\mathcal{X}$가 다음을 만족한다고 하자.

$$
\begin{gathered}
c(z)\le c(x^\star),\qquad y_P(z)\le y_P(x^\star),\\
(c(z),y_P(z))\ne(c(x^\star),y_P(x^\star)),\qquad
y_D(z)>y_D(x^\star).
\end{gathered}
\tag{8}
$$

그러면 $x^\star\notin\operatorname{PF}_P$이다. 따라서 평가 횟수가
늘어날수록 프록시 Pareto 집합만 보존하는 탐색은
$\operatorname{PF}_D$ 전체로 수렴할 수 없다.

같은 문제를 고정된 배포 예산에서 regret으로 쓸 수 있다. 배포 예산
$B$를 만족하는 후보 집합을
$\mathcal{X}_B=\{x:c(x)\le B\}$, 프록시 최적점 집합을
$S_P(B)=\arg\min_{x\in\mathcal{X}_B}y_P(x)$라 하자. 다음의 배포 gap이

$$
\delta_D(B)=
\min_{x\in S_P(B)}y_D(x)
-\min_{x\in\mathcal{X}_B}y_D(x)>0
\tag{9}
$$

이면, HQQ 최적점을 정확히 찾더라도 AWQ 최적점보다 최소
$\delta_D(B)$만큼 나쁘다. 즉 HQQ 최적화 오차를 0으로 만들어도 이 AWQ
regret은 사라지지 않는다.

*증명은 부록 A.5에 있다.* 전제는 단순한 순위 역전보다 강하다. 역전된
점이 실제 Pareto 점을 프록시 목적과 비용에서 **지배**해야 frontier
누락의 증명서가 되며, 반대로 높은 전역 상관관계는 이 증명서가 없음을
보장하지 않는다(확장 해석은 부록 A.5).

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
(9)의 gap이 실제 선택 해상도에서 자주 양수임을 보여 준다.

![1단계 후보 공급과 2단계 최종 선택에서 요구되는 프록시 충실도의 차이.](../visualize/hqq_awq/fig/narr2_two_requirements.png)

Pareto 집합 자체의 비교도 같은 결론을 준다. 동일 pool에서 AWQ target
front에 대한 HQQ front의 recall은 71.9%이고, 제외된 648개 전부가 식
(8)의 지배 증명서를 측정 archive 안에서 갖는다. JSD $10^{-3}$ 동률
허용 시에도 recall 66.8%, Jaccard 0.496으로, 동일 손실에 잡음만 가한
null의 Jaccard $0.825\pm0.007$보다 훨씬 낮다. 즉 관찰된 front 차이는
측정 해상도로 설명되지 않는다(상세 수치와 범위 한정은 부록 B.2).

이에 따라 HQQ와 AWQ의 역할을 명확히 구분한다. HQQ는 (i) 두 축 탐색,
(ii) frontier 블록 선택, (iii) 저차원 표현 학습에 사용한다. AWQ label은
결합 surrogate 학습, 2단계 archive 갱신, 최종 구조 선택에 사용한다.
그렇다고 모든 결합 후보마다 별도의 AWQ 빌드가 필요한 것은 아니다.
소수의 AWQ 정보로 보정 모델을 학습할 수 있고, 3.8절의 방법은 한 번의
빌드를 여러 후보에 분산한다. 따라서 본 논문의 정확한 주장은 결합 공간
내부에서 **배포 방법의 정보**가 필요하다는 것이며, 배포 방법을 이용한
전수 탐색이 필요하다는 것이 아니다.

### 3.7 프록시 지도 surrogate 임베딩

현재 이산 genome은 352개 cell로 구성된다. 이 중 224개는 가중치 cell,
128개는 KV 비트/그룹 및 가지치기 cell이다. 정수형 ordinal encoding은
인위적인 선형 기하를 가정한다. 예를 들어 3비트에서 2비트로 낮출 때의
오차 증가가 4비트에서 3비트로 낮출 때와 같을 이유는 없다. One-hot
encoding은 이 문제를 제거하지만 가중치와 KV 표현을 각각 672차원과
896차원으로 확장한다. 두 방식 모두 약 100개 AWQ 빌드로 구성된 초기
설계에 비해 지나치게 고차원이다.

#### 정확한 개수 대신 조건부 학습률

“Surrogate에 최소 몇 개의 표본이 필요한가?”에는 가정 없는 하나의 답이
없으므로, 이론은 정확한 숫자 하나가 아니라 **차원과 목표 오차에 따라
표본 수가 어떻게 증가하는지**를 제시해야 한다(배경 논의는 부록 A.6).

아래 정리에서 기호의 의미는 다음과 같다.

- $N$: 독립적인 학습 단위 수; 운영 분석에서는 서로 다른 AWQ 가중치
  build 수로 보수적으로 센다
- $r$: surrogate가 실제로 사용하는 표현 차원
- $s$: 손실 함수의 매끄러움; 클수록 주변 표본에서 예측하기 쉽다
- $\sigma^2$: 측정 또는 표본 잡음의 크기

**정리 5 (비모수 surrogate의 조건부 학습률).** 고정된 표현
$\phi(a)\in[0,1]^r$에서 AWQ 손실이

$$
y_D(a)=f(\phi(a))+\xi,\qquad
\mathbb E[\xi\mid\phi]=0,\quad
\mathbb E[\xi^2\mid\phi]\le\sigma^2
$$

처럼 매끄러운 함수 $f$와 평균 0인 잡음 $\xi$로 표현된다고 하자. $N$개
학습 단위가 독립적으로 뽑히고 그 분포가 공간의 일부에만 몰리지 않으며
$f$의 smoothness가 $s$이면, 가장 좋은 비모수 추정기도 worst case에서
다음 차수의 평균 제곱오차를 갖는다.

$$
\operatorname{MSE}_{\mathrm{worst}}(N)
=\Theta\!\left(N^{-2s/(2s+r)}\right).
\tag{Sur-SC}
$$

따라서 목표 RMSE를 $\epsilon$으로 두면 필요한 표본 수의 차수는

$$
N_{\min}(\epsilon)
=\Theta\!\left(\epsilon^{-(2s+r)/s}\right).
$$

식 (Sur-SC)는 비모수 회귀의 알려진 minimax 학습률에서 따른다 [17].
읽을 때 중요한 부분은 $r$이 지수에 들어간다는 점이다. 같은 smoothness와
목표 오차에서는 표현 차원이 작을수록 필요한 표본이 빠르게 줄어든다.
이것이 PLS 차원 축소를 사용하는 이론적 동기다.

**이 정리가 말하지 않는 것.** 이 결과는 평균 예측오차에 관한 것이지
Pareto 순위 복원 정리가 아니다. 한 예산 구간의 최선과 차선의 차이를
$\Delta$라 하면, 모든 예측오차가 $\Delta/2$보다 작을 때 둘의 순서를
보존할 수 있다. 하지만 $\Delta$가 거의 0인 후보가 있으면 필요한 표본은
다시 커진다. 따라서 실제 최소 표본 수는 RMSE뿐 아니라 top-1 regret과
frontier recall로 정해야 한다.

선형 모형이나 GP처럼 더 강한 함수 구조 아래의 학습률과 그 주의점은
부록 A.6에 있다. 어느 경우에도 현재 18차원 PLS--GP에 필요한 AWQ build
수를 이론만으로 특정할 수는 없다.

#### 왜 낮은 표현 차원이 존재하는가: 근사 충분성

정리 5는 표현 차원 $r$이 작을수록 필요한 표본이 줄어든다고 말하지만,
낮은 차원의 좋은 표현이 존재한다는 것은 별도의 주장이다. 그 근거는
3.5절의 실증이 지지하는 근사 충분성이다. Joint 손실이 두 축의 스칼라
점수의 단조 함수로 $\varepsilon_0$ 이내 근사된다면, 2단계가 학습해야
하는 대상은 352개 cell의 함수가 아니라 사실상 2변수 함수다.

**명제 2 (근사 충분성에 의한 2단계 차원 축소).** 모든 $a$에 대해
$|y_D(a)-F(z_W(a_W),z_{KV}(a_{KV}))|\le\varepsilon_0$이고, $F$는
단조이며 관련 범위에서 $L_F$-Lipschitz라고 하자. 2단계 후보가
$|\hat z_i-z_i|\le\varepsilon_z$인 표현 $\hat z$를 가지면, $y_D$를 오차
$\epsilon$으로 추정하는 문제는 2변수 단조 함수 $F$를 오차
$\epsilon-\varepsilon_0-L_F\varepsilon_z$로 추정하는 문제로 축소된다.
즉 2단계의 표본 요구량은 정리 5에서 $r=2$(+정확한 비용 좌표)인 경우의
차수를 따르고, $\varepsilon_0+L_F\varepsilon_z$가 줄일 수 없는 오차
floor로 남는다.

*증명 개요.* 삼각 부등식으로 오차를 분해하면 된다. $\square$

측정된 근사 충분성 잔차는 matched-$z$ 쌍 113개에서 중앙값 0.0065, 최대
0.034다. 이 명제의 실무적 무게는 $\varepsilon_z$ 가정에 있다. 새로
생성된 후보의 $z$는 관측되지 않으므로 $z$를 예측하는 표현이 필요한데,
이것이 아래 식 (10) PLS 임베딩의 이론적 지위다. PLS는 임의의 차원 축소
기법이 아니라 1단계 archive로 학습한 근사 충분 통계 $z$의 추정기이며,
$\varepsilon_z$는 1단계 archive의 held-out으로 실측할 수 있다. 이
명제의 직접 검증 — surrogate 입력을
$(\hat z_W,\hat z_{KV},c_W,c_{KV})$ 4차원으로 제한하는 ablation — 은
부록 C.2에 포함한다.

#### PLS는 예측 지도이지 탐색공간의 경계가 아니다

적은 수의 좌표로 **손실을 예측**할 수 있다는 것과, 그 좌표만으로 좋은
범주형 구조를 **빠짐없이 생성**할 수 있다는 것은 다르다. 저차원성
감사에서 비용 고정 세부 배치를 95% 설명하는 one-hot rank는 W 225, KV
150이었고 rank-8 PCA의 Pareto-front 구조 완전 복원율은 두 축 모두
0%였다(부록 B.3). 따라서 PLS는 surrogate 입력으로는 유용하지만 2단계
탐색을 latent 후보로 제한하는 근거는 아니며, ActQuant는 PLS 좌표를
예측에 사용하면서 원래 범주 공간의 block 조합과 sparse mutation을
유지한다.

대규모 1단계 HQQ archive를 이용해 지도 표현을 학습한다. $h_W(w)$와
$h_{KV}(k)$를 cell별 one-hot encoding이라 하자. 각 축에서 PLS는 1단계
JSD의 제곱근과 공분산이 큰 좌표를 찾는다.

$$
\begin{aligned}
R_W &= \operatorname{PLS}_8\!\left(h_W(w),\sqrt{z_W(w)}\right),\\
R_{KV} &= \operatorname{PLS}_8\!\left(h_{KV}(k),\sqrt{z_{KV}(k)}\right)
\end{aligned}
\tag{10}
$$

결합 surrogate의 입력은 다음과 같다.

$$
\phi(w,k)=\left[
R_W^\top h_W(w),
R_{KV}^\top h_{KV}(k),
c_W(w),c_{KV}(k)
\right]\in\mathbb{R}^{18}.
\tag{11}
$$

Matérn-$3/2$ kernel을 사용하는 ARD Gaussian process가 $\phi$에서
$\sqrt{y_D}$를 예측한다. HQQ는 표현을 학습하는 감독 정보를 제공하지만,
regression head와 최종 선택에 사용되는 모든 목적값은 AWQ label에서
학습한다.

현재 기본값은 축별 8개 성분이다. 최신 감사에서 시간 순서로 분할한
PLS@8의 전역 프록시 $R^2$는 W 0.999, KV 0.996이지만, budget으로 설명되는
효과를 제거한 $R^2$는 각각 0.426과 0.810으로 낮아졌다. 2단계 AWQ
archive에서도 보지 못한 W로 전이하는 group split과 미래 iteration을
예측하는 temporal split에서는 PLS가 같은 차원의 PCA보다 일관되게 좋지
않았다. 따라서 현재 8성분은 검증된 최적값이 아니라 동작 기본값이다.

표현 선택과 최소 표본 수에 대한 결론은 유보한다. 소수의 학습 크기
비교로는 학습곡선의 지수와 floor를 식별할 수 없고, 한 AWQ 가중치
build를 공유하는 KV companion은 독립 표본이 아니므로 총 architecture
수를 그대로 $N$으로 세면 과도하게 낙관적이기 때문이다. 운영 표본 수는
서로 다른 가중치 family 수로 세고, raw ordinal / one-hot / PCA / 지도
PLS / self-PLS를 가중치-family 단위 split에서 비교하는 ablation 설계는
부록 C.2에 있다. 이 설계가 검증하는 실제 주장은 **AWQ build 하나당 더
유용한 의사결정을 만드는가**이다.

### 3.8 Multi-KV 구성 평가

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

Multi-KV 평가는 한 번의 빌드에서 label 수를 늘리지만 **군집된 data**를
만들므로, “평가한 architecture 수”만 보고하면 실질적인 독립 표본 수를
과장하게 된다. 본 논문에서는 $K$를 **가중치 빌드당 전체 구성 수**로
정의한다($K=1$은 companion 없음). 현재 구현은 anchor 하나에 companion
10개, 즉 $K=11$이며, 고정 빌드/label/실행시간 세 예산 관점의 ablation이
완료되기 전까지 이를 최적값이 아닌 동작 기본값으로 취급한다(설계와
측정 지표는 부록 C.3).

### 3.9 가정 검증 및 반증 프로토콜 (요약)

앞의 이론은 조건부 주장이므로, 실험은 전역 상관계수를 나열하기보다 각
이론 가정을 직접 검증하는 형태로 구성해야 한다. 다음 프로토콜의 전체
설계는 부록 C에 있다.

- **차원 scaling과 direct--axis 비교** (정리 1, 명제 1; 부록 C.1):
  nested search space에서 목표 오차 도달에 필요한 build 수의 scaling을
  회귀하고, interaction 강도 $\lambda$를 조작한 replay stress test를
  수행한다.
- **Surrogate 운영 최소 표본 수** (정리 5, 명제 2; 부록 C.2): top-1
  regret, target-front recall, hypervolume gap 기준을 동시에 만족하는
  최소 가중치-build 수 $N^\star$를 grouped held-out 학습곡선으로 정한다.
  표현(raw/one-hot/PCA/PLS/z-충분성) ablation을 포함한다.
- **Multi-KV companion ablation** (부록 C.3): 고정 빌드/label/실행시간
  예산에서 $K$를 비교한다.
- **대응된 $20\times20$ 순위 격자** (정리 3; 부록 C.4): 최악 위반 마진
  $\widehat V_W,\widehat V_{KV}$를 갱신한다.
- **Front-지배 직접 검증** (따름정리 3.1; 부록 C.10): 전량 측정 pool에서
  전체 front 대비 product front의 실현 $\epsilon$-gap을 직접 재고, bound
  대비 실현 slack, band폭 스윕, 양방향 band 손실률을 보고한다.
- **축 순위 식별 대결** (정리 2; 부록 C.5): 매칭 표본 대 비매칭+보정
  추정의 순위 복원을 corner 포함 조건에서 대조한다.
- **Off-front 교체 경로** (정리 3; 부록 C.6): 돌연변이 구조 30개 이상의
  순차 교체를 직접 측정한다.
- **1단계 frontier 충실도** (식 (5); 부록 C.7): seed 반복으로
  $\eta$-cover 여부를 추정한다.
- **프록시 적용 범위** (정리 4; 부록 C.8): 네 해상도의 일치도, 지배
  증명서 수, 예산별 regret gap을 보고한다.
- **일반화** (부록 C.9): 다른 모델과 장문 벤치마크에서 전체 분석을
  반복한다.

### 3.10 현재 이론적 보장의 한계

각 결과의 적용 범위를 요약한다(전문은 부록 D). 정리 1은 smoothness
이외의 구조가 없는 함수족에 대한 minimax 하한이며, 현재 genome의 metric
dimension이 raw cell 수와 같다고 증명하지 않는다. 명제 1의 지수 이점은
유효한 $\eta$-cover를 반환한다는 조건 아래 **screening**에만 적용되고,
종단간 비용이 항상 더 작다는 정리가 아니다. 정리 2는 축 순위 **식별**의
결과이지 joint frontier **발견**의 하한이 아니며, 가산 보정이 공격적
corner에서 실패한다는 측정과 결합될 때에만 축별 설계의 필연성이
성립한다. 정리 3은 존재성 정리이며 탐색 알고리즘의 수렴 정리가 아니다.
종단간 regret에는 1단계 frontier 오차, 순서 위반 마진, 2단계 탐색 오차,
측정 잡음이 추가되며 분리해서 보고해야 한다. 정리 4는 식 (8)의 지배
증명서 또는 식 (9)의 양의 gap이 있을 때 프록시-only 수렴이 실패한다는
조건부 결과이고, 현재 증명서는 AWQ 탐색으로 수집된 archive에 대한
것이다. 정리 5는 현재 18차원 PLS--GP의 정확한 label 수를 주지 않으며,
실제 최소 build 수는 부록 C.2의 grouped held-out 학습곡선으로 결정해야
한다. Strided JSD의 상관관계도 독립적인 모델과 구성 pool에서 재현되어야
하며, 현재 근거는 하나의 모델과 하나의 compression family에 집중되어
있다. 마지막으로 정리 3의 결론은 비용 상한 제약에 적용된다. 양방향
메모리 band 보고 구간에서는 더 비싸지 않은 근사 대체점의 존재만
보장하고, 그 대체점이 같은 구간 안에 남는다고 보장하지는 않는다.
따름정리 3.1도 같은 스코프를 공유한다: 목적 공간의 $\epsilon$-지배이지
결정 공간의 Pareto 집합 포함이 아니고, $V$는 감사된 쌍 위의 empirical
sup이며, 존재 보장이지 2단계 탐색이 그 대체점을 찾는다는 보장이 아니다.

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
    <https://aclanthology.org/2024.acl-long.172/>
15. Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, et al. “RULER: What's the Real
    Context Size of Your Long-Context Language Models?” COLM, 2024.
    <https://arxiv.org/abs/2404.06654>
16. Cédric Malherbe and Nicolas Vayatis. “Global Optimization of Lipschitz
    Functions.” ICML, 2017.
    <https://proceedings.mlr.press/v70/malherbe17a.html>
17. Charles J. Stone. “Optimal Global Rates of Convergence for Nonparametric
    Regression.” *The Annals of Statistics*, 1982.
    <https://doi.org/10.1214/aos/1176345969>
18. Kirthevasan Kandasamy, Jeff Schneider, and Barnabas Poczos. “High
    Dimensional Bayesian Optimisation and Bandits via Additive Models.” ICML,
    2015. <https://proceedings.mlr.press/v37/kandasamy15.html>
19. Ziyu Wang, Frank Hutter, Masrour Zoghi, David Matheson, and Nando de
    Freitas. “Bayesian Optimization in a Billion Dimensions via Random
    Embeddings.” *Journal of Artificial Intelligence Research*, 2016.
    <https://doi.org/10.1613/jair.4806>
20. Niranjan Srinivas, Andreas Krause, Sham M. Kakade, and Matthias Seeger.
    “Gaussian Process Optimization in the Bandit Setting: No Regret and
    Experimental Design.” ICML, 2010.
    <https://icml.cc/Conferences/2010/papers/422.pdf>
21. Sattar Vakili, Nacime Bouziani, Sepehr Jalali, Alberto Bernacchia, and
    Da-shan Shiu. “Optimal Order Simple Regret for Gaussian Process Bandits.”
    NeurIPS, 2021.
    <https://proceedings.neurips.cc/paper_files/paper/2021/hash/b1300291698eadedb559786c809cc592-Abstract.html>
22. Marcela Zuluaga, Andreas Krause, and Markus Püschel. “ε-PAL: An Active
    Learning Approach to the Multi-Objective Optimization Problem.” *Journal
    of Machine Learning Research*, 2016.
    <https://jmlr.csail.mit.edu/papers/v17/15-047.html>
