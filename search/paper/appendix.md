# 부록 (Appendix)

본 부록은 `main.md`에서 옮겨 온 형식적 진술과 증명, 확장 실증 세부,
가정 검증 및 반증 프로토콜, ablation 설계, 이론적 한계 전문을 담는다.
절 번호와 식 번호((5), (6), (8), (9) 등), 참고문헌 번호 [n]은 모두
`main.md`를 기준으로 한다.

## 부록 A. 형식적 진술과 증명

### A.1 정리 1 (차원의 저주 하한): packing 형식화와 증명

본문 3.3절의 직관을 연속 공간과 이산 공간에 함께 적용하기 위해
**구분 가능한 영역의 수**를 사용한다. 거리 $r$ 이상 떨어진 후보를 최대 몇
개까지 고를 수 있는지를 packing number $\mathcal M(\mathcal Z,r)$라 한다.
분석할 거리 범위 $r\in(r_{\min},r_{\max}]$에서

$$
\mathcal M(\mathcal Z,r)\ge(a/r)^d
$$

이면, 그 범위에서 공간이 적어도 $d$차원처럼 커진다고 해석한다. 여기서
$a$는 거리의 단위를 반영하는 상수이고, $d$가 중요한 차원 항이다. 연속
cube에서는 통상적인 차원과 일치한다. 유한한 범주형 공간에서는 후보 수가
결국 $|\mathcal Z|$에서 포화되므로, 이 해석은 실제 탐색이 구분하려는
거리 범위 안에서만 사용한다.

**정리 1 (차원의 저주에 의한 평가 하한, 형식적 진술).** $\mathcal{F}_L$을
$(\mathcal{Z},\rho)$ 위의 모든 $L$-Lipschitz 손실 함수 집합이라 하자.
어떤 적응적 알고리즘이 잡음 없는 $N$개 label을 관측하고 관측점 중 하나를
$\widehat z_N$으로 반환한다고 하자. 위 packing 조건이
$r_N=a(4N)^{-1/d}\in(r_{\min},r_{\max}]$에서 성립하면, 상수 $c>0$에 대해

$$
\inf_{A_N}\sup_{f\in\mathcal{F}_L}
\mathbb{E}\!\left[f(\widehat z_N)-\min_{z\in\mathcal{Z}}f(z)\right]
\ge cLaN^{-1/d}.
\tag{CoD}
$$

왼쪽은 “가능한 알고리즘 중 가장 좋은 것을 쓰더라도, 가장 불리한
Lipschitz 손실에서는 남는 평균 regret”을 뜻한다. 오른쪽의
$N^{-1/d}$가 차원이 커질수록 학습이 느려지는 핵심 항이다.

이를 평가 수에 대한 형태로 바꾸면, $\epsilon/L$이 같은 거리 범위에 있는
동안 worst-case simple regret을 $\epsilon$ 이하로 만들기 위해

$$
N_{\mathrm{direct}}(\epsilon)
=\Omega\!\left((La/\epsilon)^d\right)
$$

개의 평가가 필요하다. 즉 목표 오차를 절반으로 줄이거나 차원을 하나
늘리는 비용이 고차원에서는 빠르게 커진다 [16].

*증명 개요.* 서로 멀리 떨어진 영역을 $N$개보다 많이 만든 뒤, 그중 한
영역에만 낮은 최솟값을 숨긴 여러 Lipschitz 함수를 생각한다. 알고리즘은
$N$회 평가로 모든 영역을 볼 수 없으므로, 최솟값이 미관측 영역에 있는
경우를 구별하지 못한다. 영역의 반지름을
$r\asymp aN^{-1/d}$로 잡으면 놓칠 수 있는 손실이 $Lr$이 되어 식
(CoD)를 얻는다. 무작위 알고리즘에도 같은 결론이 성립하도록 하는 표준
minimax 논증을 적용하면 알려진 Lipschitz 전역 최적화 하한과 일치한다
[16]. $\square$

**해석과 한계.** 이 정리에 현재 genome의 352개 cell을 그대로
$d=352$로 대입해서는 안 된다. 비슷하게 움직이는 레이어, 비용 제약, PLS,
ARD kernel은 예측에 필요한 차원을 낮출 수 있다. 반대로 공간이 유한하다는
사실만으로 탐색이 쉬워지는 것도 아니다. 매우 작은 오차를 요구하면 결국
최대 $|\mathcal Z|$개 후보를 구별하는 문제가 된다. 따라서 정리 1은
**구조를 활용하지 않는 직접 joint 탐색의 worst case**를 설명한다. 현재
문제도 이 비율을 따른다는 주장은 부록 C.1의 차원별 실험으로 별도
검증한다.

### A.2 정리 2 (축 순위의 매칭 설계 식별): 증명 개요와 한계

*증명 개요.* (i) 파트너가 전부 다르면 각 관측은 자신의 합성값 하나를
통해서만 $(F,z)$를 제약한다. 파트너별 단조 재매개화를 $F$에 합성하면
모든 관측값을 보존하면서 $z_W(u)$와 $z_W(u')$의 순서를 원하는 대로 바꿀
수 있다. (ii) 기댓값의 선형성과 생일 문제의 표준 하한. (iii) 정의에서
직접 따른다. $\square$

**한계.** 정리 2는 축 순위 **식별**의 결과이지 joint frontier **발견**의
하한이 아니다. 가산성을 가정한 joint 회귀는 배포 가능한 대역 안에서
순위를 상당히 복원할 수 있고, 이는 본문 3.5절 말미의 additive 분해
실증과 일치한다. 정리 2의 힘은 그 보정이 정확히 선택 압력이 집중되는
공격적 corner에서 실패한다는 측정과 결합될 때 나온다. 같은 예산의 매칭
표본과 비매칭 표본으로 축 순위를 복원해 대조하는 반증 실험은 부록
C.5에 포함한다.

### A.3 명제 1 (축별 Pareto screening의 조건부 이점): 증명 개요와 비용 회계

*증명 개요.* 축 $i$에서 대표점 사이의 거리를
$r_i=\eta_i/(2L_i)$로 두면, 가장 가까운 대표점으로 추정한 손실 오차가
최대 $\eta_i/2$다. 따라서 추정 Pareto 점을 남기면 임의의 원래 후보보다
비용이 높지 않고 실제 축별 손실도 최대 $\eta_i$만 나쁜 대체점을 얻는다.
축마다 필요한 대표점 수를 더하면 식 (Axis-SC)가 된다. Joint 공간을
직접 덮을 때는 축별 대표점의 모든 조합이 필요하므로 차원이 합쳐지고,
정리 1의 지수 하한이 적용된다. $\square$

**비용 회계.** 이 명제는 **1단계 screening 비용**만 비교한다. 축
frontier의 Cartesian product 크기
$|\widehat{\mathcal P}_W||\widehat{\mathcal P}_{KV}|$가 여전히
클 수 있고, interaction이 강하면 $V_i$도 커질 수 있다. 따라서 총 비용
이점은 proxy label 비용을 $\lambda_P$, AWQ label 비용을 $\lambda_D$라 할
때 실제로
$\lambda_PN_{\mathrm{axis}}+\lambda_DN_{\mathrm{stage2}}
<\lambda_DN_{\mathrm{joint}}$인지로 검증해야 한다. Additive BO가 낮은
component 차원에서 전체 차원 의존성을 완화하는 결과 [18]와 방향은
같지만, 본 연구는 joint AWQ loss의 가산성을 요구하지 않고 조건부 순서
위반 마진을 지불한다.

### A.4 정리 3 (front-product coverage): 증명

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

**따름정리 3.1의 증명.** $a^{\star}\in\operatorname{PF}_D(\mathcal{X})$는
$\mathcal{X}$의 원소이므로 정리 3을 그대로 적용하면 점별 결론을 얻고,
예산 형태 (7')는 $c(b)\le c(a^{\star})\le\tau$에서 따른다. $\mathcal{C}$의
front가 $V$-근사 Pareto front라는 것은 additive $\epsilon$-dominance의
정의를 손실 좌표에만 적용한 것이다. Band 버전: $|\hat z_i-z_i|\le
\delta_z$이면 참 축-frontier 점 $p_i$는
$\hat z_i(p_i)\le z_i(p_i)+\delta_z\le\hat z_i^{\mathrm{front}}(c_i(p_i))
+2\delta_z$를 만족해 $\mathcal{B}_i$에 포함된다. 즉 $\mathcal{B}_i$는 참
점수 기준 $(\eta_i+2\delta_z)$-cover이고, 식 (6)의 마진 조건을 같은
인자로 완화하면 정리 3의 두 교체 논증이 문자 그대로 통과한다. $\square$

**한계 주석.** 이 증명은 세 가지를 주장하지 않는다. (i) 결정 공간의
포함 — joint-최적 구조 자체는 $\mathcal{C}$ 밖일 수 있다. (ii) 양방향
band 안정성 — 지배점은 band 하단 아래로 빠질 수 있다. (iii) $V$의
전역성 — $\widehat V$는 감사된 쌍(현재 front-20 블록의 곱)의 empirical
sup이므로, off-front 역할을 포함하는 C.6의 사슬 감사 없이는 결론을
감사된 곱 밖으로 인스턴스화할 수 없다.

### A.5 정리 4 (프록시 전용 탐색의 Pareto 비일관성): 증명과 확장 해석

*증명.* 식 (8)의 앞 세 조건에 의해 $z$가 프록시 목적과 비용에서
$x^\star$를 지배하므로 $x^\star$는 $\operatorname{PF}_P$에 속할 수
없다. 그러나 마지막 부등식 때문에 이 지배 관계는 배포 목적에서
보존되지 않는다. 따라서 프록시 frontier에 수렴하는 archive는
$x^\star$를 제외한다. 식 (9)는 HQQ 최적점들 중 AWQ에서 가장 좋은 점과
실제 AWQ 최적점의 차이를 그대로 정의했으므로 두 번째 결론이 따른다.
$\square$

**해석과 한계.** 정리의 전제는 단순히 “순위 역전이 하나 있다”보다
강하다. 역전된 점이 실제 Pareto 점을 프록시 목적과 비용에서 지배해야
frontier 누락의 증명서가 된다. 반대로 높은 전역 상관관계는 이 증명서가
없음을 보장하지 않는다. 예산 간 큰 비트 차이가 전역 상관관계를 지배하는
반면, 최종 선택은 거의 같은 비용을 서로 다른 레이어에 배치한 후보의
순위로 결정되기 때문이다.

### A.6 정리 5 (조건부 학습률): 배경과 대안 함수족

“Surrogate에 최소 몇 개의 표본이 필요한가?”에는 가정 없는 하나의 답이
없다. 같은 18차원이라도 손실 함수가 매끄럽고 잡음이 작으면 적은 표본으로
학습할 수 있지만, 관측하지 않은 지점에서 손실이 임의로 바뀔 수 있다면
전체 공간을 보기 전에는 일반화를 보장할 수 없다. 따라서 이론은 정확한
숫자 하나가 아니라 **차원과 목표 오차에 따라 표본 수가 어떻게 증가하는지**를
제시해야 한다. 식 (Sur-SC)는 이 요구에 대한 비모수 worst-case 답이다.

함수 구조를 더 강하게 가정하면 학습률은 달라진다. 실제 관계가
$p$차원 선형 모형이면 RMSE $\epsilon$에 필요한 표본은 대략
$\sigma^2p/\epsilon^2$에 비례할 수 있다. 다만 $N\ge p$는 계수를
계산하기 위한 최소 조건일 뿐 정확도 보장이 아니다. GP에서도 표본 수는
좌표 수 하나가 아니라 kernel이 표현할 수 있는 함수의 복잡도
$\gamma_N$에 의해 결정되며, simple regret은 대략
$\sqrt{\gamma_N/N}$ 차수다 [20, 21]. 어느 경우에도 현재 18차원
PLS--GP에 필요한 AWQ build 수를 이론만으로 특정할 수는 없다.

## 부록 B. 추가 실증 세부

### B.1 대응 $20\times20$ 격자: 최악 사례와 additive 분해 세부

주 분석인 stride-128에서는 7,600개 비교 중 20개(0.26%)만 순서를
위반하였다. 최악의 경우도 넓은 영역의 체계적 교차가 아니었다. 가중치
2.25비트와 2.40비트 partner가 만든 KV 순위를 비교했을 때 190개 KV 쌍
중 5개가 불일치했고, 해당 AWQ JSD 차이의 중앙값과 최댓값은 각각
0.0010과 0.0020이었다. 즉 최소 $\tau=0.942$는 주로 공격적인 저비트
corner의 근접 동률에서 발생한다.

별도의 two-way additive 분해에서는 가중치 주효과가 전체 변동의 74.5%,
KV 주효과가 23.7%, interaction과 측정 잔차를 합친 항이 1.73%를
차지하였다. 반복 측정이 없는 격자이므로 마지막 항을 순수한 interaction
분산으로 해석할 수는 없다. 다만 잔차의 인접 cell 자기상관이 0.96이고
stride-32 잔차와도 0.62의 상관을 보여 구조화된 상호작용이 존재한다.
따라서 이 결과는 “손실이 가산적이다”가 아니라, **상호작용은 존재하지만
검증한 frontier product 안에서는 축별 순서를 거의 뒤집지 않는다**는
정리 3의 더 약한 조건을 지지한다.

### B.2 HQQ--AWQ Pareto 집합 중첩 감사

Pareto 집합 자체도 같은 후보 pool에서 직접 비교하였다. 정확한 지배를
사용하면 AWQ target front 2,304개 중 1,656개만 HQQ proxy front와
겹쳤다. Target-front recall은 71.9%, Jaccard overlap은 0.555이며,
제외된 648개 모두가 식 (8)의 엄격한 프록시 지배 증명서를 측정 archive
안에서 갖는다. JSD 차이 $10^{-3}$을 동률로 처리해도 target front
1,654개와 proxy front 1,677개의 교집합은 1,105개에 그쳤다(recall 66.8%,
Jaccard 0.496). 동일한 AWQ loss에 $[-10^{-3},10^{-3}]$ 잡음만 가한 5회
null의 Jaccard는 $0.825\pm0.007$이었다. 관찰된 front 차이는 단순 측정
해상도에서 예상되는 membership churn보다 훨씬 크다. 다만 이는 측정된
production archive에 대한 실증 증명서이지, 아직 평가하지 않은 전체
조합 공간의 front 크기나 recall을 추정한 것은 아니다.

### B.3 저차원성 감사: 예측 지도 대 구조 생성 공간

여기서는 두 종류의 저차원성을 구분해야 한다. 적은 수의 좌표로 **손실을
예측**할 수 있다는 것과, 그 좌표만으로 좋은 범주형 구조를 **빠짐없이
생성**할 수 있다는 것은 다르다. 최신 저차원성 감사에서 전역 변화는 몇
개의 큰 방향으로 요약되었지만, 비용을 고정한 세부 배치를 95% 설명하는
데 필요한 one-hot rank는 W 225, KV 150이었다. 또한 rank-8 PCA의
Pareto-front 구조 완전 복원율은 두 축 모두 0%였다. 반면 같은 크기의
지도 좌표는 전역 손실 예측에는 유용했다. 즉 낮은 차원의 **예측기**와
낮은 차원의 **구조 생성 공간**을 동일시할 수 없다.

따라서 PLS는 “어떤 후보를 먼저 평가할지” 정하는 surrogate 입력으로는
유용하지만, 2단계 탐색을 latent decoder가 생성한 후보로만 제한하는
근거는 아니다. ActQuant는 PLS 좌표를 예측에 사용하면서 원래 범주 공간의
block 조합과 sparse mutation을 유지한다. 이는 드문 중요 cell 선택을
놓쳤을 때 원래 공간으로 탈출하는 경로다.

## 부록 C. 가정 검증 및 반증 프로토콜

본문의 이론은 조건부 주장이다. 따라서 실험은 전역 상관계수를 나열하기보다
각 이론 가정을 직접 검증하는 형태로 구성해야 한다. 최소한 다음 분석이
필요하다.

### C.1 차원 scaling과 direct--axis 비교

정리 1과 명제 1의 scaling이 현재 문제에서도 관찰되는지 평가하려면
하나의 최종 차원에서 두 방법을 한 번 비교하는 것으로는 부족하다.
활성화할 레이어 또는 module 위치 수를
점진적으로 늘린 nested search space를 구성한다. 예를 들어 가중치
위치는 $16,32,64,128,224$개, KV 위치는 $8,16,32,64$개를 활성화하고
나머지는 동일한 기준 구성으로 고정한다. 각 공간에서 다음 네 방법을
같은 AWQ build 예산으로 비교한다.

1. 전체 joint 공간의 random 또는 NSGA-III 탐색
2. 전체 joint 공간의 동일 surrogate-assisted 탐색
3. 같은 1단계 예산으로 얻은 축별 empirical frontier product만 사용하는
   2단계 탐색
4. $\epsilon$-frontier product와 제한된 mutation을 사용하는 전체 방법

Surrogate head, 초기화 후보의 비용 분포, acquisition batch, 최종 후보
pool을 고정해야 한다. 축별 방법에 사용된 HQQ 평가도 wall-clock으로
환산해 총 비용에 포함한다. $N\in\{25,50,100,200,400\}$개의 서로 다른
AWQ 가중치 build에서 reference front 대비 hypervolume gap, IGD,
target-front recall, 예산별 top-1 regret을 보고한다. 각 목표 오차
$\epsilon$에 도달하는 최소 build 수 $N_\epsilon$을 구하고
$\log N_\epsilon$을 차원에 대해 회귀하면 단순한 최종 성능 비교보다
정리 1이 예측하는 scaling pattern을 직접 시험할 수 있다. 이때 활성
position 수를 nominal dimension으로 보고하는 동시에, 후보 pool의 Hamming
metric에서 relevant radius별 packing 수의 log--log slope를 empirical
metric dimension으로 추정한다. Kernel effective rank는 surrogate 관점의
보조 지표로 분리해 보고하며 packing dimension과 동일시하지 않는다.

유한한 공통 후보 pool을 HQQ로 전부 평가할 수 있는 축소 공간에서는 그
pool의 oracle axis frontier도 추가한다. 이는 실현 가능한 방법과 섞어
평균내는 baseline이 아니라, 1단계 추정오차와 2단계 탐색오차를 분리하는
offline upper bound로만 사용한다.

이 실험에는 두 가지 통제가 필요하다. 첫째, 현재 production archive는
이미 ActQuant가 선택한 표본이므로 direct 방법에 불리한 reference가 될
수 있다. 비교용 hold-out pool은 HQQ 비용 층과 구조적 거리를 기준으로
독립 표본화하고 모든 방법이 같은 pool에 접근하게 해야 한다. 둘째,
axis-first 방법은 interaction이 커지면 실패해야 정상이다. 대응된
$20\times20$ 격자의 two-way residual $g(w,k)$를 사용해
$y_\lambda=\mu+\alpha(w)+\beta(k)+\lambda g(w,k)$인 replay benchmark를
만들고 $\lambda\in\{0,0.5,1,2,4\}$를 변화시킨다. $\lambda$가 커질 때
$\widehat V_W+\widehat V_{KV}$와 axis-first regret이 함께 증가하는지를
보이면, 이득이 어떤 조건에서 사라지는지도 제시할 수 있다. 이는 실제
새 측정을 대체하는 증거가 아니라 interaction 강도에 대한 controlled
stress test다.

### C.2 Surrogate의 운영 최소 표본 수와 표현 ablation

정리 5의 $N_{\min}$을 현재 시스템의 숫자로 바꾸려면 먼저 성공 기준을
정해야 한다. 본 연구에서는 ``예측 RMSE가 작다'' 대신 다음 세 조건을
동시에 만족하는 최소 **서로 다른 AWQ 가중치 build 수**를 운영
$N^\star$로 정의한다.

- held-out 예산 band의 robust loss range로 정규화한 90분위 top-1
  regret이 2% 이하
- tolerance를 고정한 target-front recall이 90% 이상
- pooled reference 대비 hypervolume gap이 1% 이하

$N\in\{25,50,75,100,150,200,300,430,600\}$에서 최소 10개의 subsample
seed를 사용하고, 동일 가중치 family의 KV companion은 반드시 같은
train/test fold에 둔다. Raw genome, one-hot, PCA, PLS 차원
$r\in\{4,8,16,32\}$, cost-only negative control, 그리고 명제 2를 직접
검증하는 4차원 z-충분성 입력 $(\hat z_W,\hat z_{KV},c_W,c_{KV})$을 동일
GP head로 비교한다. 각 지표에 power-law-plus-floor 학습곡선을 적합하되, 최종
$N^\star$는 적합곡선의 점추정이 아니라 두 개 연속 $N$에서 bootstrap
95% 상한이 위 기준을 통과하는 최초 지점으로 정한다. 이를 통해 우연한
한 split의 조기 통과를 방지한다.

2%, 90%, 1%는 이 초안의 운영 기준 제안이며 이론에서 도출된 상수가
아니다. 실제 배포 허용오차가 정해져 있다면 데이터를 보기 전에 그 값으로
대체하고, threshold sensitivity도 함께 보고해야 한다.

추가로 PLS 차원별 kernel matrix의 effective rank, 학습된 ARD
lengthscale, $\gamma_N$의 empirical log-determinant 근사치를 보고한다.
차원 축소가 단순히 train error를 낮춘 것인지 실제 information gain과
필요 build 수를 줄였는지 구분하기 위해서다. 기존의 제한된 학습 크기
비교는 이 실험의 사전 관측일 뿐 최소 표본 수의 추정치로 사용하지 않는다.

**표현 ablation 설계 (본문 3.7절).** 기존의 소수 학습 크기 비교만으로는
학습곡선의 지수, 오차 floor, 또는 최소 표본 수를 식별할 수 없다. 또한
하나의 AWQ 가중치 build에서 얻은 여러 KV companion은 독립 표본이
아니므로 총 architecture 수를 그대로 $N$으로 세면 과도하게 낙관적이다.
운영 표본 수는 우선 서로 다른 가중치 family 수로 세고, companion은
동일 family 내부의 조건부 관측으로 별도 보고한다.

따라서 향후 ablation은 같은 head와 고정된 hold-out 후보에서 raw ordinal,
one-hot, 총 차원 $r\in\{4,8,16,32\}$인 PCA, 축별 성분 수
$d\in\{2,4,8,16\}$인 HQQ 지도 PLS, AWQ archive만으로 학습한 self-PLS를
비교한다. 서로 다른 AWQ 가중치 build 수
$N\in\{25,50,75,100,150,200,300,430,600\}$을 최소 10개 subsample
seed에서 비교한다. 하나의 AWQ 빌드를 공유하는 KV companion이 train과
test에 동시에 들어가 정보가 누출되지 않도록 가중치 할당 단위로
split한다. 전역 및 동일 예산 Spearman 상관계수, RMSE, 예산별 top-1
regret, Pareto hypervolume을 보고한다. 이 설계는 단순한 PLS 공간 재구성
품질이 아니라, **AWQ build 하나당 더 유용한 의사결정을 만드는가**라는
실제 주장을 검증한다.

### C.3 Multi-KV companion ablation (본문 3.8절)

Multi-KV 평가는 한 번의 빌드에서 label 수를 늘리지만 군집된 data를
만든다. 큰 $K$는 소수 가중치 할당의 KV 반응을 조밀하게 측정한다. 반면
총 label 수나 총 시간이 고정되면 작은 $K$가 더 다양한 가중치 할당을
탐색할 수 있다. 따라서 “평가한 architecture 수”만 보고하면 실질적인
독립 표본 수를 과장하게 된다. $K$는 **가중치 빌드당 전체
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

### C.4 대응된 $20\times20$ 순위 격자

각 축의 $\eta$-front에서 비용에 따라 블록 20개씩을 표본화하고, 400개
모든 곱을 HQQ와 AWQ로 평가한다. 상대 축별 Kendall $\tau$, Spearman
$\rho$, 쌍별 불일치율, 최악의 위반 마진
$\widehat V_W,\widehat V_{KV}$를 보고한다. 정리 3에 직접 연결되는 통계는
평균 $\tau$가 아니라 최대 위반 마진이다. 높은 $\tau$도 하나의 치명적인
순위 역전을 숨길 수 있다.

### C.5 축 순위 식별 대결

정리 2의 반증 실험이다. 같은 평가 예산에서 (i) 파트너를 공유하는 매칭
표본과 (ii) 파트너가 전부 다른 무작위 joint 표본에 additive 또는 GAM
보정을 적용한 추정으로 각각 축 순위를 복원하고, 참조 순위와의 Kendall
$\tau$ 및 그 순위로 구성한 frontier product의 coverage를 비교한다. 두
가지 통제가 필수다. 첫째, 통계는 전역이 아니라 같은 비용 대역 내
쌍으로 보고한다. 무작위 블록은 품질 범위가 넓어 전역 순위 통계는 거의
항상 좋게 나오기 때문이다. 둘째, 공격적 corner 셀을 표본에 반드시
포함한다. 정리 2의 주장은 보정이 corner에서 실패한다는 것이므로,
corner 없는 비교는 검정력이 없다. 예상과 달리 비매칭+보정 추정이
corner를 포함해서도 동등하다면, 정리 2의 실무적 무게는 축소해 보고해야
한다.

### C.6 Off-front 교체 경로

최소 30개의 off-front 또는 돌연변이 결합 구조 $a$를 표본화한다. 각
구조의 축별 frontier 투영을 구하고, 순차 교체 증명에 필요한
$(a_W,a_{KV})$, $(a_W,p_{KV})$, $(p_W,p_{KV})$를 측정한다. 투영된 끝점을
재사용하면 약 60회의 새 AWQ 평가가 필요하다. 이 실험은 기존 on-front
격자가 포함하지 않는 교체 대상 및 상대 축 역할을 직접 검증한다.

### C.7 1단계 frontier 충실도

각 축 탐색을 서로 다른 seed로 반복하고, 상호 지배 관계, hypervolume
차이, 다른 run의 frontier가 $\eta$-cover되는 비율을 측정한다. 이는 식
(5)에 숨겨진 1단계 오차를 추정한다. 이 분석이 없으면 정리 3은 참
frontier에는 적용되지만 실제 알고리즘이 찾은 frontier에 적용된다고
주장할 수 없다.

### C.8 프록시 적용 범위

HQQ--AWQ 일치도를 축별, 예산 간, 고정 $(c_W,c_{KV})$ cell 내부, 동일
cell에서 구조적으로 먼 쌍이라는 네 해상도에서 보고한다. 예산별로 HQQ
최상 구조의 AWQ regret과 HQQ top-$q$ shortlist 안에 AWQ 최상 구조가
포함되는 비율을 측정한다. 상관관계와 함께 Pareto overlap 및 지배
hypervolume, 식 (8)을 만족하는 target-front 제외점의 수, 식 (9)의
예산별 regret gap을 제시해야 한다. 높은 전역 $R^2$나 시각적으로 비슷한
frontier만으로 프록시 지배 증명서가 없다고 결론 내릴 수 없다.

### C.9 일반화

전체 분석을 Qwen2.5-7B처럼 attention 구조가 다른 모델 하나 이상과 긴
문맥 dataset에서 반복해야 한다. 탐색 효율을 위한 주 지표는 answer-phase
JSD로 유지할 수 있지만, 최종 구조는 perplexity 및 LongBench, RULER와
같은 장문 과제에서 평가해야 한다 [14, 15]. 탐색 지표의 Pareto 개선만으로
실제 배포 품질이 향상되었다고 결론 내릴 수 없다.

### C.10 Front-지배의 직접 검증 (따름정리 3.1)

C.4와 C.6은 따름정리 3.1의 bound 재료인 $\widehat V$를 감사한다. 그러나
$\widehat V_W+\widehat V_{KV}$는 두 교체의 최악을 각각 더한 union-bound형
상한이라 구조적으로 보수적이므로, front-지배 자체를 직접 측정하는 다음
프로토콜을 추가한다. 이 프로토콜은 검증인 동시에 반증 채널이다.

1. **실현 containment gap.** 전량 측정된 pool $P$ — 4,365 paired
   archive, 대응 격자, 그리고 탐색이 선택하지 않은 신규 무작위 pool —
   에서 $\operatorname{PF}(P)$와 $\operatorname{PF}(P\cap\mathcal{C})$를
   계산하고, 후자가 전자를 additive $\epsilon$-dominance로 덮는 최소
   $\epsilon$을 예산 대역별로 보고한다. 이론 예측은 gap
   $\le\widehat V=0.0137$이다. gap이 $\widehat V$를 초과하는 대역이
   하나라도 나오면 $\widehat V$의 empirical sup가 미달 감사(off-front
   역할 누락)라는 반증이다. 예산 상한 $\tau$를 스윕한 budgeted regret
   곡선(식 (7')의 좌우변 차), tolerance별 front recall, hypervolume
   차이를 함께 보고한다.
2. **Bound 대 실현 slack.** 대응 격자에서 각 $a$의 교체 사슬이 실제로
   만든 손실 증가 $\max_a[y_D(b(a))-y_D(a)]$를 계산해 상한
   $\widehat V_W+\widehat V_{KV}$와 비교한다. 실현 slack이 상한보다
   충분히 작으면, 정리는 보수적 보증으로 실측은 더 강한 실제 성능으로
   분리해 보고한다.
3. **Band폭 스윕.** $\epsilon$-band 폭을 스윕하며 containment gap과 후보
   공간 크기 $|\mathcal{C}|$를 함께 그린다. 현재 운영 band폭은 근거가
   인쇄되지 않은 노브이므로, coverage 개선과 후보 공간 증가의 절충
   곡선에서 knee를 읽어 운영 선택의 근거로 삼는다.
4. **양방향 band 손실률.** 배포 band 선택에서 지배점이 band 하단 아래로
   빠지는 빈도와 그때의 손실 차이를 보고해, 단측 보장(3.10절 remark)의
   실무 영향을 정량화한다.
5. **통제.** 모든 통계는 noise floor $\delta_z$ 확정 후 같은 비용 대역
   내에서 계산하고 corner 셀을 과표집한다. Pool이 탐색-선택된 표본이라는
   편향은 C.1의 held-out pool과 축소 공간 전수 oracle로만 제거되며,
   interaction 증폭 $\lambda$-stress에서 gap($\lambda$)과
   $\widehat V(\lambda)$의 동반 증가 추적도 C.1의 replay 설계를
   재사용한다.

## 부록 D. 이론적 보장의 한계 (전문)

정리 1은 smoothness 이외의 구조가 없는 함수족에 대한 minimax 하한이다.
현재 이산 양자화 genome의 metric dimension이 raw cell 수와 같다고
증명하지 않으며, 특정 GP나 진화 알고리즘이 그 하한을 정확히 달성한다고
주장하지도 않는다. 명제 1의 지수 이점은 축별 목적이 낮은 차원에서
학습되고 유효한 $\eta$-cover를 반환한다는 조건 아래 **screening**에만
적용된다. Product 크기, 2단계 AWQ 표본, proxy 비용을 합친 종단간 비용이
항상 더 작다는 정리는 아니다. Interaction이 커져 $V_i$가 커지면 더
빠르게 찾은 후보 집합의 품질이 나쁠 수 있다.

정리 2는 축 순위 **식별**의 결과이지 joint frontier **발견**의 하한이
아니다. 함수형 가정을 받아들이는 방법 — 예를 들어 가산 모형을 가정한
joint 회귀 — 은 비매칭 관측에서도 순위를 복원할 수 있으며, 실제로 배포
가능한 대역 안에서는 상당히 성공한다. 정리 2가 배제하는 것은
**가정 없는** 식별뿐이고, 그 가정이 공격적 corner에서 실패한다는 측정과
결합될 때에만 축별 설계의 필연성이 성립한다. 또한 명제 2는 (A1) 근사
충분성과 $z$ 추정 정확도 $\varepsilon_z$를 전제하며, 두 상수 모두
감사된 영역의 empirical 추정치다.

정리 3은 존재성 정리이며 NSGA-III나 surrogate의 수렴 정리가 아니다.
종단간 regret에는 추가로 (i) 1단계 frontier 오차, (ii) 측정한 순서 위반
마진, (iii) product 최적점 대비 2단계 탐색 오차, (iv) 거의 같은 구성
사이의 측정 잡음이 포함된다. 이 항들은 분리해서 보고해야 한다.
따름정리 3.1은 이 스코프를 그대로 상속하면서 두 가지를 더한다. 보장은
목적 공간의 $\epsilon$-지배이지 결정 공간의 Pareto 집합 포함이 아니며
— joint-최적 구조 자체는 off-front 블록을 포함해 product 밖일 수 있다
— slack $V_W+V_{KV}$는 union-bound형 상한이라 실현 gap과의 차이는 부록
C.10의 직접 측정으로만 확인된다. 현재
근거는 하나의 모델과 하나의 compression family에 집중되어 있다.
정리 4 역시 모든 프록시가 항상 실패한다는 무조건적 불가능성 정리가
아니다. 식 (8)의 지배 증명서 또는 식 (9)의 양의 gap이 있을 때
프록시-only 수렴이 실패한다는 조건부 결과다. 현재 4,365개 archive는
실제 운영 후보에 대한 직접 증명서를 제공하지만 AWQ 탐색으로 수집된
집합이므로, 전체 조합 공간의 front 누락률을 추정하려면 독립적인 held-out
후보와 다른 모델에서 같은 감사를 반복해야 한다.
정리 5도 현재 18차원 PLS--GP의 정확한 label 수를 주는 정리가 아니다.
Hölder smoothness, i.i.d. design, 독립 잡음 가정은 adaptive search
archive와 여러 KV companion을 포함한 현재 data에서 자동으로 성립하지
않는다. 따라서 이론은 차원에 따른 scaling을 설명하고, 실제 최소 build
수는 부록 C.2의 grouped held-out 학습곡선으로 결정해야 한다.
Strided JSD의 높은 상관관계도 독립적인 모델과 구성 pool에서 재현되어야
한다. 마지막으로 product 안의 더 저비용인 점이 양방향 메모리 band의
하한보다 낮아질 수
있다. 정리의 문자 그대로의 결론은 비용 상한 제약에 적용된다. 양방향
보고 구간에서는 더 비싸지 않은 근사 대체점이 존재함을 보장하지만, 그
대체점이 같은 구간 안에 남는다고 보장하지는 않는다.

## 부록 E. 관련 연구 상세: 고차원 블랙박스 최적화와 표본복잡도

구조를 가정하지 않은 고차원 블랙박스 최적화에는 근본적인 표본복잡도
장벽이 있다. $d$차원 Lipschitz 함수의 전역 최적화에서는 $N$회 평가 후
최악의 경우 simple regret이 $N^{-1/d}$보다 빠르게 감소할 수 없으며,
Lipschitz 최적화 알고리즘도 같은 minimax 차수에 도달한다 [16]. 잡음이
있는 $s$-smooth 비모수 회귀에서도 최적 MSE 수렴률은
$N^{-2s/(2s+d)}$이므로 차원은 surrogate 학습 속도에도 직접 들어간다
[17]. 이 결과들은 특정 GP 구현이 반드시 실패한다는 뜻이 아니라,
smoothness 이외의 구조가 없으면 낮은 표본 수로 고차원을 메울 수 없다는
worst-case 결과다.

고차원 Bayesian optimization 연구는 이러한 장벽을 additive 구조 또는
낮은 유효 차원 가정으로 우회한다. Additive GP는 각 component의 차원을
작게 제한할 때 regret의 전체 차원 의존성을 크게 완화한다 [18]. REMBO는
목적함수가 낮은 차원의 선형 부분공간을 통해 변한다는 가정 아래 원래
좌표 수가 매우 커도 최적화를 수행한다 [19]. GP-UCB와 후속 simple-regret
분석은 필요한 평가 수를 kernel의 최대 information gain
$\gamma_N$으로 표현한다 [20, 21]. 다목적 환경의 $\epsilon$-PAL도 GP
신뢰구간 아래 원하는 정확도의 Pareto 집합을 식별하는 표본 비용을
분석한다 [22]. 공통점은 표본 수가 입력 좌표 수 하나로 결정되지 않고,
함수족, kernel, smoothness, 잡음, 유효 차원, 목표 오차에 의존한다는
점이다. 따라서 본 연구는 “18차원 표현에는 18개 label이면 충분하다”와
같은 고정 표본 주장을 하지 않고, 조건부 이론과 학습곡선을 함께
사용한다.
