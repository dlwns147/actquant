# 축-우선 2단계 탐색의 이론 재구성 제안 — Theorem C(식별) 중심

2026-08-13. "왜 joint 공간을 직접 탐색하지 않고, 축별(per-axis) 탐색 →
$\epsilon$-Pareto band → 곱 공간 joint 탐색인가"에 대한 이론+분석 장의 재구성
제안. [`../tests/docs/paper_theory.md`](../tests/docs/paper_theory.md)의 기존
패키지(Theorem 1 coverage, Prop 2/3, regret ledger)를 **대체가 아니라 보완**한다:
기존 패키지에 없던 두 정리(Theorem C 식별, Theorem B 유효차원)를 추가하고,
표본복잡도 정리(main.md §3.3 정리 1, §3.5 정리 2)의 강등을 이행하며, Main
Experiments 이전에 필요한 분석 실험을 정리 상수와 1:1로 배선한다.

정리 statement는 논문에 바로 옮길 수 있게 영문, 포지셔닝·비판·주의는 한국어.

---

## 0. 요지 (한 문단)

"축별 탐색이 직접 joint 탐색보다 빠르다"는 **무조건적 정리는 성립하지 않는다**
(§1). 세울 수 있는 것은 질문 3개에 대한 정리 사슬이다 — **Q1 제한해도 잃지
않는가** (coverage, 기존 Theorem 1, 상수 $V$ 실측), **Q2 왜 그 후보를 축에서
얻어야 하는가** (Theorem C: 축 순위는 파트너-매칭 설계로만 식별 가능; 신규),
**Q3 남은 joint 문제는 왜 작은가** (Theorem B: 충분성 ⇒ 2단계 유효차원 2+α;
신규). 속도의 최종 주장("이 분업이 같은 예산의 직접 탐색을 이긴다")은 정리가
아니라 Main Experiments의 matched-budget 비교가 담당한다. 서사는 기존 D-프레임
그대로: *"stage 1 supplies the order — and that order is identifiable only from
axis slices (Thm C); only joint measurement supplies the values — and the value
problem is 2-dimensional (Thm B)."*

---

## 1. 왜 무조건적 효율성 정리는 불가능한가 (비판적 전제 — 본문에 요약 수록 권장)

1. **자기모순.** 축-우선 전략 자체가 joint 공간 알고리즘의 하나(특정 표본
   설계)다. "모든 joint 알고리즘보다 빠르다"는 정리는 자기 자신을 배제해야
   해서 쓸 수 없다.
2. **가산성 반격.** 손실이 정확히 가산적이면($y=f(w)+g(k)$) joint 무작위
   표본이 오히려 유리하다 — 표본 하나가 두 축 정보를 동시에 주고(이중 근무),
   가법 모형 회귀는 성분별(축별) 수렴률을 달성한다. 우리 자신의 실측(주효과
   98.3%, MCKP additive $R^2=0.979$, 대역 내 NSGA 동률)이 이 반격을 지지한다.
   따라서 "거의 가산적"이라는 실증이 강할수록 "joint에서 가법 모형 쓰면
   되잖아"라는 공격도 강해진다 — 축 설계의 필연성은 가산성이 아니라
   **비가산성(포화)**에서 나온다(§3).
3. **minimax 비대칭.** 구 정리 1(CoD $\Omega(\epsilon^{-D})$)·정리 2(Axis-SC)는
   구조 없는 worst-case 진술이라, 구조-인지 직접 탐색(additive GP [18] 등)에는
   적용되지 않는다. "우리에게만 구조를 주는" 비교는 표준 리뷰 공격.
   paper_theory.md §5의 folklore 강등 판정과 정합하게 **Introduction motivation
   한 문단으로 내린다** (main.md §3.3/§3.5는 현재 본문 정리로 남아 있어 내부
   판정과 모순 상태 — §7 반영 지시 참조).

---

## 2. Setup (기존과 동일, 한 줄 추가)

paper_theory.md §0의 Setup을 그대로 쓰되, $F$에 대한 실측 성질을 명시한다:

> The joint loss is empirically a **monotone composite**
> $y(a) \approx F(z_W(a_W), z_{KV}(a_{KV}))$ with $F$ nondecreasing in each
> argument (A2, $\tau_{\min}=0.942$) but **non-additive**: the measured
> saturation slope $\partial y/\partial z_{KV}$ ranges from 1.26 to $-0.06$
> across the weight bands, and the doubly-aggressive corner deviates from the
> additive completion by up to 0.223 (vs. deployable-band RMSE 0.0010).

이 비가산성이 §3의 식별 정리를 발동시키는 조건이다 (가산이면 Theorem C의
결론이 무력해진다 — Remark C.1).

---

## 3. Theorem C — Matched-design identification (신규; "왜 하필 축 슬라이스인가")

Q2를 담당한다: coverage(Q1)는 "축별 front의 곱이 좋다"까지만 말하고, **그
front를 애초에 어떻게 얻을 수 있는가**는 말하지 않는다. Theorem C는 축 순위라는
대상이 흩어진 joint 관측으로는 원리적으로 식별 불가이고, 축 슬라이스가 그것을
식별하는 최소 실험설계임을 말한다.

> **Theorem C (axis-ranking identification requires matched designs).**
> Let $y(w,k) = F(z_W(w), z_{KV}(k))$ with $F: \mathbb{R}^2 \to \mathbb{R}$
> strictly increasing in each argument but otherwise unknown, and $z_W, z_{KV}$
> unknown.
>
> **(i) Non-identifiability from unmatched observations.** Let
> $\mathcal{O} = \{y(w_j, k_j)\}_{j=1}^N$ be any finite set of observations in
> which no two observations share a KV partner ($k_j$ pairwise distinct). Then
> for any two weight configurations $u \ne u'$ observed with distinct partners,
> there exist two triples $(F, z_W, z_{KV})$ and $(\tilde F, \tilde z_W,
> \tilde z_{KV})$, both consistent with every observation in $\mathcal{O}$ and
> both with strictly increasing $F$, such that $z_W(u) < z_W(u')$ under the
> first and $\tilde z_W(u) > \tilde z_W(u')$ under the second. Hence no
> estimator identifies the axis ranking from unmatched data without additional
> functional-form assumptions.
>
> **(ii) Collision scarcity of random designs.** Under i.i.d. sampling of $N$
> joint configurations with the KV marginal uniform on $\mathcal{X}_{KV}$, the
> expected number of partner-sharing pairs is $\binom{N}{2}/|\mathcal{X}_{KV}|$;
> at least one matched pair requires $N = \Omega(\sqrt{|\mathcal{X}_{KV}|})$.
>
> **(iii) Axis slices are the minimal matched design.** The design
> $\{(u, r_{KV}) : u \in S\}$ renders **every** pair of observations matched;
> $|S|$ evaluations identify the full ranking of $S$ within the model class of
> (i). A multi-partner grid $\{(u, v) : u \in S, v \in P\}$ additionally audits
> the partner-invariance of that ranking — the audited ordering-violation
> margin $V$ of Theorem 1.

*Proof sketch.* (i) With all partners distinct, each observation constrains
$F, z$ only through its own composite value; compose $F$ with partner-indexed
monotone reparameterizations to absorb any desired swap of $z_W(u), z_W(u')$
while reproducing every observed value (full construction: 부록, ~반 페이지).
(ii) linearity of expectation + birthday bound. (iii) direct. $\square$

> **Remark C.1 (the additive escape hatch, and why it is closed here).** If
> $F(a,b) = a + b$, unmatched data does identify rankings: differences
> $y(u,k) - y(u',k')$ decompose and a fitted partner effect can be subtracted
> (additive-model regression achieves component-wise rates). Theorem C therefore
> has force exactly because the measured $F$ is non-additive in the aggressive
> region (§2): the subtraction correction is wrong precisely at the corners
> where selection pressure concentrates.

> **Remark C.2 (operational meaning for joint search — hitchhiking).** A joint
> EA credits or blames **both** blocks of a sample with one confounded score: a
> mediocre $w$ survives on a generous partner (free-riding), a good $w$ dies on
> an aggressive one (guilt by association). Increasing the evaluation budget
> does not help — unmatched observations stay unmatched (ii) — and an adaptive
> joint algorithm that deliberately constructs matched comparisons is, by
> definition, performing axis probing: the axis slice is the canonical minimal
> instance of the only design class that works.

**실측 앵커 (본문 인용용):**
- 무작위/QS joint 표본은 매칭쌍을 실제로 못 만든다: correlation의 QS 200표본
  중 유일 W 블록 183개 — 상호작용·순위 질문이 회귀로만 추정 가능해져
  `--grid_sample`(paired factorial)을 별도로 만들어야 했던 것이 이 정리의
  in-repo 실물 증거다.
- 축 슬라이스가 식별한 순위의 파트너-불변성: 20×20 격자 $\tau_{\min}=0.942$,
  $\hat V = 0.0137$ (Theorem 1의 상수 — 두 정리의 분업: C는 "얻을 수 있다",
  1은 "얻은 것이 안정적이다").

**비판적 검증 / 정직한 한계:**
- **식별 ≠ 최적화 하한.** Theorem C는 축 *순위 식별*의 하한이지 joint *front
  발견*의 하한이 아니다. 가산성을 가정한 joint 탐색이 대역 내에서 성공할 수
  있음은 우리 MCKP 실측이 보여준다 — 그 성공이 코너에서 깨진다는 것(additive
  completion 오차 0.223)까지가 정직한 전체 그림이다. "직접 탐색이 실제로
  지는가"는 Main Experiments 몫.
- **자명성 공격.** 증명이 짧다. 방어는 기존 프레임 그대로: simple but
  load-bearing — 기여는 정리 자체가 아니라 (a) stage-1이 생산하는 대상(축
  순위)을 정확히 특정하면 이 정리가 축-우선 설계의 *필요조건화*가 된다는
  전개, (b) 매칭 희소성의 in-repo 실증, (c) Prop 2와의 대칭 결합.
- **Prop 2와의 분업 (완전한 대칭).** Prop 2: 축 관측만으로는 내부 값(교환율)
  식별 불가 → joint 측정 필요. Theorem C: 비매칭 joint 관측만으로는 축 순위
  식별 불가 → 축 슬라이스 필요. 두 방향의 식별 불가능성이 합쳐져 "축
  슬라이스 + 소량 joint 측정"이라는 2단계 설계를 **양쪽에서** 필연화한다.
  이 대칭을 본문에서 명시하면 설계가 휴리스틱이 아니라 식별 이론의 귀결로
  선다.
- **novelty 확인 필요.** econometrics의 단조 합성 식별(예: nonseparable
  models with monotonicity, Matzkin 계열)과 DOE의 blocking 문헌에 인접 결과가
  있을 수 있다 — 제출 전 문헌 확인 후 차별점(이산 조합 공간 + Pareto
  screening 목적 + 생일 역설 정량화) 명시할 것.

---

## 4. Theorem B — Stage-2 effective dimension (신규; "남은 문제는 왜 작은가")

Q3의 stage-2 쪽을 담당한다. 구 표본복잡도 정리와 달리 folklore가 아니라 이
문제의 **측정된 구조(A1 충분성)**에서 나오는 문제-특정 결과다.

> **Theorem B (sufficiency reduces stage 2 to a bivariate problem).** Assume
> (A1) with constant $\varepsilon_0$:
> $|y(a) - F(z_W(a_W), z_{KV}(a_{KV}))| \le \varepsilon_0$ for all
> $a \in \mathcal{X}$, with $F$ monotone (A2). Suppose stage-2 candidates carry
> a representation $\hat z = (\hat z_W, \hat z_{KV})$ with
> $|\hat z_i - z_i| \le \varepsilon_z$ and $F$ is $L_F$-Lipschitz on the
> relevant range. Then estimating $y$ over the candidate space to accuracy
> $\varepsilon$ reduces to estimating the bivariate monotone function $F$ to
> accuracy $\varepsilon - \varepsilon_0 - L_F \varepsilon_z$: the stage-2
> sample requirement is that of a **2-dimensional** regression problem (plus
> exact cost coordinates), independent of the 352 genome cells.

*Proof sketch.* 삼각 부등식으로 오차 분해 후 2-D 비모수 회귀율 적용. $\square$

**실측 앵커**: $\hat\varepsilon_0$ median 0.0065 / sup 0.034 (matched-$z$ 113쌍,
QS n=200; slope-corrected identity residual 0.0015–0.002). 상호작용의 코너
국소성(전체 변동의 1.73%)이 $\varepsilon_0$의 구조적 근거.

**비판적 검증 / 정직한 gap:**
- **$\varepsilon_z$ 가정이 실제 gap이다.** 신규 후보(미측정 mutant)의 $z$는
  관측 불가이므로 "z를 $\varepsilon_z$로 예측하는 표현"이 필요하다 — 이것이
  **PLS 임베딩의 이론적 지위**다(현재 main.md §3.7은 PLS를 공학 장치로만
  서술; Theorem B와 연결하면 지위가 승격된다). $\varepsilon_z$는 1차 아카이브
  기반 PLS→z 복원 오차로 실측 가능.
- (A1)의 sup는 전 공간 검증 불가 — 기존 스탠스대로 audited-domain 한정 서술.
- **직접 검증 실험 (저비용, §6 E3)**: stage-2 surrogate 입력을
  $(\hat z_W, \hat z_{KV}, c_W, c_{KV})$ 4차원으로 제한 vs 프로덕션 PLS-18
  vs full one-hot, 동일 held-out(가중치-가족 split)에서 cell/band 채점. (A1)이
  맞다면 4차원이 근접해야 하고, 근접하지 못하면 이 정리의 적용 범위를 축소
  서술해야 한다.

---

## 5. 사슬 전체 — 질문 3개 × (정리, 가정, 분석 실험, 잔여) 매핑

이 표가 이론+분석 장의 뼈대다. **분석 실험은 전부 Main Experiments 이전**(탐색
A/B 없이 실행 가능)이고, 각 정리의 상수를 직접 잰다.

| 질문 | 정리 | 핵심 가정 | 상수(실측) | 분석 실험 (사전 감사) | 상태 |
|---|---|---|---|---|---|
| Q1 곱 공간으로 제한해도 잃지 않는가 | **Theorem 1** (front-product coverage, V-형) | 비용 분리(항등) + (A3) front 적정성 | $\hat V = 0.0137$ (범위의 2.6%) | 20×20 격자(완료); **무작위 paired 격자** ~144 evals(front 편향 제거); **off-front 사슬 감사** 60 evals | 부분 완료 |
| Q2 왜 그 front를 축에서 얻어야 하는가 | **Theorem C** (matched-design identification) | 단조 합성 $F$, 비가산(실측) | 매칭쌍 0/200 (QS); $\tau_{\min}=0.942$ | **식별 대결** ~250 evals: 같은 예산의 매칭 표본 vs 비매칭+가법 보정 표본에서 축 순위 복원 τ 비교 + noise floor 40 evals | 신규 |
| Q3a stage-1 비용이 왜 감당 가능한가 | 조건부 명제 (구 정리 2, **motivation 강등**) | smoothness + 유효차원 | packing-slope 미측정 | 축별/joint 유효차원 추정 (기존 표본 재사용) | 강등+저비용 |
| Q3b 남은 joint 문제가 왜 작은가 | **Theorem B** (effective dimension 2+α) | (A1) $\varepsilon_0$ + 표현 $\varepsilon_z$ | $\hat\varepsilon_0$ 0.0065/0.034 | **z-충분성 ablation** (4-dim vs PLS-18 vs full; 기존 인프라) | 신규 |
| Q2.5 왜 정확 front가 아니라 $\epsilon$-band인가 | Cor 1.2 (noise robustness) | 1차 노이즈 $\delta_z$ | $\delta_z$ 미확정 | noise floor 45 evals (Q2와 공유) | 대기 |
| 왜 2차는 AWQ 측정인가 | Prop 2 + Prop 3 (기존) | — | $\rho_{\text{int}}=0.40$ 등 | 완료 (프록시 스코프는 본 문서 범위 밖) | 완료 |
| 총합 | regret ledger | — | $r_{\text{stage1}}, r_{\text{search}}$ **미정량** | **원리적으로 사전 감사 불가** — Main Experiments로 이관 | 이관 |

**$\epsilon$-band의 두 가지 이유 (Q2.5, 본문 한 단락):** (i) 1차 점수 노이즈
$\delta_z$ 하에서 정확-front만 남기면 참-front 점이 노이즈 한 번에 탈락한다 —
$2\delta_z$-band가 보장을 slack $2(\varepsilon_0+\delta_z)$로 복원(Cor 1.2).
(ii) 같은 비용·근접 loss의 서로 다른 레이어 패턴을 남겨야 2차의 교배·돌연변이
재료가 된다(band 내 구조 다양성 — main.md §3.4가 이미 서술). 돌연변이는 곱
공간을 포함한 채 확장이므로 Q1의 존재 보장을 깨지 않는다.

**분업의 정직한 경계:** 사전 감사가 인증하는 것은 **지형(landscape)** 상수
($V, \varepsilon_0, \delta_z$, 상호작용 위치)뿐이다. **알고리즘** 항 — 1차가
front를 실제로 찾는가($r_{\text{stage1}}$), 2차가 덮인 점을 찾는가
($r_{\text{search}}$) — 는 탐색을 돌려야만 측정되며 Main Experiments의
matched-budget 비교가 담당한다. 이 분업을 본문에 표로 명시하는 것 자체가
"정리가 알고리즘 성공을 보장한다"는 과장 공격의 차단이다.

---

## 6. 분석(사전 감사) 실험 배터리 — 설계 함정 포함

전부 단일 방법(HQQ) 세계에서 실행 가능, 합계 ~450–550 evals ≈ 수 GPU-시간.

- **D0 noise floor** (~40 evals): 동일 arch 반복측정(calib 서브셋 교체) →
  $\delta_z$. 아래 모든 flip/순위 통계의 해석 전제.
- **D1 무작위 paired 격자** (12×12 = 144 evals): front가 아닌 **무작위** W/KV
  블록(비용 층화 + 코너 셀 고정)의 전 조합. 산출: 파트너-간 τ, $\hat V_{rand}$,
  ANOVA 상호작용 몫. 기존 front-기반 20×20과 대조해 front-조건화가 판정을
  바꾸는지 확인.
- **D2 코너 프로브** (~60 evals): 공격적 저비트 셀 추가 조밀 측정 — 전역
  지표가 희석하는 상호작용의 위치 확인.
- **D3 식별 대결** (D1 재사용 + ~100 evals): Theorem C의 직접 실증. 같은
  평가 예산으로 (a) 매칭 표본, (b) 비매칭+가법/GAM 보정 표본에서 축 순위를
  복원해 참조 순위와 Kendall τ 비교. 예상: (b)는 가산성 가정에 의존해서만
  복원되고 코너 포함 시 붕괴 — 붕괴하지 않으면 Theorem C의 실무적 무게를
  축소 서술해야 한다(반증 가능성 명시).
- **D4 기준 파트너 민감도** (~90 evals): $r_{KV}$를 고/중/저 3곳으로 바꿔 축
  순위 재측정 — stage-1의 기준-고정 설계 선택의 안전성.
- **D5 유효차원 가산성** (기존 표본 재사용): Hamming 거리 vs $\Delta y$
  packing slope, 축별 vs joint — Q3a의 "차원이 실제로 더해지는가".
- **E3 z-충분성 ablation** (기존 인프라): §4 참조.

**설계 함정 4개 (모든 통계에 적용):**
1. **전역-vs-대역내 착시**: 무작위 블록은 품질 범위가 넓어 전역 순위 통계가
   공짜로 좋게 나온다(비트 축 지배 — 프록시 상관에서 이미 확인한 함정). 모든
   순위/flip 통계는 **같은 비용 대역 내 쌍**으로 보고.
2. **영역 이동**: 사전 감사는 무작위 영역, 탐색은 front(극단 집합)에 집중 —
   코너 고정 층화로 완화하되 "필요조건, 충분조건 아님"을 명시.
3. **임계값 사전 등록**: 합격선을 사후에 정하면 데이터 스누핑.
   EXPERIMENT_PLAN §1의 $\lambda$-상호작용 증폭 실험($y_\lambda = \mu + \alpha
   + \beta + \lambda g$)으로 axis-first regret이 실제로 꺾이는 $V$ 수준을 찾아
   그 지점을 사전 등록; 가능하면 둘째 모델(Qwen)에서 게이트 블라인드 검증.
4. **원리적 한계**: (A3)·$r_{\text{search}}$는 지형이 아니라 알고리즘 성질 —
   사전 감사로 불가, 최소 1회의 소규모 anchor 비교(축소 공간 충분)로 감사
   프로토콜 자체를 검증한 뒤 새 모델에는 프로토콜만 적용.

이 배터리는 논문 주장도 개선한다: "한 모델에서 A/B를 이겼다"(전이 불가)가
아니라 "**~500 evals짜리 사전 감사 프로토콜의 출력이 정리의 상수를 채우고,
합격이면 축-우선이 안전하다** — Llama에서 1회 anchor로 검증, 새 모델엔 감사만
적용"이라는 전이 가능한 방법론이 된다.

---

## 7. main.md 반영 지시 (구체적)

**반영 상태 2차 (2026-08-13):** **따름정리 3.1 (전체 joint front의
$\epsilon$-지배)** 을 main.md §3.5에 신설 — 정리 3(점별 V-지배)의
front-수준 따름정리로, (i) 목적공간 $\epsilon$-지배 ≠ 결정공간 포함,
(ii) 단측 비용 방향, (iii) union-bound slack의 보수성을 명시. 부록
A.4에 증명(+band 버전, 한계 주석), 부록 C.10에 직접 검증 프로토콜
(실현 containment gap·bound 대 실현 slack·band폭 스윕 knee·양방향 band
손실률·통제)을 추가하고, §3.5 ANOVA 문단을 "정리의 전제가 아니라 V가
작은 이유의 보조 증거"로 재배치, §3.9 프로토콜 목록과 §3.10/부록 D
한계에 반영. 근거: 정리 3의 V-형은 additive 기반이 아니지만(순차 교체 +
마진), slack 합산·ANOVA 배치가 additive 인상을 줬던 것을 서술로 교정.

**반영 상태 (2026-08-13):** 3–5번 완료 — main.md에 **정리 2**(축 순위의
매칭 설계 식별 = 본 문서의 Theorem C, §3.5 신설)와 **명제 2**(근사 충분성
차원 축소 = Theorem B, §3.7 신설)를 통합했고, 구 정리 2(Axis-SC)는
**명제 1**로 격하, §3.9에 "축 순위 식별 대결" 반증 실험과 z-충분성
ablation arm, §3.10에 식별≠발견 한계 단락, 초록·서론에 식별 문장을
추가했다. **미이행: 1번(§3.3 정리 1의 서론 강등)과 2번의 위치 이동** —
정리 1은 아직 §3.3 본문 정리로 남아 있다(별도 편집 판단 필요).

1. **§3.3 (정리 1 CoD)**: 본문 정리 지위 해제 → Introduction의 motivation 한
   문단으로 축약 이동 ("구조 없는 worst-case는 지수적; 본 논문은 구조를
   측정하고 그 구조가 허용하는 설계를 택한다"). packing-number 형식화는 부록.
2. **§3.5 (정리 2 Axis-SC)**: "정리" → 조건부 명제로 격하, D5(packing slope)
   실측과 함께 제시. §3.5의 실질 무게는 정리 3(coverage)과 실증 분석으로 이미
   충분하다.
3. **§3.5 앞에 Theorem C 신설** (§3 "축별 frontier를 먼저 찾는 이유"의 첫
   번째 논거로): 현재 §3.5는 "얼마나 적은 평가로 추리는가"(비용)부터
   시작하는데, 논리 순서는 "그 대상(축 순위)이 축 설계로만 식별된다"(Theorem
   C)가 먼저다. Remark C.2(무임승차/연좌제)는 독자 직관용으로 본문 유지 권장.
4. **§3.7 (PLS)**: Theorem B를 §3.7 도입부에 배치해 PLS의 지위를 "공학적 차원
   축소"에서 "충분 통계 $z$의 추정기($\varepsilon_z$ 공급자)"로 승격.
5. **초록·서론의 표본복잡도 문장**: "minimax 분석이 …임을 보인다" 계열 문장을
   "측정된 구조(단조 합성·비가산·충분성)가 축-우선 설계를 식별-이론적으로
   필연화한다" 계열로 교체.
6. **금지 문장 (paper_theory §6과 정합)**: "축별 탐색이 항상 빠르다",
   "정리가 알고리즘의 성공을 보장한다", "손실이 가산적이다", "352차원이므로
   지수적으로 어렵다"(worst-case 한정 없이).

---

## 8. 리뷰어 공격면 (이 문서 신규분만; 기존 표는 paper_theory §6)

| 공격 | 확률 | 방어 |
|---|---|---|
| "Theorem C 자명 — 당연히 매칭해야 비교 가능" | 높음 | simple/load-bearing 자기 규정 + 생일 역설 정량화 + QS 183/200 실물 증거 + Prop 2 대칭 결합. "당연한 것을 설계 원리로 승격해 감사한 논문이 없었다"로 위치 |
| "가산 모형 joint 회귀면 충분 (Remark C.1)" | 확실 | D3 식별 대결을 우리가 먼저 실행 — 대역 내 동률이면 그렇게 보고하고 코너+무가정 강건성으로 주장 한정 (이 정직성이 방어) |
| "Theorem B는 (A1) 재포장" | 중 | 인정하되 역할 명시: (A1)의 *결과*를 표본복잡도 언어로 옮겨 "왜 2차 예산이 작아도 되는가"를 정량화; E3 ablation이 반증 가능성 공급 |
| "식별 정리가 최적화와 무슨 상관" | 중 | 스코프 명시(식별 ≠ 최적화 하한) + Remark C.2(EA 신호 오염) + Main Experiments로 잔여 주장 이관 — 과장 안 함이 방어 |
| "econometrics/DOE에 이미 있다" | 중 | 제출 전 문헌 확인(§3 novelty 항목); 차별점 = 이산 조합 공간·Pareto screening 목적·매칭 희소성 정량화 |

## 9. 제출 전 TODO (이 문서 기준 증분)

1. Theorem C (i)의 완전한 구성 증명 반 페이지 (부록).
2. D3 식별 대결 + D0 noise floor 실행 (~150 evals) — Theorem C의 실증이자
   반증 시도.
3. E3 z-충분성 ablation — Theorem B의 실증 (기존 인프라, GPU-시간 소).
4. D1 무작위 paired 격자 — Theorem 1 상수의 front-편향 제거.
5. main.md §3.3/§3.5/§3.7 재배치 (§7 지시).
6. Theorem C novelty 문헌 확인 (nonseparable monotone identification, DOE
   blocking).
