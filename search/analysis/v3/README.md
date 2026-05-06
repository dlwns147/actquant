# LLM Quantization NAS — Per-Method Pareto 조합의 수학적 정당화

본 문서는 LLM mixed-precision quantization (W bits / KV bits·group-size / KV dim
pruning) 의 Neural Architecture Search 에서 **per-method local Pareto Frontier
의 조합이 전체 결합 search 의 practical $\epsilon$-Pareto approximation 임**을
수학적으로 정당화하고, ARD-GP / RBF surrogate 의 held-out residual 로 $\epsilon$
가 실제로 작음을 실증한다. 결과는 결정론적 uniform bound 가 아니라 **measured
surrogate residual 하의 empirical 2$\epsilon$-Pareto evidence** 형태로 해석한다.

데이터: Llama-3.1-8B-Instruct (n_block=32, n_linear=7), AWQ + KIVI + Channel/Token
KV-quant, wikitext2. 3-way random N=200 측정, fixed split = 50 train (27
quantile-grid + 23 maximin) / 150 test. **모든 surrogate 의 입력은 per-method
scalar summary $z_i = \ell_i(a_i) =$ standalone JSD/CE loss
(`prev_metric`)** — exact Hoeffding $f_i$ 가 아닌 **low-dimensional sufficient
statistic / proxy** 로 사용한다 (§2.1 참조).

---

## 1. Problem — 결합 탐색의 불가능성

### 1.1 실제 search space 크기

세 method 가 동시에 작용하면 search space 는 method 별 선택의 Cartesian
product 이며, `config/llama.json` (n_block=32, n_linear=7) 와
`scripts/search_think.sh` (W_BITS="2 3 4", K/V_BITS="2 3 4", K/V_GROUP_SIZE
= {32, 64, 128}, K/V_PRUNING_DIM = "0 16 32 48 64") 의 옵션으로 계산하면

| Method | Per-position 옵션 수 | Position 수 | Cardinality |
|---|---|---|---|
| W bits ($\{2,3,4\}$) | $3$ | $n_{\text{block}} \cdot n_{\text{linear}} = 32 \cdot 7 = 224$ | $3^{224} \approx 7.5 \times 10^{106}$ |
| KV bits × gs ($\{2,3,4\}\times\{32,64,128\} = 9$) | $9$ | $n_{\text{block}} \cdot 2 \text{ (K,V)} = 64$ | $9^{64} \approx 1.2 \times 10^{61}$ |
| KV dim ($\{0,16,32,48,64\}$) | $5$ | $n_{\text{block}} \cdot 2 \text{ (K,V)} = 64$ | $5^{64} \approx 5.4 \times 10^{44}$ |
| **Joint full $\|\mathcal{X}_{\text{full}}\|$** | — | — | $\boxed{\;3^{224} \cdot 9^{64} \cdot 5^{64} \;\approx\; 4.8 \times 10^{212}\;}$ |

즉 사용자 공식 그대로
$|\mathcal{X}_{\text{full}}| = (\text{W bits})^{n_{\text{linear-W}}} \cdot
(\text{KV bits·gs})^{n_{\text{linear-KV}}} \cdot
(\text{KV dim})^{n_{\text{linear-KV}}}$.
직접 sweep 은커녕 행성 단위 compute 로도 한 model 의 joint sweep 은
불가능하다. 어떤 형태로든 분해(decomposition)에 의존해야 한다.

### 1.2 Theorem 1 — Exponential search growth (sample-complexity necessity)

**Discrete hit-probability form.** 각 axis 가 finite set 이고 $\mathcal{X} =
\prod_{j=1}^d \mathcal{X}_j$, $|\mathcal{X}_j| = m_j$ 라 하자. 어떤 target set
$T \subseteq \mathcal{X}$, $|T| = t$, $p = t/|\mathcal{X}|$ 에 대해 $N$ 번의 i.i.d.
uniform random sampling 으로 $T$ 를 적어도 한 번 hit 할 확률은 $1 - (1-p)^N$
이고, 성공확률 $1 - \eta$ 를 얻기 위해서는

$$
N \;\ge\; \frac{\log(1/\eta)}{p}.
$$

새 axis $d+1$ ($|m_{d+1}|$) 가 추가되면 $t$ 가 그대로일 때 $p \to p/m_{d+1}$,
즉 같은 hit probability 를 위한 sample 수가 **multiplicative 하게** $m_{d+1}$
배 증가한다.

**Continuous covering form.** $[0,1]^d$ 를 $\ell_\infty$ 거리에서 fill distance
$\delta$ 로 cover 하기 위한 sample 수의 lower bound 는

$$
N_\delta(d) \;\ge\; (1/\delta)^d.
$$

즉 동일 fill distance 유지 시 sample 수가 dimension $d$ 에 대해 **지수적으로**
증가 (curse of dimensionality, Bellman).

**Corollary (본 setup 정량 환산)**:

$$
|\mathcal{X}_{\text{full}}| = 3^{224} \cdot 9^{64} \cdot 5^{64} \approx 4.79 \times 10^{212}, \qquad
|\mathcal{C}| = |PF_W| \cdot |PF_{KV}| \cdot |PF_{KVD}| \approx 3.04 \times 10^{7}.
$$

축소비율

$$
\frac{|\mathcal{C}|}{|\mathcal{X}_{\text{full}}|} \;\approx\; 6.3 \times 10^{-206}
\quad\Longleftrightarrow\quad
\text{full 대비 약 } 1.6 \times 10^{205} \text{ 배 축소}.
$$

> *Direct joint search suffers from a multiplicative explosion of the discrete
> Cartesian product. Equivalently, in a continuous relaxation, maintaining a
> fixed fill distance requires a covering number exponential in the intrinsic
> dimension. Therefore, per-method decomposition is not merely an engineering
> choice but a **sample-complexity necessity**.*

추가로 multi-objective optimization 에선 objective 수 또는 Pareto front 차원이
커질수록 nondominated point 수가 급증하고 PF 정확 표현에 필요한 point 수도 빠르게
증가한다 (von Lücken et al. 2014). **결합 axis 에서 직접 search 보다 method 별
search 후 low-dimensional surrogate 학습이 sample-efficient**.

---

## 2. Mathematical Foundation — Hoeffding–Sobol Decomposition

### 2.1 변수 정의 및 핵심 가정

세 method 를 $i \in \{W, KV, KVD\}$ 로 indexing 하고 각 method 의 architecture 를
$a_i \in \mathcal{X}_i$ 라 하자. 전체 architecture 는

$$
a = (a_W, a_{KV}, a_{KVD}) \in \mathcal{X}_W \times \mathcal{X}_{KV} \times \mathcal{X}_{KVD}.
$$

각 method 단독 search 에서 얻은 scalar summary (per-method JSD/CE loss) 를

$$
z_i = \ell_i(a_i) \in \mathbb{R}
$$

로 둔다. $z_i$ 가 본 README 의 `prev_metric`. 결합 측정 loss 는 $y(a) =
\mathcal{L}_{\text{all}}(a)$.

**핵심 가정**: 결합 loss 의 중요한 변동이 원래의 수백 차원 bit/dim decision 이
아니라 세 개의 per-method scalar summary 로 대부분 설명된다, 즉

$$
y(a_W, a_{KV}, a_{KVD}) \;\approx\; F(z_W, z_{KV}, z_{KVD}).
$$

이는 **exact Hoeffding decomposition 으로부터 자동으로 따라오는 성질이 아니라**,
"per-method standalone JSD 가 method 의 결합 효과에 대한 충분히 informative 한
sufficient statistic / proxy" 라는 **검증해야 하는 가정**이다. §5 의 ARD-GP
$R^2 \approx 0.996$ 결과가 이 가정을 실험적으로 지지한다.

### 2.2 Hoeffding–Sobol decomposition (on the surrogate input space)

위 가정 하에서 $F \in L^2$ 이고 input distribution 이 product measure 이면,
Sobol–Hoeffding 정리에 의해 $F$ 는 다음과 같이 **유일하게** 직교 분해된다:

$$
\boxed{\;
F(z) \;=\; F_0
     \;+\; \underbrace{F_W(z_W) + F_{KV}(z_{KV}) + F_{KVD}(z_{KVD})}_{\displaystyle F_{\mathrm{add}}(z)\ \text{— first-order (additive)}}
     \;+\; \underbrace{F_{W,KV} + F_{W,KVD} + F_{KV,KVD}}_{\text{pairwise interaction}}
     \;+\; \underbrace{F_{W,KV,KVD}}_{\text{three-way}}
\;}
$$

직교 조건 $\mathbb{E}[F_S(z_S) \mid z_T] = 0\ (\forall T \subsetneq S)$ 로부터
$\mathrm{Var}(F) = \sum_S \mathrm{Var}(F_S)$.

**중요**: 여기서 $F_i(z_i)$ 는 다른 변수에 대한 conditional expectation 으로
정의되며, "per-method standalone $z_i$ 자체와 동일하다"는 자동 보장은 없다. 본
연구는 $F_{\text{add}}$ 를 surrogate 위에서 추정하고, 그 잔차 $r = F - F_{\text{add}}$
의 norm 을 **empirical** 로 측정한다.

### 2.3 Sobol indices — 분산 기반 정량화

각 항의 기여도를 분산 비율로 정량화한 것이 **Sobol index**:

$$
S_i = \frac{\mathrm{Var}(F_i(z_i))}{\mathrm{Var}(F)}, \qquad
S_{ij} = \frac{\mathrm{Var}(F_{ij}(z_i,z_j))}{\mathrm{Var}(F)}, \qquad
\sum_{S \subseteq \{W,KV,KVD\}} V_S = \mathrm{Var}(F).
$$

- **First-order ratio**: $\sum_i S_i$
- **Total interaction**: $1 - \sum_i S_i$
- **Total-order index**: $S_{T,i} = S_i + \sum_{j\neq i} S_{ij} + S_{W,KV,KVD}$
  (단일 input 과 그것이 포함된 모든 interaction)
- 본 연구는 ARD-GP 위에서 Saltelli pick-and-freeze Monte Carlo estimator 로 추정.

### 2.4 Theorem 2 — Surrogate 2ε-Pareto stability

**Definition (α-Pareto).** Loss $y$ 와 complexity $c$ 모두 minimize. 점 $a$ 가
$y$ 에 대해 *α-Pareto* 라는 것은 어떤 $a'$ 도 동시에 $c(a') \le c(a)$ 와
$y(a') < y(a) - \alpha$ 를 만족하지 않는다는 뜻. $\alpha = 0$ 이면 standard Pareto.

**Setup.** Surrogate $\hat y : \mathcal{X} \to \mathbb{R}$ 가 domain $\mathcal{D}
\subseteq \mathcal{X}$ 위에서 true loss $y$ 를 uniform 하게 근사:

$$
\bigl\| \hat y - y \bigr\|_{\infty, \mathcal{D}}
\;=\; \sup_{a \in \mathcal{D}} \bigl| \hat y(a) - y(a) \bigr|
\;\le\; \epsilon.
$$

Complexity $c(a)$ 는 정확히 계산.

**Theorem 2A (PF containment).** 위 setup 에서

$$
\mathrm{Pareto}_{\mathcal{D}}(\hat y, c) \;\subseteq\; 2\epsilon\text{-Pareto}_{\mathcal{D}}(y, c).
$$

**Proof.** $\hat a \in \mathrm{Pareto}_{\mathcal{D}}(\hat y, c)$ 가 $2\epsilon\text{-Pareto}$
가 아니면 어떤 $a' \in \mathcal{D}$ 가 존재하여 $c(a') \le c(\hat a)$, $y(a') <
y(\hat a) - 2\epsilon$. Uniform bound 로부터

$$
\hat y(a') \le y(a') + \epsilon
< y(\hat a) - \epsilon \le \hat y(\hat a),
$$

즉 $a'$ 가 $(\hat y, c)$ 에서 $\hat a$ 를 dominate — 모순. ∎

**Theorem 2B (Budgeted regret form).** 임의 budget $\tau$ 에 대해

$$
a^*_\tau \in \arg\!\!\min_{a \in \mathcal{D}: c(a) \le \tau} y(a),
\qquad
\hat a_\tau \in \arg\!\!\min_{a \in \mathcal{D}: c(a) \le \tau} \hat y(a).
$$

그러면

$$
\boxed{\quad y(\hat a_\tau) \;\le\; y(a^*_\tau) + 2\epsilon. \quad}
$$

**Proof.** $y(\hat a_\tau) \le \hat y(\hat a_\tau) + \epsilon \le \hat y(a^*_\tau)
+ \epsilon \le y(a^*_\tau) + 2\epsilon$. ∎

> **해석 (오해 방지)**: Theorem 2 는 *frontier Hausdorff distance* 가 $\le 2\epsilon$
> 라는 강한 주장이 아니다. 정확한 의미는 "*고정된 complexity budget 에서 surrogate
> 로 고른 design 의 true loss 가 true budgeted optimum 보다 $2\epsilon$ 이상
> 나쁘지 않다*". §6 의 budgeted regret 해석.

> **일반성**: Theorem 2 는 surrogate $\hat y$ 의 *형태에 무관* 하다 ($F_{\mathrm{add}}$,
> ARD-GP, RBF tps+linear, M10 모두 적용). 단, $\epsilon$ 은 *그 surrogate* 의
> residual.

> **Empirical vs deterministic**: 본 연구에서 보고하는 $\hat\epsilon^{\text{test}}$
> 는 **150 held-out joint sample 위의 empirical sup-residual** 이지 $\mathcal{X}$
> 전체 $4.8 \times 10^{212}$ 위의 deterministic uniform bound 가 아니다. 따라서
> 결과는 항상 "**measured residual 하의 empirical 2$\epsilon$-Pareto evidence**"
> 로 표현. Frontier 영역까지 확장하려면 §6 Step 6 의 frontier-knee audit 필요.

### 2.5 Theorem 3 — Local PF Cartesian product coverage

**Setup.** 각 method $i \in \{W, KV, KVD\}$ 에 대해 local proxy loss
$z_i(a_i) = \ell_i(a_i)$, local complexity $c_i(a_i)$, local Pareto frontier

$$
\mathrm{PF}_i \;=\; \mathrm{Pareto}_{a_i \in \mathcal{X}_i}\bigl( z_i, c_i \bigr),
\qquad
\mathcal{C} \;=\; \mathrm{PF}_W \times \mathrm{PF}_{KV} \times \mathrm{PF}_{KVD}.
$$

Joint complexity $c(a) = \Phi(c_W(a_W), c_{KV}(a_{KV}), c_{KVD}(a_{KVD}))$.
Surrogate scorer $H(a) = G(z_W(a_W), z_{KV}(a_{KV}), z_{KVD}(a_{KVD}))$.

**Conditions.**
- **(C1)** $\Phi$ 는 각 argument 에 대해 monotone nondecreasing.
- **(C2)** $G : \mathbb{R}^3 \to \mathbb{R}$ 은 각 coordinate $z_i$ 에 대해
  monotone nondecreasing on the candidate-relevant region.

> (C2) 는 additive surrogate $G(z) = \sum_i h_i(z_i)$ 의 경우 각 $h_i$ 가
> nondecreasing 이면 자동 충족. ARD-GP 같은 nonparametric scorer 의 경우 **별도
> coordinate-wise monotonicity audit** 이 필요 (§5.2 audit 결과 참조).

**Theorem 3 (Coverage).** (C1)–(C2) 하에서 임의의 $a \in \mathcal{X}_{\text{full}}$
에 대해 $b \in \mathcal{C}$ 가 존재하여

$$
c(b) \le c(a), \qquad H(b) \le H(a).
$$

**Proof.** Finite $\mathcal{X}_i$ 에서 임의의 $a_i$ 는 $\mathrm{PF}_i$ 의 한 점
$b_i$ 에 의해 weakly dominated: $z_i(b_i) \le z_i(a_i)$ 이고 $c_i(b_i) \le
c_i(a_i)$. (그렇지 않으면 dominance chain 을 따라 더 나은 점으로 이동, finite set
이므로 결국 PF 도달.) $b = (b_W, b_{KV}, b_{KVD}) \in \mathcal{C}$ 에 대해 (C1)
로 $c(b) \le c(a)$, (C2) 로 $H(b) \le H(a)$. ∎

**Corollary (true-loss coverage).** 추가로 $\|y - H\|_{\infty, \mathcal{X}_{\text{full}}}
\le \epsilon$ 이면 $y(b) \le H(b) + \epsilon \le H(a) + \epsilon \le y(a) + 2\epsilon$.
즉

$$
\boxed{\;
\forall a \in \mathcal{X}_{\text{full}},\ \exists\, b \in \mathcal{C} :\
c(b) \le c(a)\ \text{and}\ y(b) \le y(a) + 2\epsilon.
\;}
$$

**Theorem 3' (Approximate local PF, archive coverage slack).** 실제로 $\mathrm{PF}_i$
는 NSGA2 archive 에서 추출한 **empirical PF** 이다. Archive PF 가 $\delta_i$-covering
property 를 가진다고 하자: 임의의 $a_i \in \mathcal{X}_i$ 에 대해 어떤 $b_i \in
\mathrm{PF}_i$ 가 존재하여 $c_i(b_i) \le c_i(a_i)$, $z_i(b_i) \le z_i(a_i) + \delta_i$.
$G$ 가 coordinate-wise Lipschitz 이고 $|G(z) - G(z')| \le \sum_i L_i |z_i - z_i'|$
이면

$$
H(b) \le H(a) + \sum_i L_i \delta_i,
\qquad
y(b) \le y(a) + 2\epsilon + \sum_i L_i \delta_i.
$$

> **해석**: exact local PF 가 아닌 archive PF 의 경우 최종 corridor 는
> $2\epsilon$ 이 아니라 $2\epsilon + \sum_i L_i \delta_i$. 본 setup 의 archive
> 크기 ($|PF_W|=733$, $|PF_{KV}|=321$, $|PF_{KVD}|=129$) 가 충분하면 $\delta_i$
> 는 작아 추가 slack 무시 가능 — 본 README 는 이 가정 하에 결과를 보고.

### 2.6 Surrogate 형태별 (C2) audit — partial derivative 공식과 검증

**핵심**: Theorem 3 의 (C2) 는 *raw $z_i$ 좌표에 대한 surrogate 의 partial
derivative 부호 조건* 이며 surrogate 형태에 따라 검증 방식이 다르다. 단순 R²
가 높다고 자동 충족되지 않음. ARD length-scale 또한 sensitivity 일 뿐
monotonicity 가 아니다.

| Surrogate | $\partial G/\partial z_i$ 형태 | 검증 방법 |
|---|---|---|
| **M0** $\;G = \sum_i \beta_i z_i$ | $\beta_i$ (constant) | OLS 계수 부호 |
| **M1** $\;G = \beta_0 + \sum_i \beta_i z_i$ | $\beta_i$ (constant) | OLS 계수 부호 (intercept 무관) |
| **M10** $\;G = \beta_0 + \sum \beta_i z_i + \sum q_i z_i^2 + \sum_{i<j}\gamma_{ij} z_i z_j$ | affine in $z$: $\beta_i + 2 q_i z_i + \sum_{j\ne i} \gamma_{ij} z_j$ | box 8-corner check (affine ⇒ extreme on corner) + dense |
| **RBF cubic+linear** $\;G = \sum_k \alpha_k r_k^3 + \beta^\top z + b$ | $\beta_i + 3 \sum_k \alpha_k r_k (z_i - x_{k,i})$ | dense gradient audit |
| **RBF tps+linear** $\;G = \sum_k \alpha_k r_k^2 \log r_k + \beta^\top z + b$ | $\beta_i + \sum_k \alpha_k (2 \log r_k + 1)(z_i - x_{k,i})$ | dense gradient audit |
| **ARD-GP** $\;G(z) = \sum_k \alpha_k k(z, x_k)$, $\alpha = (K + \sigma_n^2 I)^{-1} y$ | $\beta_i + \sum_k \alpha_k k(z, x_k) (x_{k,i} - z_i) / \ell_i^2$ | dense gradient audit |

> **중요한 구분**: 1D main effect $g_i(z_i) = \mathbb{E}_{z_{-i}}[G(z)] - f_0$
> 의 monotonicity 와 *full surrogate* $G(z_W, z_{KV}, z_{KVD})$ 의
> coordinate-wise monotonicity 는 다른 조건이다. 후자가 더 강한 조건이며
> Theorem 3 에 필요한 것은 *후자*. 1D main effect 단조성은 Hoeffding additive
> $\hat F_{\mathrm{add}} = \sum_i h_i(z_i)$ 에 대해서만 (C2) 를 자동 보장
> (script: `verify_main_effect_monotonicity.py`).

**audit 결과** — `archive/verify_monotonicity_all_surrogates.py`. 두 audit set 위에서
finite-difference gradient 부호 비율 측정 — **두 set 모두 동일한 bounding box
$[z_W^{\min}, z_W^{\max}] \times [z_{KV}^{\min}, z_{KV}^{\max}] \times [z_{KVD}^{\min},
z_{KVD}^{\max}] = [0.019, 0.657] \times [0.018, 0.136] \times [0.019, 0.328]$ 안에
있고, 분포만 다름**:

- **$\mathcal{D}_{\mathrm{meas}}$** = 200 measured joint sample. *Non-uniform*
  (실제 측정 분포). Candidate scoring 시 surrogate 가 평가하는 영역의 proxy.
- **$\mathcal{D}_{\mathrm{box}}$** = 5000 Latin-hypercube fill of the same
  bounding box. *Uniform* fill of box-interior — 측정점이 sparse 한 interior
  pocket 까지 cover. Box 바깥 외삽이 아니라 **box 안의 interpolation hole** 까지
  test.

| Surrogate | $\mathcal{D}_{\mathrm{meas}}$ frac ≥ 0 (W / KV / KVD) | $\mathcal{D}_{\mathrm{box}}$ frac ≥ 0 (W / KV / KVD) | 적용 영역 |
|---|---|---|---|
| **M0** ($\beta = [+0.61, +1.24, +0.51]$, all $+$) | 100 / 100 / 100 | 100 / 100 / 100 | **GLOBAL** ✓ |
| **M1** ($\beta = [+0.60, +1.08, +0.47]$, all $+$) | 100 / 100 / 100 | 100 / 100 / 100 | **GLOBAL** ✓ |
| **M10** | 100 / 100 / **99.5** | 88.1 / 83.3 / 68.5 | $\mathcal{D}_{\mathrm{meas}}$ ≥99.5% ✓; box VIOLATED |
| **RBF cubic+linear** | 100 / **98.0** / 99.5 | 100 / 100 / 100 | $\mathcal{D}_{\mathrm{meas}}$ 98% (small wiggle); $\mathcal{D}_{\mathrm{box}}$ ✓ |
| **RBF tps+linear** | 100 / **99.0** / 99.5 | 100 / 100 / 100 | $\mathcal{D}_{\mathrm{meas}}$ 99% ✓; $\mathcal{D}_{\mathrm{box}}$ ✓ |
| **ARD-GP** | 100 / 100 / **99.0** | 100 / 79.0 / 57.9 | $\mathcal{D}_{\mathrm{meas}}$ 99% ✓; box VIOLATED |
| (Hoeffding additive $\hat F_{\mathrm{add}} = \sum h_i$, 1D check) | 100 / 100 / 100 (correlated MC) | 100 / 100 / 100 | **GLOBAL** ✓ |

**해석**:

- **M0 / M1**: OLS 결과 모든 계수가 strict positive → **(C2) 가 무조건 global 충족**.
  Theorem 3 적용 가장 깔끔. 단 residual 이 큼 ($\hat\epsilon_\infty^{\mathrm{M1}} = 0.0590$).
- **M10**: $q_i, \gamma_{ij}$ 가 모두 negative (curvature/saturation 흡수) 이지만
  $\beta_i$ 가 충분히 커서 $\mathcal{D}_{\mathrm{meas}}$ 에선 99.5% 충족.
  Box-interior fill ($\mathcal{D}_{\mathrm{box}}$) 의 8-corner 중 일부 음수 →
  **theorem 적용은 $\mathcal{D}_{\mathrm{meas}}$ 한정**.
- **RBF cubic / tps**: $\mathcal{D}_{\mathrm{meas}}$ 에서 KV 축 small wiggle
  (98–99%, 측정점이 몰린 영역의 fitting 변동). 흥미롭게도 $\mathcal{D}_{\mathrm{box}}$
  의 sparse interior 에서는 100% — RBF interpolant 가 측정점이 적은 hole 영역에서
  linear tail 이 dominate 하여 globally monotone.
- **ARD-GP**: $\mathcal{D}_{\mathrm{meas}}$ 99% ≥ 0 → measured 분포 영역 한정
  (C2) 충족. $\mathcal{D}_{\mathrm{box}}$ 의 sparse interior 에서는 KV 79%,
  KVD 58% 로 깨짐 — box 바깥 외삽이 아니라 **box 안 interpolation hole** 에서
  GP posterior wiggle 이 monotonicity 깨뜨림.
- **Hoeffding $\hat F_{\mathrm{add}} = \sum h_i(z_i)$**: 1D main effect $h_i$
  가 strictly monotone (correlated MC, 38/38 grid pts) → (C2) 자동 global ✓.

> Figures: [`figures/verify_monotonicity_all.png`](figures/verify_monotonicity_all.png)
> (6 surrogate × 3 axis gradient histogram on $\mathcal{D}_{\mathrm{meas}}$),
> [`figures/verify_monotonicity.png`](figures/verify_monotonicity.png) (1D main
> effect),
> [`figures/verify_ardgp_monotonicity.png`](figures/verify_ardgp_monotonicity.png)
> (ARD-GP 3D coord-wise).

본 setup 에서 PF 조합 cardinality 는

$$
|\mathrm{PF}_W| \cdot |\mathrm{PF}_{KV}| \cdot |\mathrm{PF}_{KVD}|
\;=\; 733 \cdot 321 \cdot 129 \;\approx\; 3.04 \times 10^{7},
$$

full $\approx 4.8 \times 10^{212}$ 대비 **약 205 자릿수 축소**.

---

## 3. Why Direct Measurement of $F_S$ Is Impractical (배보다 배꼽)

이론상 각 항 $F_S$ 는 conditional expectation 으로 분리되지만, LLM 단위에서
직접 추정하려면

| 항 | 추정 공식 | 측정 비용 |
|---|---|---|
| $F_i(z_i)$ | $\mathbb{E}_{z_{-i}}[F(z)] - F_0$ | 차원 $i$ 의 각 grid 점마다 나머지 marginalize: $\mathcal{O}(N_{\text{grid}} \cdot N_{\text{MC}})$ |
| $F_{ij}$ | 2-D marginalization | $\mathcal{O}(N_{\text{grid}}^2 \cdot N_{\text{MC}})$ |
| $F_{ijk}$ | full 3-D marginalization | $\mathcal{O}(N_{\text{grid}}^3 \cdot N_{\text{MC}})$ |

본 setup (sample 당 분 단위) 에서 $F_i$ 만 정확히 추정해도 수천 측정,
$F_{ij}$ 추정엔 수만 측정이 필요하다. 즉 **ANOVA 항을 직접 측정하는 비용이
full sweep 비용을 초과**하여 분해의 의미가 사라진다. 배보다 배꼽이 커진다.

해결책: 적은 수의 측정으로 $y_{\mathrm{full}}$ 자체를 세 개의 prev_metric 입력
공간 위에서 **response surface regression** 하고, surrogate 위에서 Sobol /
ANOVA 분해를 수행한다.

---

## 4. Practical Pipeline — Per-Method PF + Response Surface

### 4.1 1단계: method-별 NSGA2 search → PF + 1차 loss

각 method 단독 search 로 다음을 회수 (실측):

| Method | Archive 크기 | PF 크기 | Complexity 범위 |
|---|---|---|---|
| W (wbits) | 10,450 | **733** | [2.25, 4.23] bits |
| KV (kvbits·gs) | 4,147 | **321** | [2.25, 4.59] bits |
| KVD (kvdim) | 4,530 | **129** | [96, 128] dim |

PF 위의 각 점은 $(x_i, c_i)$ 쌍 (per-method JSD, complexity) 이다. **이
$x_i$ 가 surrogate 의 입력** 이다.

### 4.2 2단계: 50 random 3-way 측정 → response surface

200개 random architecture 의 결합 JSD $y$ 를 측정한 뒤, fixed split 으로

- **27 quantile-grid**: 각 method JSD 의 quantile $\{0.1, 0.5, 0.9\}$ 의
  $3^3 = 27$ 조합. 200 random sample 중 각 grid 점에 가장 가까운 distinct
  sample 을 Hungarian assignment 로 매칭.
- **23 maximin extras**: 27-grid 외부에서 farthest-point 로 23개 추가
  → train pool **50개**.
- **150 test** (나머지, fixed).

### 4.3 후보 surrogate

| Model | Form | Param |
|---|---|---|
| M0 | $\alpha x_W + \beta x_{KV} + \gamma x_{KVD}$ | 3 |
| M1 | M0 + intercept | 4 |
| M8 | M1 + $x_i^2$ for each $i$ | 7 |
| M9 | M1 + $x_i x_j$ for each pair | 7 |
| **M10** | M1 + 모든 squared + 모든 pairwise (full quadratic) | 10 |
| RBF cubic+linear | $\sum_j \alpha_j \|x - x_j\|^3 + \beta^\top x + b$ | nonparametric |
| RBF tps+linear | thin-plate spline + linear tail | nonparametric |
| **ARD-GP** | $k(x,x') = \sigma_f^2 \exp\bigl(-\sum_i \tfrac{(x_i-x_i')^2}{2 l_i^2}\bigr) + \sigma_n^2 \delta$ | 5 학습 |

M10 까지가 사용자가 명시한 "first-order + quadratic + interaction" 의 closed-form
parametric series, RBF / ARD-GP 는 비모수 확장.

---

## 5. Empirical Results (검증된 수치)

### 5.1 Surrogate 비교 (150 test R², JSD input)

| Model | N=27 (grid) | N=50 (grid+maximin) | 비고 |
|---|---|---|---|
| M0 (1차, no intercept) | 0.952 | **0.946** | first-order만 |
| M1 (1차, w/ intercept) | 0.952 | **0.946** | + intercept |
| M9 (M1 + interaction) | 0.955 | 0.958 | interaction 추가 |
| M8 (M1 + squared) | 0.933 | 0.972 | quadratic main |
| **M10 (full quadratic)** | 0.937 | **0.981** | full 2nd-order |
| RBF cubic+linear | 0.962 | 0.994 | nonparametric |
| RBF tps+linear | 0.984 | 0.996 | thin-plate spline |
| **ARD-GP** | 0.972 | **0.996** | anisotropic kernel |

핵심 관찰:

- **First-order additive (M0/M1) 만으로 R² ≈ 0.946** — additive 분해가 강력한
  근사임을 확인.
- **Quadratic + interaction (M10) → R² ≈ 0.981** — 단순 다항식 확장이 +0.035 향상.
- **ARD-GP / RBF tps → R² ≈ 0.996** — 50 sample 만으로 거의 완벽한 surrogate.

스크립트: [`01_surrogate_comparison.py`](01_surrogate_comparison.py),
figure: [`figures/01_surrogate_*.png`](figures/).

### 5.2 ARD-GP 내부 분석 — automatic relevance determination

ARD-GP kernel hyperparameter $(\sigma_f^2, \sigma_n^2, l_W, l_{KV}, l_{KVD})$ 는
log-marginal-likelihood 최대화로 학습된다:

$$
\log p(y\mid X)
= -\tfrac{1}{2} y^\top (K + \sigma_n^2 I)^{-1} y
  -\tfrac{1}{2}\log\det(K + \sigma_n^2 I)
  -\tfrac{n}{2}\log 2\pi.
$$

차원 $i$ 가 무관할수록 $l_i \to \infty$ 가 되어 그 차원이 자동으로 무시된다.

**(a) Length scales (50-train fit, JSD input)** — `02_ard_gp_analysis.py` 출력

| Dim | $l_i$ (raw) | $l_i / \mathrm{range}_i$ | sensitivity $1/\tilde l_i$ |
|---|---|---|---|
| **W** | 0.2867 | **0.449** | **2.23** |
| KV | 0.2594 | 2.189 | 0.46 |
| KVD | 0.2533 | 0.821 | 1.22 |

→ **W 가 KV 의 ≈4.9× 더 민감**, 순서 W > KVD > KV.

**(b) Sobol decomposition (Saltelli pick-freeze, $N=2048$ on ARD-GP)**

| Term | Value (± 95% CI from Saltelli) |
|---|---|
| $S_W$ (main) | **0.8447 ± 0.051** ★ 통계적으로 dominant |
| $S_{KV}$ | 0.0634 ± 0.017 |
| $S_{KVD}$ | 0.0407 ± 0.016 |
| $S_{T,W}$ (total) | 0.8868 ± 0.046 |
| $S_{W,KV}$ (pairwise) | 0.0066 ± 0.073 ⚠️ CI > estimate |
| $S_{W,KVD}$ | 0.0366 ± 0.075 ⚠️ CI > estimate |
| $S_{KV,KVD}$ | 0.0077 ± 0.026 ⚠️ CI > estimate |
| **$\sum_i S_i$ (additive ratio)** | **0.9488 (94.9%)** |
| **$1 - \sum_i S_i$ (interaction mass)** | **0.0512 (5.1%)** |

→ Pairwise $S_{ij}$ 들의 confidence interval 이 추정값보다 큼. 따라서 "어느
pairwise interaction 이 가장 크다" 라고 단정하기보다 **"pairwise interaction
estimates are small and statistically weak; the aggregate interaction mass
is approximately 5 %"** 가 정확한 표현이다. §7 의 Hier-WD 가 이론적으로 우월한지의
판정도 이 신뢰구간 한계 때문에 inconclusive — 실측에서 Hier-WK 가 약간 우세한
것과 일관 (§7.3).

**(c) Active subspace** — gradient covariance $\mathbb{E}[\nabla F \nabla F^\top]$ spectrum

$$
\lambda_2 / \lambda_1 = 0.0711, \qquad \lambda_3 / \lambda_1 = 0.0308.
$$

First eigenvector entries: $(W, KV, KVD) = (-0.59,\ -0.74,\ -0.33)$ (부호는
arbitrary).

> **해석 주의**: eigenvector 절댓값 기준으로는 KV component (0.74) 가 가장 큼.
> 그러나 ARD length-scale sensitivity (range-normalized) 는 $W > KVD > KV$.
> 두 결과가 어긋나는 이유는 active subspace 가 **input variance 가중**
> gradient covariance 인 반면 ARD sensitivity 는 input range 정규화한 length
> scale 이기 때문. 안전한 표현:
>
> *ARD length-scale sensitivity indicates W > KVD > KV (range-normalized).
> The active subspace spectrum is effectively one-dimensional; the leading
> direction combines W and KV variations, while ARD attributes the strongest
> local sensitivity to W after range normalization.*

스크립트: [`02_ard_gp_analysis.py`](02_ard_gp_analysis.py),
figure: [`figures/02_ard_gp_*.png`](figures/).

### 5.3 Empirical ε bounds — held-out 150 test residuals

150 test 잔차 $r = y_{\mathrm{actual}} - y_{\mathrm{pred}}$ 의 norm:

| Surrogate | $R^2_{\text{MSE}}$ | $\hat\epsilon_\infty^{\text{test}} = \|r\|_\infty$ | $\hat\epsilon_2^{\text{test}} = \mathrm{RMS}(r)$ |
|---|---|---|---|
| M1 (linear additive) | 0.9461 | 0.0590 | 0.0268 |
| M10 (full quadratic) | 0.9813 | 0.0496 | 0.0157 |
| RBF cubic+linear | 0.9942 | 0.0310 | 0.0088 |
| RBF tps+linear | **0.9955** | **0.0241** | **0.0077** |
| **ARD-GP (full)** | **0.9956** | 0.0255 | **0.0077** |
| Hier-WD-TPS-linear | 0.9896 | 0.0532 | 0.0118 |
| **Additive $\hat F_{\mathrm{add}}$ (Hoeff. 1차만)** | 0.8351 | **0.0858** | **0.0468** |

> **R² 정의 (모든 분석 스크립트 통일)**:
> $R^2 = 1 - \dfrac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}
> = 1 - \dfrac{\mathrm{SSE}}{\mathrm{SST}}$
> (scikit-learn `r2_score` 와 동일, "coefficient of determination").
> [`01`–`04`](.) 및 [v2/analysis_v2.py](../v2/analysis_v2.py) 모두 이 표준 정의 사용.
>
> **M1 R²=0.946 vs Hoeffding $\hat F_{\mathrm{add}}$ R²=0.835 차이의 원인은 정의가
> 아니라 모델 specification**:
> - **M1**: OLS 로 $y \approx \beta_0 + \beta_W z_W + \beta_{KV} z_{KV} +
>   \beta_{KVD} z_{KVD}$ 직접 fit → training MSE 직접 최소화
> - **$\hat F_{\mathrm{add}}$**: ARD-GP 를 먼저 fit (log marginal likelihood 최대화)
>   한 뒤 1D conditional expectation $g_i(z_i) = \mathbb{E}[F(z)\mid z_i] - F_0$ 를
>   Gaussian-kernel marginal smoothing (bandwidth $h = 0.15 \cdot \sigma_i$) 로 추출
>
> 두 모델 모두 additive structural form 을 공유하나 **fitting objective + GP
> regularization + smoothing bias** 가 달라 R² 가 다름. M1 은 직접 OLS 이므로
> training set 위에서는 더 tight 한 fit, $\hat F_{\mathrm{add}}$ 는 ARD-GP 에서
> 유도되어 implicit 정규화가 들어감.

**Additive 잔차 분산 비율 vs Sobol 이론 일치**:

$$
\frac{\mathrm{Var}(y_{\mathrm{actual}} - \hat F_{\mathrm{add}})}{\mathrm{Var}(y_{\mathrm{actual}})}
\;=\; 7.4\%
\quad\overset{\text{cf. 이론}}{\approx}\quad
1 - \sum_i S_i = 5.1\%
$$

이론값과 실측값이 sampling noise + finite-N + monotone calibration 차이 범위 내
일치 (2.3% gap).

**Theorem 2 + Theorem 3 통합 — surrogate 별 empirical $2\hat\epsilon$ corridor
+ monotonicity 검증 status**:

| Surrogate $\hat y$ | $\hat\epsilon_\infty^{\text{test}}$ | $2\hat\epsilon$ corridor | / $\ln 2$ | (C2) audit | Theorem 3 적용 영역 |
|---|---|---|---|---|---|
| **M0** $\beta_i z_i$ | 0.0643 | 0.1286 | 18.6 % | $\beta = [+0.61, +1.24, +0.51]$ | **GLOBAL** ✓ |
| **M1** $\beta_0 + \beta_i z_i$ | 0.0590 | 0.1180 | 17.0 % | $\beta = [+0.60, +1.08, +0.47]$ | **GLOBAL** ✓ |
| Hoeffding $\hat F_{\mathrm{add}}$ | 0.0858 | 0.1716 | 24.8 % | $h_i'(z_i) > 0$ ∀$i$ ✓ | GLOBAL (1D MC) |
| **M10** full quadratic | 0.0496 | 0.0992 | 14.3 % | $\mathcal{D}_{\mathrm{meas}}$ 99.5% ✓; box ✗ | $\mathcal{D}_{\mathrm{meas}}$ 한정 |
| **RBF cubic+linear** | 0.0310 | 0.0620 | 8.9 % | $\mathcal{D}_{\mathrm{meas}}$ 98% (mostly); box ✓ | (Lipschitz slack 고려 시) |
| **RBF tps+linear** | 0.0241 | 0.0482 | 7.0 % | $\mathcal{D}_{\mathrm{meas}}$ 99% ✓; box ✓ | $\mathcal{D}_{\mathrm{meas}}$ ≥99% ✓ |
| **ARD-GP (full)** | **0.0255** | **0.0510** | **7.4 %** | $\mathcal{D}_{\mathrm{meas}}$ 99% ✓; box ✗ | $\mathcal{D}_{\mathrm{meas}}$ 한정 |

---

**Pipeline 의 두 가지 역할 분리**:

본 NAS pipeline 에서 surrogate 는 두 역할을 한다:
1. **Candidate restriction** (Theorem 3 PF-product coverage): $\mathcal{X}_{\text{full}} \to \mathcal{C} = \prod_i \mathrm{PF}_i$ 정당화
2. **Candidate scoring** (Theorem 2 PF stability): $\mathcal{C}$ 위에서 best 점 선정

**역할 1 (restriction)** 은 **(C2) 가 globally 충족되는 surrogate** 를 써야 안전하므로
**M1 또는 Hoeffding $\hat F_{\mathrm{add}}$** 가 정확한 선택. **역할 2 (scoring)** 은
**candidate region 한정 (C2) 만 충족하면 충분** 하므로 **ARD-GP / RBF tps** 가 더
정확.

**최종 corridor 의 세 가지 표현 옵션**:

**Case A (강한 claim, candidate-relevant region 한정).** ARD-GP 가 동시에 두
역할 모두 수행. Theorem 3 의 (C2) 가 $\mathcal{D}_{\mathrm{meas}}$ 위 99% 충족
이므로 **candidate region 한정** corridor:

$$
\boxed{\;
\hat\epsilon_\infty^{\text{GP, test}} = 0.0255 \;\Rightarrow\;
2\hat\epsilon_{\mathrm{GP}} = 0.051 \quad (\approx 7.4\%\text{ of }\ln 2)
\;}
$$

> 단 이 corridor 의 *Theorem 3 derivation* 은 ARD-GP 의 (C2) 가 $\mathcal{X}_{\mathrm{full}}$
> 전체에서 충족된다고 가정. 실측은 $\mathcal{D}_{\mathrm{meas}}$ 99% 만 보장
> ($\mathcal{D}_{\mathrm{box}}$ 에서 KV/KVD 깨짐). 따라서 *Theorem 2* (surrogate
> PF stability) corridor 로만 해석하는 것이 안전.

**Case B (가장 깔끔, $\hat F_{\mathrm{add}}$ 단일 사용).** Restriction 과 scoring
모두 Hoeffding additive $\hat F_{\mathrm{add}}$ 로. (C2) global 충족 자동 보장:

$$
\boxed{\;
\hat\epsilon_\infty^{\text{add, test}} = 0.0858 \;\Rightarrow\;
2\hat\epsilon_{\mathrm{add}} = 0.172 \quad (\approx 24.8\%\text{ of }\ln 2)
\;}
$$

가장 보수적이지만 가정 없이 모든 정리가 정확히 적용.

**Case C (recommended, 두 역할 분리).** Restriction 은 M1 (또는 $\hat F_{\mathrm{add}}$)
으로 (C2) global 충족. Scoring 은 ARD-GP 로 더 정확. Budgeted regret form
(Theorem 2B) 에서 두 error 가 더해짐:

$$
y(\hat a_\tau) \;\le\; \min_{c(a) \le \tau} y(a) \;+\; 2\epsilon_{\mathrm{M1}} \;+\; 2\epsilon_{\mathrm{GP}}.
$$

Empirical 대입: $2(0.0590) + 2(0.0255) = 0.169$ (≈ 24.4% of $\ln 2$). 또는 더 보수적으로
M1 대신 $\hat F_{\mathrm{add}}$ 쓰면 $0.172 + 0.051 = 0.223$.

> *We use the additive M1 surrogate (β coefficients verified positive) to
> justify the local PF Cartesian product restriction. ARD-GP, RBF tps+linear,
> and M10 are then used for accurate candidate scoring; their
> coordinate-wise monotonicity is empirically verified ($\ge 99\%$) on the
> candidate-relevant region $\mathcal{D}_{\mathrm{meas}}$. When this audit
> passes, the smaller residual yields a tighter empirical 2$\hat\epsilon$-Pareto
> corridor for the scoring step. Otherwise, the formal candidate restriction
> guarantee remains tied to the monotone additive surrogate.*

---

⚠️ **중요한 표현 한계**: 위 모든 $\hat\epsilon^{\text{test}}$ 는 **150 random
held-out sample 에서 측정한 empirical sup-residual** 이지, 전체 $4.8 \times
10^{212}$ search space 에 대한 deterministic uniform bound 가 아니다. 즉:

> *The reported $\hat\epsilon_\infty^{\text{test}}$ is not a deterministic
> supremum over $\mathcal{X}_{\text{full}}$. It is an empirical worst-case
> residual over 150 held-out random joint samples. Therefore, the resulting
> $2\hat\epsilon$ corridor should be interpreted as **empirical evidence under
> the sampled joint distribution**, not as a formal global guarantee.*

$\hat\epsilon_2^{\text{test}}$ 는 RMS error 이므로 worst-case Pareto guarantee 에
직접 들어가지 않고 **typical / average-case residual scale** 로 해석. 더 강한
주장을 하려면 §6 Step 6 의 frontier-knee audit + Mahalanobis nearest-train
distance audit 으로 random test 가 아닌 frontier-relevant region 에서도 residual
bound 가 유지되는지 확인해야 한다.

**JSD range 주의**: base-$e$ JSD 의 이론 상한은 $\ln 2 \approx 0.693$, base-2 JSD
는 $[0, 1]$. 본 측정은 base-$e$ 이므로 $0.094$ (typical) ~ $0.172$ (worst-case)
corridor 는 $\ln 2$ 대비 13.6 % ~ 24.8 %. CE loss 또는 unbounded 변환의 경우
ε 는 normalize 후 해석해야 한다.

스크립트: [`03_pareto_combination.py`](03_pareto_combination.py).

---

## 6. Final Workflow — ε-Pareto for Full Search

```
┌──────────────────────────────────────────────────────────────┐
│ Step 1. Per-method NSGA2 search (1-axis 단독)                │
│   ─ W axis    → PF_W   (|PF_W|=733)                          │
│   ─ KV axis   → PF_KV  (|PF_KV|=321)                         │
│   ─ KVD axis  → PF_KVD (|PF_KVD|=129)                        │
│   각 PF point 에는 single-axis JSD x_i 가 함께 저장됨         │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 2. 50 actual 3-way measurement                          │
│   ─ 27-grid (quantile {0.1,0.5,0.9}^3) + 23 maximin extras   │
│   ─ y_actual = combined JSD                                  │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 3. ARD-GP fit on 50 train (input = (x_W, x_KV, x_KVD))  │
│   ─ length scales l_i ← log-marginal-likelihood              │
│   ─ Sobol: Σ S_i ≈ 0.95 (additive),  interaction ≈ 5%        │
│   ─ 150 test: R²=0.996,  ε_∞(add)=0.086,  ε_2(add)=0.047     │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 4. Surrogate ε̂ bound + (C2) audit — 150 test, 2 audits  │
│   ─ ε̂_∞ per surrogate (Theorem 2):                            │
│       M0:0.064  M1:0.059  M10:0.050  RBF-c:0.031  RBF-t:0.024 │
│       ARD-GP:0.026   F_add:0.086                              │
│   ─ (C2) audit per surrogate (Theorem 3):                     │
│       M0/M1: β_i 모두 nonneg → GLOBAL ✓                      │
│       F_add: 1D h_i monotone (38/38 corr-MC pts) → GLOBAL ✓  │
│       M10:    D_meas 99.5% ✓ / D_box VIOLATED                │
│       RBF-t:  D_meas 99% ✓  / D_box ✓                       │
│       ARD-GP: D_meas 99% ✓  / D_box VIOLATED (KV 79%/KVD 58%)│
│   (※ all ε̂ empirical on 150 test, not deterministic uniform) │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 5. Two-role split + Pareto sorting (§5.3 Case C)        │
│   ─ C = PF_W × PF_KV × PF_KVD ≈ 3.04·10⁷ candidates          │
│   Role 1 — Candidate restriction (Theorem 3 PF coverage):    │
│     M1 (β_i>0) → C 가 X_full 을 ε_M1=0.059 corridor 내 지배  │
│   Role 2 — Candidate scoring (Theorem 2 PF stability):       │
│     ARD-GP → score 의 ε_GP=0.026 corridor                    │
│   Combined budget regret (Theorem 2B):                       │
│     y(â_τ) ≤ min y(a) + 2ε_M1 + 2ε_GP = 0.169                │
│                                                               │
│   Corridor 옵션 요약 (§5.3 cases):                            │
│   • Case A: ARD-GP 단일, candidate region 한정 → 0.051       │
│   • Case B: F_add 단일, global 가정 무 → 0.172               │
│   • Case C (recommended): split = 0.169 ~ 0.223              │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 6. Frontier & support audits (강한 claim 위한 권장)      │
│   (a) Frontier-knee audit: predicted PF 의 knee/extreme       │
│       10–20개를 추가 실측 → frontier ε̂ ≈ random test ε̂ 검증  │
│   (b) Mahalanobis support audit: candidate PF C 전체에 대해   │
│       d_train(z) = min_{z_j∈X_train} ‖z − z_j‖_{Σ⁻¹} 계산하고  │
│       knee/extreme 점이 train support 내인지 확인              │
│       (extrapolation 점 → ARD-GP corridor 신뢰도 저하)        │
│   (c) 두 audit 통과 시 §5.3 Case A 의 0.051 corridor 강화 가능│
└──────────────────────────────────────────────────────────────┘
```

본 분석은 **held-out 150 random sample 의 empirical $\hat\epsilon$ + Theorems
1–3 의 결합** 으로 정당화한다. 즉 결과는 결정론적 uniform bound 가 아니라
**measured surrogate residual 하의 empirical 2$\epsilon$-Pareto evidence**.
더 강한 frontier 보장을 원하면 Step 6 의 frontier/Mahalanobis audit 으로 확장
권장.

---

### 6.1 What this method does and does not claim

> **Strong claims (직접 지지됨)**
> 1. Full joint space $4.79 \times 10^{212}$ 는 직접 탐색 불가능 (Theorem 1).
> 2. Per-method PF product $\approx 3.04 \times 10^{7}$ 가 후보로 충분 — 약
>    $1.6 \times 10^{205}$ 배 축소.
> 3. Standalone per-method loss $z_i$ 가 joint loss 를 잘 설명: M1 (linear
>    additive) $R^2 = 0.946$, M10 (quadratic) $R^2 = 0.981$, ARD-GP / RBF tps
>    $R^2 = 0.996$.
> 4. Sobol additive ratio $\sum_i S_i = 94.9 \%$ → interaction mass $\approx
>    5.1\%$ (small).
> 5. **M0/M1 의 (C2) 가 globally 충족** ($\beta_i$ 모두 strict positive)
>    → Theorem 3 PF-coverage 가 *Hoeffding additive / M1* 에 대해 정확히 성립,
>    Case B/C 는 가정 없이 적용.
> 6. Theorem 2 (surrogate stability) + Theorem 3 (PF coverage) 는 정리로 증명.
>
> **Empirical-only claims (deterministic global guarantee 가 아님)**
> 7. 보고된 $\hat\epsilon^{\text{test}}$ 는 150 random held-out sample 위의
>    empirical sup-residual 이지 $\mathcal{X}_{\text{full}}$ 전체의 uniform
>    bound 가 아니다 → 항상 "*empirical 2$\epsilon$-Pareto evidence*" 로 표현.
> 8. ARD-GP / M10 의 (C2) 는 **measured-distribution $\mathcal{D}_{\mathrm{meas}}$
>    한정** 99% 충족. Box-interior uniform fill $\mathcal{D}_{\mathrm{box}}$
>    (측정점이 sparse 한 hole 포함) 에선 깨짐 → Case A 의 좁은 0.051 corridor
>    는 measured-distribution 영역 한정.
> 9. RBF tps+linear 는 두 영역 모두 ≥99% — *practical 으로* monotone-audited
>    nonparametric scorer.
> 10. Pairwise Sobol $S_{ij}$ 는 CI > estimate. "aggregate interaction mass
>     ≈ 5%" 만 보고 (individual pair ranking 미주장).
>
> **Final summary (한 문장으로)**
> *Our method does not claim a deterministic global Pareto guarantee over the
> $4.8 \times 10^{212}$ full space. Instead, the candidate restriction
> $\mathcal{C} = \prod_i \mathrm{PF}_i$ is justified by Theorem 3 under the
> monotone additive surrogate M1 (whose OLS coefficients are verified strictly
> positive — global C2), and ARD-GP / RBF tps+linear (verified $\ge 99\%$
> coordinate-wise monotone on the candidate-relevant region $\mathcal{D}_{\mathrm{meas}}$)
> serve as accurate scorers. The resulting empirical 2$\hat\epsilon$-Pareto
> corridor lies between $0.118$ (M1-only) and $0.051$ (ARD-GP, region-restricted),
> and the joint loss is explained by the three per-method scalar summaries with
> $R^2 \approx 0.996$, with estimated interaction mass only about 5%.*

---

## 7. Supplementary — Hierarchical Surrogate Analysis (50-train constrained)

### 7.1 Motivation

§5 의 direct ARD-GP/RBF 는 50 train 샘플의 (xW, xKV, xKVD, y) 를 통째로 사용한다.
대안으로, 함수를 **2D pair surface + 1D residual** 로 명시적 분해해 학습할 수
있다 (Hoeffding 분해의 ANOVA 항을 직접 모방):

$$
\hat y_{\text{hier}}(x_W, x_{KV}, x_{KVD})
\;=\; \underbrace{f_{\text{pair}}(x_a, x_b)}_{\text{TPS + linear tail}}
   \;+\; \underbrace{g(x_c)}_{\text{1-D linear residual}}
$$

여기서 $(a,b,c)$ 는 (W, KV, KVD) 의 한 순열이고, $f_{\text{pair}}$ 는 $f_a + f_b
+ f_{ab}$, $g$ 는 $f_c$ 를 흡수한다 (Hoeffding 분해의 main + pairwise 항).
$f_{ac}, f_{bc}, f_{abc}$ 는 무시된다.

**공정 비교 원칙**: 본 §7 에서는 **train_pool 50 sample 외의 데이터를 일체
사용하지 않는다**. pair 측정 ($f_{\text{pair}}$ 학습) 과 residual 측정 ($g$ 학습)
모두 50-train 의 부분집합으로 제한 — 이렇게 해야 direct ARD-GP (50 3-way) 와
sample budget 이 동등.

> 주의: 외부 pair-only CSV (200 추가 sample) 를 사용하면 pair surface 가
> 더 정확해지지만, "pair 측정이 cheap & abundant" 라는 별도 가정에 의존한다.
> 본 절은 **동일 budget 가정** (B ≤ 50, train_pool 내부) 하의 결과만 다룬다.

### 7.2 Setup

세 hier 구성 (Sobol pairwise $S_{ab}$ — §5.2 참조):

| Config | Pair surface | Residual axis | Sobol $S_{ab}$ |
|---|---|---|---|
| Hier-WK | $(W, KV)$ | KVD | 0.0066 |
| **Hier-WD** | $(W, KVD)$ | KV | **0.0366 (largest)** |
| Hier-KD | $(KV, KVD)$ | W | 0.0077 |

이론상 Hier-WD 가 가장 큰 pairwise interaction 을 명시적으로 흡수하므로
hier 중 1위.

**Surrogate input** (§5 direct ARD-GP/RBF 와 동일): $(x_W, x_{KV}, x_{KVD}) =$
**per-method JSD vector** (각 PF 위 nearest-complexity 점의 JSD). Hier 의 pair
surface 도 $(x_a, x_b)$ = 2D JSD pair, residual 도 $x_c$ = 1D JSD 를 입력으로
받음. 즉 hier 와 direct 가 **동일 input space (3D JSD)** 에서 surrogate 구조만
다르게 비교됨.

**Surrogate models 사용**:
- **Pair surface $f_{\text{pair}}(x_a, x_b)$ (2D)**: 두 kernel 비교
  - **TPS + linear tail** — bending energy $\int (f_{xx}^2 + 2 f_{xy}^2 + f_{yy}^2) \,dxdy$ 최소화
  - **RBF cubic + linear tail** — $\phi(r) = r^3$ basis
- **Residual $g(x_c)$ (1D)**: 세 model 비교
  - linear (deg=1), quadratic (deg=2), RBF cubic + linear tail

**Sample 분배 (50-budget)**:
각 budget $B$ 에서 train_pool 50 을 두 부분집합으로 분할:
- $n_{\text{pair}}$ samples → $f_{\text{pair}}$ 학습
- $n_{\text{3way}}$ samples → residual $g$ 학습 ($x_c, y - f_{\text{pair}}$)
- 제약: $n_{\text{pair}} + n_{\text{3way}} \le 50$
- Allocation = `disjoint` (분리 subset) 또는 `overlap` (모두 50)

30 seed median 보고.

### 7.3 종합 sweep (4 design choices × 12 n_pair)

Design space: (pair × allocation × pair_kernel × resid_model × $n_{\text{pair}}$).
30 seed median, 150 test 평가.

#### (a) Pair kernel 비교 — TPS vs RBF cubic (best alloc·resid·n_p per pair)

| Pair | TPS pair | RBF cubic pair | Δ |
|---|---|---|---|
| **Hier-WK** | **0.8745** | 0.7721 | $-0.102$ |
| Hier-WD | **0.8590** | 0.7618 | $-0.097$ |
| Hier-KD | $-0.098$ | $-0.797$ | $-0.699$ |

**TPS 가 모든 pair config 에서 RBF cubic 보다 우월**. 이유:

- **TPS** 는 bending-energy 최소화 surface
  $\arg\min_f \int \bigl(f_{xx}^2 + 2 f_{xy}^2 + f_{yy}^2\bigr)\,dx\,dy$
  → smooth interpolation, sparse 2D data 에서 안정.
- **RBF cubic** 은 $\phi(r) = r^3$ basis function 으로 빠르게 자라는 형태
  → 2D 에서 extrapolation 발산. 특히 Hier-KD ($-0.80$) 는 W 정보 누락 +
  cubic 의 발산이 결합되어 catastrophic.

#### (b) Pair ordering 비교 — best per pair (best surrogate 선택 후)

| Pair | best alloc | best resid | best $n_p$ | R² | $\epsilon_\infty$ | $\epsilon_2$ | $S_{ab}$ (theory) |
|---|---|---|---|---|---|---|---|
| **Hier-WK** | disjoint | quad | 14 | **0.8745** | 0.1109 | 0.0408 | 0.0066 |
| Hier-WD | disjoint | lin | 38 | 0.8590 | 0.1122 | 0.0433 | **0.0366** |
| Hier-KD | disjoint | quad | 6 | **−0.103** | 0.2755 | 0.1209 | 0.0077 |

(모두 pair_kernel = TPS)

**이론 vs 실측 어긋남**: Sobol 이론은 $S_{W,KVD}=0.0366$ 가 가장 크므로
**Hier-WD 가 최선** 이라 예측. 그러나 50-budget 실측에서는 **Hier-WK 가 약간 우월**
($\Delta R^2 = +0.016$).

**원인 분석** — finite-sample effect:
$S_{ab}$ 가 클수록 pair surface 곡률이 크고, TPS fit 에 더 많은 sample 필요.
실제로 Hier-WD 최적 $n_{\text{pair}}=38$ vs Hier-WK $n_{\text{pair}}=14$. Hier-WD
는 50-budget 의 76% 를 pair 에 써 residual 은 12 sample 만 남고, Hier-WK 는
28% 만 pair 에 써 residual 에 36 sample. **더 많은 residual sample 이
Hier-WK 의 안정성에 기여**.

**Hier-KD 의 catastrophic 실패** ($R^2 \approx -0.1$): pair=(KV, KVD) 가
dominant axis $W$ ($S_W = 0.845$) 정보를 누락하고, 1D residual model 로 W 의
비선형 main effect 를 표현 불가능. **Hier 설계 필수조건: dominant main effect
가 pair 에 포함되어야 함**.

#### (c) Residual model 비교 (TPS pair, disjoint, best $n_p$)

| Pair | linear | quadratic | RBF cubic 1D |
|---|---|---|---|
| Hier-WK | 0.8693 | **0.8745** | 0.7242 |
| Hier-WD | **0.8590** | 0.8544 | 0.6816 |
| Hier-KD | $-0.098$ | $-0.103$ | $-1.745$ |

- **linear ≈ quadratic** (Hier-WK 만 quad 가 +0.005 미세 우세 → KVD 의 약한
  비선형성)
- **RBF cubic residual 일관 악화** ($\Delta R^2 = -0.10 \sim -0.15$). 이유:
  1D residual fit 에 sample 12 ~ 36 만 있는 상황에서 cubic basis 가 overfit.
  Polynomial 이 더 안정.

#### (d) Sample allocation 비교 (TPS pair, deg=1)

| Pair | disjoint best | overlap (all 50) |
|---|---|---|
| Hier-WK | 0.8693 ($n_p=14$) | 0.7688 |
| Hier-WD | 0.8590 ($n_p=38$) | 0.7568 |
| Hier-KD | $-0.098$ ($n_p=18$) | $-1.295$ |

- **overlap = 최악** (모든 pair 에서 R² 8–10 % 손실) — 50 sample 이 모두 pair
  학습에 사용되면 TPS 가 train residual 을 0 에 fit → residual 학습 신호 소멸.

#### (e) $n_{\text{pair}}$ 민감도 (TPS pair, disjoint, deg=1)

| $n_{\text{pair}}$ | 6 | 10 | 14 | 18 | 22 | 26 | 30 | 34 | 38 | 42 |
|---|---|---|---|---|---|---|---|---|---|---|
| **WK** | 0.821 | 0.857 | **0.869** | 0.831 | 0.823 | 0.831 | 0.827 | 0.826 | 0.830 | 0.825 |
| **WD** | 0.777 | 0.794 | 0.844 | 0.837 | 0.815 | 0.834 | 0.840 | 0.841 | **0.859** | 0.846 |
| KD | $-0.12$ | $-0.22$ | $-0.18$ | $-0.10$ | $-0.17$ | $-0.24$ | $-0.37$ | $-0.42$ | $-0.41$ | $-0.58$ |

- **Hier-WK**: peak $n_p = 14$, 이후 plateau (≈ 0.83). pair surface 가 단순해
  14 sample 로 수렴.
- **Hier-WD**: peak $n_p = 38$, pair surface 곡률 크므로 더 많은 sample 필요.
- **Hier-KD**: $n_{\text{pair}}$ 무관하게 catastrophic.

### 7.4 결론

| 비교 | 결과 |
|---|---|
| **Direct ARD-GP @ B=50** | **R² = 0.9956**, $\epsilon_\infty = 0.026$ |
| **Best hier (Hier-WK, disjoint, deg=2, $n_p=14$)** | R² = 0.8745, $\epsilon_\infty = 0.111$ |
| Δ R² | $-0.121$ — 동일 budget 하 hier 가 명백히 열등 |

**이론적 정렬**:
- pair 안에 dominant axis ($W$, $S_W=0.85$) 를 반드시 포함해야 함 (Hier-KD 가
  fail 하는 이유)
- 큰 $S_{ab}$ 는 이론적 우위지만 finite-sample 에서는 pair fit 비용으로 상쇄
  → Hier-WK ≈ Hier-WD (실측 동등)
- 따라서 hier 설계 시 **dominant main effect 가 pair 에 포함되어야 함**
  (필수조건), $S_{ab}$ 크기는 budget 충분할 때만 의미 (sufficient 조건)

**ε-Pareto 관점** (JSD 이론 range $\ln 2 \approx 0.693$ 기준):
$$
\epsilon_\infty^{\text{Hier-WK}} = 0.111
\;\Rightarrow\; 2\epsilon_\infty = 0.222 \;\;(\approx 32\%\text{ of }\ln 2)
$$
direct ARD-GP 의 $0.051\;(\approx 7.4\%)$ 보다 4× 넓고, Hoeffding additive
$y_{\mathrm{add}}$ 의 $0.172\;(\approx 24.8\%)$ 보다도 넓다.
**50-constrained hier 는 theorem 측면에서도 direct 보다 약함** —
sample budget 이 동등할 때는 implicit anisotropic 학습 (ARD-GP) 이 explicit
ANOVA 분해 모방 (TPS+poly) 을 압도.

> 외부 cheap pair-only 측정 source (예: 200 sample 의 (W, KVD) pair-only CSV) 가
> 존재해야 hier-WD 가 실용적 의미. 본 README 는 동일 budget 가정 (50 train 내)
> 하의 결과만 다룸.

> 검증 노트: ARD-GP 는 §5/01 과 동일 hyperparameter setting (`length_scale_bounds
> =(1e-4,1e4)`, `n_restarts=20`, normalize_y, alpha=1e-8) 사용. N=50 에서 stable
> 한 fit 으로 raw length scales $[l_W, l_{KV}, l_{KVD}] = [0.2867, 0.2594, 0.2533]$
> ([§5.2 (a)](#52-ard-gp-내부-분석--automatic-relevance-determination) 의 값) 가
> 본 README 의 일관된 reference. (이전 일부 archive 보고서에서 다른 bound 설정
> 으로 fit 한 값들과 혼동되지 않도록 주의.)

스크립트: [`04_hier_surrogate_analysis.py`](04_hier_surrogate_analysis.py),
figures: [`figures/04_hier_*.png`](figures/).

---

## 디렉토리 구조

```
analysis/v3/
├── README.md                     ← 본 narrative + 수학 분석
├── 01_surrogate_comparison.py    ← §4–5.1: surrogate model 비교
├── 02_ard_gp_analysis.py         ← §5.2–5.3: ARD-GP 내부 + ε bounds
├── 03_pareto_combination.py      ← §5.3: empirical 2ε-Pareto evidence
├── 04_hier_surrogate_analysis.py ← §7: hier surrogate 보조 분석
├── archive/
│   ├── verify_main_effect_monotonicity.py     ← §2.6 (A): 1D h_i monotone
│   ├── verify_ardgp_coordinate_monotonicity.py ← §2.6 (B): ARD-GP 3D coord
│   ├── verify_monotonicity_all_surrogates.py  ← §2.6: 6 surrogate × 3 axis
│   └── ...
└── figures/
    ├── 01_surrogate_*.png         (3개)
    ├── 02_ard_gp_*.png            (5개)
    ├── 03_pareto_combination_*.png (2개)
    ├── 04_hier_*.png              (6개)
    ├── verify_monotonicity.png       ← 1D main effect g_i + gradient
    ├── verify_ardgp_monotonicity.png ← ARD-GP coord-wise on D_meas/D_box
    └── verify_monotonicity_all.png   ← 6 surrogate × 3 axis gradient hist
```

---

## 핵심 수치 요약 (검증됨, 50/150 fixed split)

| 측정 | 값 |
|---|---|
| Full joint search space $|\mathcal{X}_{\mathrm{full}}|$ | $3^{224} \cdot 9^{64} \cdot 5^{64} \approx 4.8 \times 10^{212}$ |
| PF cardinalities | $|PF_W|=733,\ |PF_{KV}|=321,\ |PF_{KVD}|=129$ |
| PF combination size $|\mathcal{C}|$ | $\approx 3.04 \times 10^{7}$ |
| Sobol additive ratio $\sum_i S_i$ | **0.9488 (94.9%)** |
| Total interaction $1 - \sum_i S_i$ | **0.0512 (5.1%)** |
| Pairwise Sobol (CI > estimate) | $S_{W,KV}=0.007 \pm 0.073,\ S_{W,KVD}=0.037 \pm 0.075,\ S_{KV,KVD}=0.008 \pm 0.026$ |
| Active subspace $\lambda_2/\lambda_1$ | 0.0711 (≈ 1D) |
| ARD length-scale sensitivity (range-norm) | $W > KVD > KV$ |
| **ARD-GP $R^2$ (50-train, 150-test)** | **0.9956** |
| RBF tps+linear $R^2$ | 0.9955 |
| RBF cubic+linear $R^2$ | 0.9942 |
| M10 (full quadratic) $R^2$ | 0.9813 |
| M1 (1차 additive) $R^2$ | 0.9461 |
| Additive $\hat F_{\mathrm{add}}$ (Hoeffding) $R^2$ | 0.8351 |
| (모든 $R^2$ = $1 - \mathrm{SSE}/\mathrm{SST}$, scikit-learn `r2_score`) |  |
| **$\hat\epsilon_\infty^{\text{test}}$ (M1 on 150 test)** | 0.0590 |
| **$\hat\epsilon_\infty^{\text{test}}$ (ARD-GP on 150 test)** | **0.0255** (empirical, not uniform) |
| **$\hat\epsilon_\infty^{\text{test}}$ (additive on 150 test)** | **0.0858** (empirical, not uniform) |
| **$\hat\epsilon_2^{\text{test}}$ (additive on 150 test)** | 0.0468 (RMS, typical scale) |
| Empirical $\mathrm{Var}(r)/\mathrm{Var}(y)$ vs Sobol $1-\sum S_i$ | 7.4 % vs 5.1 % (gap 2.3 %) |
| **(C2) audit (Thm 3) — M0/M1 OLS β** | $\beta_W = +0.60$, $\beta_{KV} = +1.08$, $\beta_{KVD} = +0.47$ — all $+$, **GLOBAL** ✓ |
| **(C2) audit — Hoeffding $\hat F_{\mathrm{add}}$ 1D gradients** | $\nabla g_W \in [+0.189, +1.189]$, $\nabla g_{KV} \in [+0.372, +0.925]$, $\nabla g_{KVD} \in [+0.084, +0.681]$ — **GLOBAL** ✓ |
| **(C2) audit — ARD-GP coord-wise** | $\mathcal{D}_{\mathrm{meas}}$ 99% ✓; $\mathcal{D}_{\mathrm{box}}$ KV 79%/KVD 58% ✗ (region-restricted) |
| **(C2) audit — RBF tps+linear** | $\mathcal{D}_{\mathrm{meas}}$ 99% ✓; $\mathcal{D}_{\mathrm{box}}$ ✓ |
| **(C2) audit — M10 full quadratic** | $\mathcal{D}_{\mathrm{meas}}$ 99.5% ✓; $\mathcal{D}_{\mathrm{box}}$ ✗ |
| **2$\epsilon$-Pareto corridor (Case A: ARD-GP scorer, region-restricted)** | $2\hat\epsilon = 0.051$ (≈ 7.4 % of $\ln 2$) |
| **2$\epsilon$-Pareto corridor (Case B: $\hat F_{\mathrm{add}}$ single, global)** | $2\hat\epsilon = 0.172$ (≈ 24.8 % of $\ln 2$) |
| **2$\epsilon$-Pareto corridor (Case C: M1 + ARD-GP split, recommended)** | $2(\epsilon_{\mathrm{M1}} + \epsilon_{\mathrm{GP}}) = 0.169$ |

> 모든 ε 값은 **150 random held-out sample 위의 empirical 측정** 이며, 전체
> $4.8 \times 10^{212}$ search space 위의 deterministic uniform bound 가 아니다.
> 더 강한 frontier 보장을 원할 경우 §6 Step 6 의 frontier-knee audit 권장.

→ Interaction ≈ 5% 로 작아 **per-method PF Cartesian 조합이 사실상 optimal
Pareto 와 동치**이며, ARD-GP 가 50 sample 만으로 R²=0.996 의 정밀도로 확인.
