# PROMPT — AWQ 기반 2차 탐색을 예측기-주도 "수정 탐색(correction search)"으로 재설계

아래 프롬프트를 새 세션에 그대로 붙여넣어 진행한다.

---

## 목표

/NAS/SJ/actquant/search 리포의 second_search.py(joint W×eff_kvbits NAS)와 post_search.py(최종 선택)를
AWQ 레짐(실측 arch당 ~502s, 총 예산 수백 evals)에 맞게 **예측기-주도**로 재설계·구현·검증한다.
핵심 방향 4가지: (A) 측정 배치 규칙(acquisition) 도입, (B) y_hqq 포화-측정 + δ_W 잔차 예측기,
(C) 탐색 공간 축소(correction search), (D) post_search 예측-front + 적응형 verify.
Phase 0→5 순서로 진행하되 각 Phase의 판정 게이트를 통과해야 다음으로 간다.
막히면 결정 로그를 남기고 독립적인 다음 단계로 진행한다(질문 대기 금지).

## 컨텍스트 — 이미 실측으로 확정된 사실 (재검증·재도출 금지)

파이프라인: 1차 per-axis NAS(search.py, HQQ) → 2차 joint NAS(second_search.py, NSGA-III,
1차 블록 재조합 + BandTable P1/P2 + L2 freeze) → post_search.py `--second_expr`(실측-only 정렬
+ `--select_measured_best`/`--verify_topk` racing). AWQ 모드 = `--w_method awq --eval_workers N`
(utils/awq_pool.py AWQEvalPool, 4 GPU 워커, 시간당 ~28 evals). 측정 캐시 = `--seed_results`
(save/awq_alloc_flip, *specs*.json + *results*.jsonl, 프로토콜 일치 필수).

확정 사실:
1. global OOS ρ는 ~0.99로 포화(비트축이 설명) — 진짜 문제는 **within-cell**(같은 (wbits,eff_kvbits)
   셀 내 allocation 순위) ρ 0.3~0.6. global 지표로 모델을 고르지 말 것.
2. HQQ→AWQ 전이붕괴 지도(셀내 순위 ρ): w3.28 **0.78** / w3.0 **0.29** / w2.8 **0.15**;
   저KV 셀은 0.68(무해). 붕괴는 W축 특이.
3. δ = y_awq − y_hqq는 **KV-독립 ≈ 순수 W 함수** (AWQ×KV 상호작용 작음).
4. round-0의 δ̂ 셀내 ρ는 0.215(약함) — 단 이는 분해 구조 없이 추정한 수치.
5. LCO 외삽 실패(ρ −0.44) → **최종 선택 셀에는 AWQ 실측 앵커 필수**. 예측-only 선택 금지.
6. surrogate input 실측(셀-평균 OOS ρ, 88/538-arch AWQ 아카이브): genome-rbf@N=88 **0.136 사망**;
   feat15d global 0.904 ≈ y_hqq raw 입력 0.911; 셀내는 hist N=100 .42 / N=200 .57 / N=430 .54,
   selfpls N=200 .51 / N=430 .60(교차점 N≈200), feat .46, plstyp .63.
7. hist+ridge(순수 additive 회귀)는 셀내 실패(.14–.29) — 셀내 순위는 상호작용 잔차에 있음.
   (측정-marginal 기반 additive는 미판별 — Phase 5 게이트.)
8. HQQ elite 합의: 224 W-셀 중 ~196 합의, 쟁점 ~28 (tests/wbits_layer_importance.py).
   L2 freeze(`--agree_frac`)로 이미 코드화되어 있음.
9. verify_topk 3~5 racing이 전 셀에서 AWQ-best 회수(round-0 PASS) — racing은 신뢰 가능한 도구.
10. al_frac(global AL quota)은 **HQQ 값싼-eval 레짐**에서 음수 판정으로 제거됨. AWQ 레짐에
    자동 이전되는 결론은 아니나, σ 캘리브레이션 미확인 acquisition은 같은 실패를 반복할 수 있음.
11. utils/select.py subset_select가 유일하게 살아남은 down-selector(edge 유지 29/30)지만
    **comp-geometry only** — 예측값/불확실성을 전혀 쓰지 않음 (second_search.py `_downselect`).
12. 기존 자산: save/awq_alloc_flip(88-arch seed), tests/awq_alloc_flip/*.py(fold 하네스:
    surrogate_input_check.py, embedding_input_check.py, selfpls_check.py, hist_input_check.py,
    analyze_round0.py), AWQ 프로덕션 런(save/ 아래 `2607151504` 태그로 검색, 450 evals).
    538-arch 아카이브 로딩 방법은 embedding_input_check.py 참조.

측정 프로토콜(모든 신규 AWQ 실측이 이것과 일치해야 seed_results 재활용 가능):
wikitext2, seqlen 2048, n_sample 128, JSD, stride 128, prefill_prompt, last_tokens 512,
attn_sink는 기존 런과 동일, w_method awq.

환경 주의: **백그라운드 셸은 반드시 리포 루트(cd /NAS/SJ/actquant/search) 후 실행**(상대경로
의존 — 과거 프로덕션 기동 즉사 원인). git 없음 → 수정 전 prev/에 원본 백업. GPU 4장.
seed_results 디렉토리에 비포맷 파일 넣지 말 것(*specs*.json 전수 글롭됨).

## Phase 0 — 노이즈 천장 감사 (~30 AWQ evals, ~4h)

- 목적: 셀내 y-spread 대비 측정 노이즈 → **도달 가능한 셀내 ρ 상한** 확정.
- 주의: 같은 arch를 같은 캘리브 세트로 재평가하면 결정론적이라 노이즈가 안 잡힌다.
  노이즈 원천은 **캘리브 샘플 추출**이므로, evaluator의 캘리브 로더 시드를 바꿔
  (코드 확인 후 필요시 시드 인자 노출) 같은 arch를 서로 다른 캘리브 드로우로 2–3회 측정.
- 설계: 프로덕션 아카이브에서 3–4개 셀 × 3–4 archs × 2–3 드로우. σ_noise와
  셀내 순위의 드로우-간 안정성(Spearman)을 함께 보고.
- 판정 게이트: 추정 ρ_max가 현재 selfpls 셀내 ρ(~0.6)+0.05 이하면 **예측기 고도화 중단**,
  Phase 2(acquisition)와 Phase 4(racing)만 진행.

## Phase 1 — 오프라인 예측기 검증 (HQQ 실측만, AWQ 0 evals)

1a. **y_hqq 페어링**: 538-arch AWQ 아카이브의 전 archs를 HQQ로 실측(arch당 수십 초,
    1 GPU 반나절). 별도 캐시에 저장(AWQ seed 디렉토리와 분리).
1b. **모델 비교** (tests/awq_alloc_flip fold 프로토콜 재사용, 셀-평균 OOS ρ, 저-W 밴드 분리 보고):
    - baseline: selfpls, hist (기존 수치 재현으로 하네스 검증부터)
    - M1: y_hqq(실측) + comp 만
    - M2: y_hqq(실측) + **δ_W 잔차 GP** (입력 = W-half hist feature; KV 풀링으로 표본 공유)
    - M3: M2 + **같은-셀 차분 학습** (pairwise/difference; |Δy|/σ_noise 가중 — Phase 0의 σ 사용;
      Hamming 거리별 성능 곡선도 보고)
- 판정 게이트: M2/M3가 selfpls 대비 셀내 ρ **+0.1 이상**, 특히 w≤3.0 붕괴 밴드에서 개선
  → Phase 3의 예측기로 채택. 아니면 selfpls 유지, 개선은 acquisition/racing에서만.

## Phase 2 — 루프 구조 변경 (second_search.py)

1. iter당 실측 수를 8–16으로 낮춘 소배치 운용이 가능하도록 확인/조정(n_iter 인자 활용,
   iterations 증가와 조합). 배치마다 refit되는지 확인.
2. `_downselect` 확장: **geometry 쿼터(기존 subset_select) + decision 쿼터** 혼합.
   decision 쿼터 = 셀별 "이 측정이 최종 선택 오류 확률을 얼마나 줄이나"(1안: 셀별
   predicted-best와 measured-best gap × 예측 불확실성; 2안: jackknife 앙상블 Thompson).
   선행 조건: k-fold z-score로 σ 캘리브레이션 확인(80–95% coverage) — 실패 시 rank 기반만.
3. **셀 앵커 보장**: budget box grid의 최종 후보 셀마다 최소 k_anchor(기본 3) 실측 보장.
   부족 셀 충족이 decision 쿼터보다 우선(LCO 가드).
4. 모든 새 동작은 플래그 opt-in(기본값 = 기존 동작 보존). HQQ 모드 스모크
   (iterations 2, n_iter 8)로 회귀 없음 확인(hv/coverage 로그 비교).

## Phase 3 — HQQ rung + 밴드 라우팅 (프로덕션 적용)

- NSGA shortlist(iter당 100–200 genome)를 AWQ 실측 전에 **HQQ로 전수 실측**하는 중간 rung.
  구현: 워커 1개를 HQQ evaluator 전담으로 배치(또는 유휴 슬롯 활용). y_hqq 캐시 별도 관리.
- **라우팅 규칙 = 전이붕괴 지도**: w-밴드 ρ 높은 영역(≥3.2)은 HQQ 순위로 후보 컷(AWQ 절약),
  붕괴 밴드(w<3.0)는 HQQ rung 스킵하고 AWQ 예산 집중.
- 예측기 = Phase 1 승자(y_hqq 실측 feature + δ_W 잔차).
- 탐색 공간 축소: AWQ 모드에서 `--agree_frac`을 조여 자유 차원을 쟁점 셀 수준(~28–60)으로
  제한하는 설정을 기본 권장값으로 캘리브레이션.

## Phase 4 — post_search 예측-front + 적응형 verify (post_search.py select_joint 확장)

- `--second_expr` 아카이브에 더해 예측기로 확장한 **가상 후보**(BandTable/블록-곱 셀내 생성)를
  셀별 랭킹에 포함. 기존 `--select_measured_best` 골격 위에 **적응형 racing**:
  고정 top-5 대신 "미검증 최선 후보의 하한(셀-조건부 Mondrian conformal 또는 앙상블
  quantile)이 측정 최선을 못 이길 때까지" AWQ verify 추가.
- 하드 가드: 실측 앵커 없는 셀에서는 예측 후보 선택 금지(LCO).

## Phase 5 (조건부) — contested-dim AWQ marginal 백본

- Phase 1에서 additive 신호가 셀내에 살아있다는 판별이 나올 때만:
  쟁점 ~28 셀 × 대안 옵션의 per-module marginal을 AWQ로 직접 실측(~50–100 evals)
  → additive DP(MCKP) 백본 + GP 잔차. 저비트 극단은 racing이 맡는 하이브리드.

## 산출물 규칙

- Phase마다: 판정 수치 + 채택/기각 결정 로그를 tests/awq_correction_search/ 아래
  스크립트 + README.md로 남긴다(기존 tests/awq_alloc_flip 스타일).
- 코드 변경 전 prev/ 백업, 스모크 PASS 후 프로덕션.

## 하지 말 것

- n_doe 512급 AWQ 실측 DOE / geometry-only downselect 유지 / 캘리브 안 된 σ로 EI·EHVI /
  예측-only 최종 선택 / global ρ 기준 모델 선택 / genome-rbf 소표본 학습 /
  seed 디렉토리 오염 / 리포루트 cd 없이 백그라운드 기동.
