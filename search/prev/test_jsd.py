import torch
import torch.nn as nn
from torch.distributions import Categorical

torch.manual_seed(0)

B, C = 5, 7                      # batch, classes
logits_p = torch.randn(B, C)     # 모델 P의 로짓
logits_q = torch.randn(B, C)     # 모델 Q의 로짓

# ----- 확률 및 로그확률 -----
logP = logits_p.log_softmax(-1)                  # log P
logQ = logits_q.log_softmax(-1)                  # log Q
P = logP.exp()                                   # P
Q = logQ.exp()                                   # Q

eps = 1e-12
M = 0.5 * (P + Q)                                # M = (P + Q)/2
logM = (M.clamp_min(eps)).log()

# ----- 1) KL(P||M) 수학적 정의로 계산 -----
kl_pm_math = (P * (logP - logM)).sum(dim=-1).mean()

# ----- 2) KL(P||M) KLDivLoss(log_target=True)로 계산 -----
# KLDivLoss(input=logM, target=logP)  => KL(P||M)
kldiv = nn.KLDivLoss(reduction='batchmean', log_target=True)
kl_pm_torch = kldiv(logM, logP)
kl_mp_torch = kldiv(logP, logM)

print(f"KL(P||M) math:  {kl_pm_math.item():.8f}")
print(f"KL(P||M) torch: {kl_pm_torch.item():.8f}")
print(f"KL(M||P) torch: {kl_mp_torch.item():.8f}")

# ----- 3) distributions API로도 cross-check -----
# Categorical은 KL(Cat(P) || Cat(M)) = sum P (logP - logM)
kl_pm_dist = torch.distributions.kl_divergence(
    Categorical(probs=P), Categorical(probs=M)
).mean()
print(f"KL(P||M) torch.distributions: {kl_pm_dist.item():.8f}")

# ===== JSD(P||Q) 검증 =====
# JSD = 0.5*(KL(P||M) + KL(Q||M))
kl_qm_math = (Q * (logQ - logM)).sum(dim=-1).mean()
kl_qm_torch = kldiv(logM, logQ)
kl_qm_dist = torch.distributions.kl_divergence(
    Categorical(probs=Q), Categorical(probs=M)
).mean()

jsd_math = 0.5 * (kl_pm_math + kl_qm_math)
jsd_torch = 0.5 * (kl_pm_torch + kl_qm_torch)
jsd_dist  = 0.5 * (kl_pm_dist + kl_qm_dist)

print(f"JSD math:  {jsd_math.item():.8f}")
print(f"JSD torch: {jsd_torch.item():.8f}")
print(f"JSD torch.distributions: {jsd_dist.item():.8f}")

# # ===== 선택: bit 단위로 보고 싶다면 =====
# import math
# print(f"JSD (bits): { (jsd_torch / math.log(2)).item():.8f }")