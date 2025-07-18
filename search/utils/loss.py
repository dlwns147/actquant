import torch
import torch.nn as nn
from torch.nn import functional as F

class JSD(nn.Module):
    def __init__(self, tau=1., reduction='batchmean', ignore_index=-100):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)
        # self.kl = nn.KLDivLoss(reduction='sum', log_target=True)
        self.tau = tau
        self.ignore_index = ignore_index

    def forward(self, p: torch.tensor, q: torch.tensor, mask=None):
        p, q = p[mask], q[mask]
        p, q = (p / self.tau).log_softmax(-1), (q / self.tau).log_softmax(-1)
        m = (0.5 * (p + q))
        return 0.5 * (self.kl(m, p) + self.kl(m, q))
            
        # p, q = (p / self.tau).log_softmax(-1), (q / self.tau).log_softmax(-1)
        # m = (0.5 * (p + q))
        # jsd = 0.5 * F.kl_div(m, p, reduction='none', log_target=True).sum(dim=-1) + F.kl_div(m, q, reduction='none', log_target=True).sum(dim=-1)
        # return (jsd * mask.flatten()).sum()

        # mask 적용
        # jsd = jsd * mask
        # loss = jsd.sum() / (mask.sum() + 1e-8)
