import torch
from torch import nn


class HyperbolicDynamics(nn.Module):
    def __init__(self, d_h, hidden=256, c=1.0):
        super().__init__()
        self.c = c
        self.net = nn.Sequential(
            nn.Linear(d_h, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_h),
        )
    def exp0(self, v, c):  # inline expmap0
        import math
        sc = math.sqrt(c)
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.tanh(sc*n/2)/(sc*n) * v
    def mobius_add(self, x, y, c):
        x2 = (x*x).sum(-1, keepdim=True); y2 = (y*y).sum(-1, keepdim=True); xy = (x*y).sum(-1, keepdim=True)
        num = (1+2*c*xy + c*y2)*x + (1-c*x2)*y
        den = 1 + 2*c*xy + (c**2)*x2*y2
        return num / den.clamp_min(1e-6)
    def log0(self, x, c):
        import math
        sc = math.sqrt(c)
        n = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        arg = (sc*n).clamp(max=1-1e-6)
        return (2/sc) * torch.atanh(arg) / n * x
    def forward(self, h_t):
        # predict small tangent step and move on ball
        v = self.net(self.log0(h_t, self.c))
        delta = self.exp0(0.5 * v, self.c)  # scale step
        return self.mobius_add(h_t, delta, self.c)