from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import expmap0, logmap0, _sqrt_c  # keep your existing utilities

class EuclidToHypAlign(nn.Module):
    """
    Build attention weights in Euclidean space between current ρ(G_l) and demo ρ(G_l),
    then aggregate demo SK (hyperbolic) embeddings into current hyperbolic using a
    Poincaré-ball weighted barycenter (no linear mixing).

    Shapes:
      curr_rho  : [B, A, d_e]
      demo_rho  : [B, N, L, A, d_e]
      demo_hyp  : [B, N, L, d_h]   (Poincaré ball, curvature c)
      demo_mask : [B, N, L]  (1 valid, 0 pad). Optional.

    Returns:
      curr_hyp  : [B, d_h]
      attn      : [B, H, N, L]  (softmax weights per head; H=1 if single head)
    """
    def __init__(
        self,
        d_e: int,
        d_h: int,
        heads: int = 4,
        c: float = 1.0,
        temperature: Optional[float] = None,
        ln_euclid: bool = True,
    ):
        super().__init__()
        assert heads >= 1 and d_e % heads == 0, "d_e must be divisible by heads"
        self.d_e = d_e
        self.d_h = d_h
        self.H = heads
        self.c = float(c)
        self.tau = temperature

        self.ln_q = nn.LayerNorm(d_e) if ln_euclid else nn.Identity()
        self.ln_k = nn.LayerNorm(d_e) if ln_euclid else nn.Identity()

        self.hd = d_e // heads
        self.Wq = nn.Linear(d_e, heads * self.hd, bias=False)
        self.Wk = nn.Linear(d_e, heads * self.hd, bias=False)

    def forward(
        self,
        curr_rho: torch.Tensor,        # [B, A, d_e]
        demo_rho: torch.Tensor,        # [B, N, L, A, d_e]
        demo_hyp: torch.Tensor,        # [B, N, L, d_h]
        demo_mask: Optional[torch.Tensor] = None,  # [B, N, L]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, A, d_e = curr_rho.shape
        Bk, N, L, Aa, d_e2 = demo_rho.shape
        Bh, Nh, Lh, d_h = demo_hyp.shape
        assert (Bk, Bh) == (B, B) and (N, Nh) == (N, N) and (L, Lh) == (L, L)
        assert Aa == A and d_e2 == d_e and d_h == self.d_h

        # Project to multi-head spaces
        q = self.Wq(self.ln_q(curr_rho))                 # [B, A, H*hd]
        k = self.Wk(self.ln_k(demo_rho))                 # [B, N, L, A, H*hd]

        q = q.view(B, A, self.H, self.hd).permute(0, 2, 1, 3)          # [B, H, A, hd]
        k = k.view(B, N, L, A, self.H, self.hd).permute(0, 4, 1, 2, 3, 5)  # [B, H, N, L, A, hd]
        
        # Per-agent scores then average across A  →  [B, H, N, L]
        # scores_a: [B,H,A,N,L] = <q_h,a, k_h,*,a> / sqrt(hd)
        # scores_a = torch.einsum('bhad,bhnlahd->bhanl', q / (self.hd ** 0.5), k)
        scores_a = torch.einsum('bhad,bhnlad->bhanl', q / (self.hd ** 0.5), k)
        # Mean over agent index to collapse to demo frame grid (you suggested "mean" is OK)
        scores = scores_a.mean(dim=2)  # [B, H, N, L]

        if demo_mask is not None:
            mask = (demo_mask == 1).unsqueeze(1)        # [B,1,N,L]
            scores = scores.masked_fill(~mask, float('-inf'))

        if self.tau is not None:
            scores = scores / self.tau

        # Softmax over the support (N×L)
        scores_flat = scores.view(B, self.H, N * L)
        attn_flat = F.softmax(scores_flat, dim=-1)      # [B,H,N*L]
        attn = attn_flat.view(B, self.H, N, L)          # [B,H,N,L]

        # ---------- Hyperbolic weighted barycenter ----------
        # demo_hyp: [B,N,L,d_h]; attn: [B,H,N,L]
        vals = demo_hyp                                      # [B,N,L,d_h]
        w = attn.unsqueeze(-1)                               # [B,H,N,L,1]

        # Move values to tangent @0
        v = logmap0(vals, self.c)                            # [B,N,L,d_h]
        v = v.unsqueeze(1)                                   # [B,1,N,L,d_h]
        m_h = (w * v).sum(dim=(2, 3))                        # [B,H,d_h]   (weighted sum in tangent)
        y_h = expmap0(m_h, self.c)                           # [B,H,d_h]

        # Average heads in tangent, then map back
        v_heads = logmap0(y_h, self.c)                       # [B,H,d_h]
        m = v_heads.mean(dim=1)                              # [B,d_h]
        curr_hyp = expmap0(m, self.c)                        # [B,d_h]

        # stay inside ball
        max_rad = (1.0 / _sqrt_c(self.c)) - 1e-5
        norm = torch.norm(curr_hyp, dim=-1, keepdim=True).clamp_min(1e-6)
        curr_hyp = curr_hyp * torch.clamp(max_rad / norm, max=1.0)
        return curr_hyp, attn
