# alignment.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from utilities import expmap0, logmap0, _sqrt_c  # and your barycenter helpers if any

class EuclidToHypAlign(nn.Module):
    """
    Euclid→Hyp alignment:
      - Build attention in Euclidean space between current ρ(G_l) (queries)
        and demo ρ(G_l) (keys).
      - Aggregate *hyperbolic* demo embeddings via a Poincaré weighted mean
        (no linear mixing).

    Shapes:
      curr_rho  : [B, A, d_e]
      demo_rho  : [B, N, L, A, d_e]     # per (demo, time, agent-node)
      demo_hyp  : [B, N, L, d_h]        # per (demo, time) frame (NO agent-node axis)
      demo_mask : [B, N, L]  (1 valid, 0 pad). Optional.

    Returns:
      curr_hyp  : [B, A, d_h]           # per current agent-node
      attn      : [B, H, A, N, L]       # Euclidean attention over (N,L), per head
    """
    def __init__(
        self,
        d_e: int,
        d_h: int,
        heads: int = 4,
        c: float = 1.0,
        tie_agent_indices: bool = True,
        temperature: Optional[float] = None,
        ln_euclid: bool = True,
    ):
        super().__init__()
        assert heads >= 1
        self.d_e = d_e
        self.d_h = d_h
        self.h = heads
        self.c = c
        self.tie_agent_indices = tie_agent_indices
        self.tau = temperature

        self.ln_q = nn.LayerNorm(d_e) if ln_euclid else nn.Identity()
        self.ln_k = nn.LayerNorm(d_e) if ln_euclid else nn.Identity()
        # per-head projection (dot-product attention in Euclidean component)
        assert d_e % heads == 0, "d_e must be divisible by #heads"
        self.d_e_head = d_e // heads
        self.Wq = nn.Linear(d_e, heads * self.d_e_head, bias=False)
        self.Wk = nn.Linear(d_e, heads * self.d_e_head, bias=False)

    def forward(
        self,
        curr_rho: torch.Tensor,          # [B, A, d_e]
        demo_rho: torch.Tensor,          # [B, N, L, A, d_e]
        demo_hyp: torch.Tensor,          # [B, N, L, d_h]
        demo_mask: Optional[torch.Tensor] = None,  # [B, N, L]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, A, d_e = curr_rho.shape
        Bd = demo_rho.shape[0]
        if B != Bd:
            assert B % Bd == 0, f"Batch mismatch: curr_rho B={B} not a multiple of demo batch Bd={Bd}"
            reps = B // Bd
            demo_rho = demo_rho.unsqueeze(1).repeat(1, reps, 1, 1, 1, 1).view(B, *demo_rho.shape[1:])
            demo_hyp = demo_hyp.unsqueeze(1).repeat(1, reps, 1, 1, 1).view(B, *demo_hyp.shape[1:])
            if demo_mask is not None:
                demo_mask = demo_mask.unsqueeze(1).repeat(1, reps, 1, 1).view(B, *demo_mask.shape[1:])

        assert d_e == self.d_e
        assert demo_rho.shape[:4] == ( *demo_hyp.shape[:2], demo_hyp.shape[2], A), \
            "demo_rho must be [B,N,L,A,d_e] and match demo_hyp [B,N,L,d_h]"

        N, L = demo_hyp.shape[1], demo_hyp.shape[2]
        d_h = demo_hyp.shape[-1]

        # --- Euclidean projections (split heads) ---
        q = self.Wq(self.ln_q(curr_rho))                 # [B, A, H*d_e_head]
        q = q.view(B, A, self.h, self.d_e_head).transpose(1, 2)          # [B, H, A, d_eh]

        k = self.Wk(self.ln_k(demo_rho))                 # [B, N, L, A, H*d_e_head]
        k = k.view(B, N, L, A, self.h, self.d_e_head).permute(0, 4, 3, 1, 2, 5)
        # k: [B, H, A_demo, N, L, d_eh]

        # --- tie agent indices: use the same agent index on demo side ---
        # Now q is [B,H,A_q,d_eh] and k is [B,H,A_demo,N,L,d_eh] where we want A_demo == A_q
        # Select the diagonal across the agent dimension by indexing A_demo with A_q
        # This effectively builds scores only between agent a and the demo's agent a.
        # Resulting k_same: [B, H, A, N, L, d_eh]
        idx = torch.arange(A, device=curr_rho.device)
        # expand idx to [B,H,A,N,L,1] for gather:
        idx_expand = idx.view(1, 1, A, 1, 1, 1).expand(B, self.h, A, N, L, 1)
        k_same = torch.gather(k, dim=2, index=idx_expand.expand(B, self.h, A, N, L, self.d_e_head))

        # --- scaled dot-product scores over (N,L) ---
        # q:      [B,H,A, d_eh]
        # k_same: [B,H,A,N,L,d_eh]
        scores = (q.unsqueeze(3).unsqueeze(4) * k_same).sum(-1)   # [B,H,A,N,L]
        scores = scores / (self.d_e_head ** 0.5)

        if demo_mask is not None:
            # mask invalid (N,L)
            dm = demo_mask[:, None, None, :, :].to(dtype=scores.dtype)  # [B,1,1,N,L]
            scores = scores.masked_fill(dm == 0, float('-inf'))

        if self.tau is not None:
            scores = scores / self.tau

        # softmax over (N,L)
        scores_flat = scores.view(B, self.h, A, N * L)            # [B,H,A,S]
        attn_flat = torch.softmax(scores_flat, dim=-1)
        attn = attn_flat.view(B, self.h, A, N, L)                 # [B,H,A,N,L]

        # --- Hyperbolic aggregation: Poincaré weighted mean over (N,L) ---
        vals = demo_hyp.view(B, N * L, d_h)                       # [B,S,d_h]
        # compute per (B,H,A): weighted mean in tangent at 0, then expmap back
        vals_tan0 = logmap0(vals, self.c)                         # [B,S,d_h]
        w = attn_flat.unsqueeze(-1)                                # [B,H,A,S,1]
        vals_tan0 = vals_tan0.unsqueeze(1).unsqueeze(1)           # [B,1,1,S,d_h]
        m_h = (w * vals_tan0).sum(dim=3)                          # [B,H,A,d_h]
        y_h = expmap0(m_h, self.c)                                # [B,H,A,d_h]

        # average heads in tangent at 0
        y_h_tan0 = logmap0(y_h, self.c)                           # [B,H,A,d_h]
        m = y_h_tan0.mean(dim=1)                                  # [B,A,d_h]
        curr_hyp = expmap0(m, self.c)                             # [B,A,d_h]

        # final projection (safety)
        max_rad = (1.0 / _sqrt_c(self.c)) - 1e-5
        norm = torch.norm(curr_hyp, dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(max_rad / norm, max=1.0)
        curr_hyp = curr_hyp * scale

        curr_hyp = torch.mean(curr_hyp, dim=1)

        return curr_hyp, attn



