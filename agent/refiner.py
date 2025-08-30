import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utilities import expmap0, logmap0


class ScaledDotAttn(nn.Module):
    """
    Single-head scaled dot-product attention.
    Query:  (B, d_q)
    Keys:   (B, N, d_k)
    Values: (B, N, d_v)
    Returns pooled (B, d_v) and weights (B, N).
    """
    def __init__(self, d_q, d_k, d_v):
        super().__init__()
        self.scale = 1.0 / sqrt(d_k)
        self.Wq = nn.Linear(d_q, d_k, bias=False)
        self.Wk = nn.Linear(d_k, d_k, bias=False)  # assumes keys already in d_k
        self.Wv = nn.Linear(d_v, d_v, bias=False)  # optional projection on V

    def forward(self, q, K, V, mask=None):
        # q: (B, d_q) -> (B, d_k)
        qk = self.Wq(q).unsqueeze(1)         # (B, 1, d_k)
        Kp = self.Wk(K)                      # (B, N, d_k)
        scores = (qk * Kp).sum(-1) * self.scale  # (B, N)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        a = F.softmax(scores, dim=-1)        # (B, N)
        Vp = self.Wv(V)                      # (B, N, d_v)
        pooled = torch.bmm(a.unsqueeze(1), Vp).squeeze(1)  # (B, d_v)
        return pooled, a


class CurrentHypTokenRefiner(nn.Module):
    """
    Builds a current hyperbolic token u \in B_c^Dh that:
      • is *seeded* from rho(G) and demo hyperbolic embeddings
      • is *refined* K times via cross-attention to demos in tangent space

    Inputs:
      demo_hyp : (B, N, Dh)  # hyperbolic points (Poincaré ball)
      rho_g    : (B, Dg)     # Euclidean features rho(G)

    Output:
      u        : (B, Dh)     # hyperbolic point (Poincaré ball)
    """
    def __init__(
        self,
        Dh: int,        # hyperbolic dim
        Dg: int,        # rho(G) dim
        K: int = 2,     # refinement steps
        c: float = 1.0, # curvature > 0  (ball radius = 1/sqrt(c))
        hidden: int = None
    ):
        super().__init__()
        self.Dh, self.Dg, self.K, self.c = Dh, Dg, K, c
        h = hidden or max(Dh, Dg)

        # --- Seeding ---
        self.seed_g = nn.Linear(Dg, Dh, bias=True)      # W_rho
        self.seed_d = nn.Linear(Dh, Dh, bias=False)     # W_demo
        self.pooler = ScaledDotAttn(d_q=Dh, d_k=Dh, d_v=Dh)

        # --- Refinement ---
        # Queries depend on current token (tangent) and rho(G)
        self.q_from_u = nn.Linear(Dh, Dh, bias=False)
        self.q_from_g = nn.Linear(Dg, Dh, bias=True)

        # Keys/Values from demo (tangent)
        self.kv_proj = nn.Linear(Dh, Dh, bias=False)

        # Small MLP in tangent to compute delta update
        self.delta_mlp = nn.Sequential(
            nn.Linear(Dh, h),
            nn.GELU(),
            nn.Linear(h, Dh)
        )

        # Optional stabilization
        self.norm_tan = nn.LayerNorm(Dh)

    def _attn_pool_demos(self, demo_tan, rho_g):
        """
        Use rho_g as the query to pool demo_tan (all in tangent space @ 0).
        demo_tan : (B, N, Dh)
        rho_g    : (B, Dg)
        returns (B, Dh)
        """
        # Make query in Dh
        q = self.seed_g(rho_g)                      # (B, Dh)
        pooled, _ = self.pooler(q, demo_tan, demo_tan)  # (B, Dh)
        return pooled

    def forward(self, demo_hyp: torch.Tensor, rho_g: torch.Tensor, mask: torch.Tensor = None):
        """
        mask: optional (B, N) boolean for valid demo positions
        """
        B, N, Dh = demo_hyp.shape
        assert Dh == self.Dh

        # Map demos to tangent @ 0
        demo_tan = logmap0(demo_hyp, self.c)          # (B, N, Dh)

        # ---- Seed current token in tangent @ 0 ----
        demo_pool = self._attn_pool_demos(demo_tan, rho_g)   # (B, Dh)
        seed_tan = self.seed_g(rho_g) + self.seed_d(demo_pool)  # (B, Dh)
        u = expmap0(seed_tan, self.c)                          # (B, Dh) hyperbolic

        # ---- K refinement steps ----
        for _ in range(self.K):
            u_tan = logmap0(u, self.c)                         # (B, Dh)
            u_tan = self.norm_tan(u_tan)

            # Build query from current token + rho(G)
            q = self.q_from_u(u_tan) + self.q_from_g(rho_g)    # (B, Dh)

            # K,V from demos (tangent)
            Kmat = self.kv_proj(demo_tan)                      # (B, N, Dh)
            Vmat = Kmat                                        # tie K,V (ok for single-head)

            # Attention
            scores = torch.einsum('bd,bnd->bn', q, Kmat) / sqrt(Dh)
            if mask is not None:
                scores = scores.masked_fill(~mask.bool(), float('-inf'))
            a = F.softmax(scores, dim=-1)                      # (B, N)
            delta = torch.bmm(a.unsqueeze(1), Vmat).squeeze(1) # (B, Dh)

            # Tangent update + exp back
            upd = self.delta_mlp(delta)                        # (B, Dh)
            u = expmap0(u_tan + upd, self.c)                   # using exp@0 with residual in tan

        return u  # hyperbolic (B, Dh)
