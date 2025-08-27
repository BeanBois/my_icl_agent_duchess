import torch
import torch.nn as nn
from utilities import logmap0  # assumed available in your project

class ProductCrossAttention(nn.Module):
    """
    Cross-attention between a hyperbolic query sequence (path along geodesic)
    and Euclidean context features (rho outputs). Hyperbolic points are mapped
    to the tangent space at 0 via logmap0 before attention.

    Args:
        num_heads: int
        euc_head_dim: int            # per-head dim for keys/vals
        hyp_dim: int                 # dimensionality of hyperbolic points (e.g., 2)
        z_dim: int                   # output latent dim
        curvature: float             # c for Poincaré ball log map
        tau: float                   # temperature for attention scaling
    """
    def __init__(self, num_heads, euc_head_dim, hyp_dim, z_dim, curvature=1.0, tau=0.5):
        super().__init__()
        self.h = num_heads
        self.dh = euc_head_dim
        self.curvature = curvature
        self.tau = tau

        # projectors
        self.wq = nn.Linear(hyp_dim,  self.h * self.dh, bias=False)  # on tangent q
        self.wk = nn.Linear(self.h * self.dh, self.h * self.dh, bias=False)  # on rho feat (already H*Dh)
        self.wv = nn.Linear(self.h * self.dh, self.h * self.dh, bias=False)
        self.wo = nn.Linear(self.h * self.dh, z_dim)

    @staticmethod
    def _split_heads(x, h, dh):
        # x: [B, T/A, H*Dh] -> [B, H, T/A, Dh]
        B, TA, HD = x.shape
        assert HD == h * dh
        x = x.view(B, TA, h, dh).transpose(1, 2)  # [B, H, TA, Dh]
        return x

    @staticmethod
    def _combine_heads(x):
        # x: [B, H, T, Dh] -> [B, T, H*Dh]
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return x

    def forward(self, hyp_seq, rho_ctx):
        """
        hyp_seq: [B, T, hyp_dim]      (e.g., geodesic path points)
        rho_ctx: [B, A, H*Dh]         (rho features per agent-node)
        Returns:
            z: [B, T, z_dim]          (latent sequence)
            attn_mean: [B, T, A]      (mean over heads)
        """
        B, T, hyp_dim = hyp_seq.shape
        Bb, A, HD = rho_ctx.shape
        assert B == Bb, "Batch mismatch between hyp_seq and rho_ctx."
        H_guess = HD  # H*Dh

        # 1) map hyperbolic queries to tangent space
        #    (uses your project's logmap0, Poincaré ball)
        q_tan = logmap0(hyp_seq, self.curvature)          # [B, T, hyp_dim]

        # 2) linear projections
        Q = self.wq(q_tan)                                # [B, T, H*Dh]
        K = self.wk(rho_ctx)                              # [B, A, H*Dh]
        V = self.wv(rho_ctx)                              # [B, A, H*Dh]

        # 3) split heads
        Q = self._split_heads(Q, self.h, self.dh)         # [B, H, T, Dh]
        K = self._split_heads(K, self.h, self.dh)         # [B, H, A, Dh]
        V = self._split_heads(V, self.h, self.dh)         # [B, H, A, Dh]

        # 4) attention scores: [B, H, T, A]
        #    scaled by sqrt(Dh) and temperature tau
        scale = (self.dh ** 0.5) * self.tau
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # 5) softmax over context (A)
        attn = scores.softmax(dim=-1)                    # [B, H, T, A]

        # 6) aggregate values: [B, H, T, Dh]
        out_heads = torch.matmul(attn, V)

        # 7) combine heads and project out
        out = self._combine_heads(out_heads)             # [B, T, H*Dh]
        z = self.wo(out)                                  # [B, T, z_dim]

        # mean attention over heads (useful for inspection/later)
        attn_mean = attn.mean(dim=1)                     # [B, T, A]
        return z, attn_mean
