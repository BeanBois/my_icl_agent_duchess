import torch
import torch.nn as nn
import torch.nn.functional as F
# ========= Poincaré ball utilities (curvature c > 0) =========

def _safe_norm(x, dim=-1, keepdim=False, eps=1e-15):
    return torch.clamp(torch.norm(x, dim=dim, keepdim=keepdim), min=eps)

def _project_to_ball(x, c, eps=1e-5):
    # Ensure ||x|| < 1/sqrt(c)
    sqrt_c = c ** 0.5
    norm = _safe_norm(x, dim=-1, keepdim=True)
    max_norm = (1. - eps) / sqrt_c
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale

def lambda_x(x, c):
    # Conformal factor λ_x^c = 2 / (1 - c ||x||^2)
    x = _project_to_ball(x, c)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 - c * x2).clamp_min(1e-15)

def mobius_add(x, y, c):
    """
    Möbius addition on the Poincaré ball (Ganea et al. 2018).
    """
    x = _project_to_ball(x, c)
    y = _project_to_ball(y, c)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / den.clamp_min(1e-15)

def poincare_distance_sq(x, y, c):
    """
    d_c(x,y)^2 = ((2/√c) atanh( √c ||(-x) ⊕ y|| ))^2
    """
    sqrt_c = c ** 0.5
    x = _project_to_ball(x, c)
    y = _project_to_ball(y, c)
    diff = mobius_add(-x, y, c)               # (-x) ⊕ y
    norm = _safe_norm(diff, dim=-1)
    arg = torch.clamp(sqrt_c * norm, max=1 - 1e-7)
    dist = (2.0 / sqrt_c) * torch.atanh(arg)
    return dist * dist

def log_map_x(x, y, c):
    """
    log_x(y) on the Poincaré ball.
    log_x(y) = (2 / (λ_x sqrt(c))) * atanh( sqrt(c) ||(-x) ⊕ y|| ) * u / ||u||
               where u = (-x) ⊕ y
    """
    sqrt_c = c ** 0.5
    lam = lambda_x(x, c)                      # [*, 1]
    u = mobius_add(-x, y, c)
    unorm = _safe_norm(u, dim=-1, keepdim=True)
    # avoid 0 direction
    u_dir = u / unorm
    arg = torch.clamp(sqrt_c * unorm, max=1 - 1e-7)
    scale = (2.0 / (lam * sqrt_c)) * torch.atanh(arg)  # [*,1]
    return scale * u_dir

def exp_map_x(x, v, c):
    """
    exp_x(v) on the Poincaré ball.
    exp_x(v) = x ⊕ ( tanh( (λ_x sqrt(c)/2) ||v|| ) * v / (sqrt(c) ||v||) )
    """
    sqrt_c = c ** 0.5
    lam = lambda_x(x, c)
    vnorm = _safe_norm(v, dim=-1, keepdim=True)
    # direction
    v_dir = v / vnorm
    # scale
    factor = torch.tanh((lam * sqrt_c * vnorm) / 2.0) / (sqrt_c)
    y = v_dir * factor
    out = mobius_add(x, y, c)
    return _project_to_ball(out, c)

# ========= Product-manifold attention layer =========

class ProductManifoldAttention(nn.Module):
    """
    Product-space attention (H × E):
      - Scores from product metric: s = -(λ_H d_H^2 + λ_E ||.||^2) / τ
      - Euclidean aggregation: weighted average
      - Hyperbolic aggregation: weighted Karcher mean (log/exp at current point)
    """
    def __init__(
        self,
        de: int,              # Euclidean feature dim (de)
        dh: int,              # Hyperbolic dim (dh)
        z_dim: int,           # output latent dim
        curvature: float = 1.0,
        tau: float = 1.0,
        lambda_euc: float = 1.0,
        lambda_hyp: float = 1.0,
        use_layernorm: bool = True,
        dropout: float = 0.0,
        proj_hidden: int = 0,   # 0 = direct proj to z; >0 = small MLP per factor
    ):
        super().__init__()
        assert curvature > 0, "Poincaré ball curvature c must be > 0."
        self.c = curvature
        self.tau = tau
        self.le = lambda_euc
        self.lh = lambda_hyp

        # Factor-wise projections → latent
        # Split z into two halves (E and H); if odd, Euclidean gets the extra unit.
        z_e = z_dim // 2 + (z_dim % 2)
        z_h = z_dim // 2
        def make_head(in_dim, out_dim):
            if proj_hidden and proj_hidden > 0:
                return nn.Sequential(
                    nn.Linear(in_dim, proj_hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(proj_hidden, out_dim, bias=False),
                )
            else:
                return nn.Linear(in_dim, out_dim, bias=False)

        self.proj_e = make_head(de, z_e)
        self.proj_h = make_head(dh, z_h)

        self.norm = nn.LayerNorm(z_dim) if use_layernorm else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _flatten_demos(demo_rho_batch, demo_hyp_emb):
        """
        Flatten over (N, L) → M = N*L.
        demo_rho_batch: [B,N,L,A,de] → [B,M,A,de]
        demo_hyp_emb:   [B,N,L,dh]   → [B,M,dh]
        """
        B, N, L, A, de = demo_rho_batch.shape
        M = N * L
        demo_rho_flat = demo_rho_batch.view(B, M, A, de)
        demo_hyp_flat = demo_hyp_emb.reshape(B, M, -1)  # dh
        return demo_rho_flat, demo_hyp_flat  # [B,M,A,de], [B,M,dh]


    def _ensure_demo_roots(self, curr_hyp_emb, N):
        """
        Accepts curr_hyp_emb as [B, dh] or [B, N, dh].
        Returns roots as [B, N, dh].
        """
        if curr_hyp_emb.dim() == 2:
            B, dh = curr_hyp_emb.shape
            return curr_hyp_emb.unsqueeze(1).expand(B, N, dh)
        elif curr_hyp_emb.dim() == 3:
            return curr_hyp_emb
        else:
            raise ValueError(f"curr_hyp_emb must be [B,dh] or [B,N,dh], got {curr_hyp_emb.shape}")

    def _karcher_mean_points(self, pts, c, iters: int = 5, tol: float = 1e-6):
        """
        pts: [B, A, N, dh] points on the Poincaré ball.
        Returns μ: [B, A, dh] (Fréchet/Karcher mean over N).
        """
        B, A, N, dh = pts.shape
        mu = pts[:, :, :1, :].clone()  # [B,A,1,dh], init from first
        for _ in range(iters):
            v = log_map_x(mu.expand(B, A, N, dh), pts, c)         # [B,A,N,dh]
            g = v.mean(dim=2, keepdim=True)                       # [B,A,1,dh]
            mu = exp_map_x(mu, g, c)                              # [B,A,1,dh]
            if g.norm(dim=-1).max().item() < tol:
                break
        return mu.squeeze(2)  # [B,A,dh]


    def forward(self,
                curr_rho_batch: torch.Tensor,  # [B, A, de]
                curr_hyp_emb: torch.Tensor,    # [B, N, dh]  (or [B, dh] -> broadcast)
                demo_rho_batch: torch.Tensor,  # [B, N, L, A, de]
                demo_hyp_emb: torch.Tensor     # [B, N, L, dh]
                ) -> torch.Tensor:             # [B, A, z_dim]

        device = curr_rho_batch.device
        B, A, de = curr_rho_batch.shape
        Bn, N, L, A_, de_ = demo_rho_batch.shape
        # assert Bn == B and A_ == A and de_ == de, "demo_rho_batch shape mismatch"
        _, N2, L2, dh = demo_hyp_emb.shape
        assert N2 == N and L2 == L, "demo_hyp_emb shape mismatch"

        # ---- NEW: ensure per-demo roots [B,N,dh] ----
        roots = self._ensure_demo_roots(curr_hyp_emb, N)  # [B,N,dh]

        # ---------- Product-metric attention scores (per demo) ----------
        # Euclidean: ||curr_e - demo_e||^2
        #   curr_e  [B,1,1,A,de]  vs  demo_e [B,N,L,A,de] → [B,N,L,A]
        curr_e = curr_rho_batch.unsqueeze(1).unsqueeze(1)                 # [B,1,1,A,de]
        e_diff_sq = ((curr_e - demo_rho_batch) ** 2).sum(dim=-1)          # [B,N,L,A]
        e_diff_sq = e_diff_sq.permute(0, 3, 1, 2).contiguous()            # [B,A,N,L]

        # Hyperbolic: d^2( root_n, y_{n,ℓ} )
        #   roots → [B,N,1,dh] → [B,N,L,dh] to match demo_hyp_emb
        roots_exp = roots.unsqueeze(2).expand(B, N, L, dh)                 # [B,N,L,dh]
        dH2 = poincare_distance_sq(
            roots_exp.reshape(B, N*L, dh),
            demo_hyp_emb.reshape(B, N*L, dh),
            self.c
        ).view(B, N, L)  # [B,N,L]
        dH2 = dH2.unsqueeze(1).expand(B, A, N, L)                          # [B,A,N,L]

        # scores per demo/time: s = -(λ_H d_H^2 + λ_E ||.||^2) / τ
        scores = -(self.lh * dH2 + self.le * e_diff_sq) / max(self.tau, 1e-8)  # [B,A,N,L]

        # ---------- Token weights *within each demo* (softmax over L) ----------
        alpha_tok = torch.softmax(scores, dim=-1)  # [B,A,N,L]

        # ---------- Euclidean aggregation per demo ----------
        # demo_rho_batch: [B,N,L,A,de] -> [B,A,N,L,de]
        demo_e = demo_rho_batch.permute(0, 3, 1, 2, 4).contiguous()        # [B,A,N,L,de]
        e_out_n = torch.sum(alpha_tok.unsqueeze(-1) * demo_e, dim=-2)      # [B,A,N,de]

        # ---------- Hyperbolic aggregation per demo (log/exp at each root_n) ----------
        # logs: log_{root_n}( y_{n,ℓ} ) for all ℓ, then weight by alpha_tok
        log_vecs = log_map_x(
            roots_exp.reshape(B, N*L, dh),              # x
            demo_hyp_emb.reshape(B, N*L, dh),           # y
            self.c
        ).view(B, N, L, dh)                              # [B,N,L,dh]
        # broadcast to [B,A,N,L,dh] to apply per-agent alpha
        log_vecs_ba = log_vecs.unsqueeze(1).expand(B, A, N, L, dh)         # [B,A,N,L,dh]
        v_n = torch.sum(alpha_tok.unsqueeze(-1) * log_vecs_ba, dim=-2)     # [B,A,N,dh]

        # exp back at each root_n, per agent
        roots_ba = roots.unsqueeze(1).expand(B, A, N, dh)                  # [B,A,N,dh]
        h_out_n = exp_map_x(roots_ba, v_n, self.c)                         # [B,A,N,dh]

        # ---------- Fuse across demos (learned gate) ----------
        # Demo-level gate from the scores averaged over time (higher = closer)
        demo_scores = scores.mean(dim=-1)                                  # [B,A,N]
        gamma = torch.softmax(demo_scores, dim=-1)                         # [B,A,N]

        # Euclidean fuse: weighted average across N
        e_out = torch.sum(gamma.unsqueeze(-1) * e_out_n, dim=2)            # [B,A,de]

        # Hyperbolic fuse: Karcher mean across {h_out_n}_n using gamma
        # Do a weighted Karcher mean by repeating points proportionally to gamma in tangent.
        # Practical & differentiable: take μ0 as Karcher mean (unweighted), then do one
        # weighted update step in tangent using gamma expectation.
        # For stability we do a few unweighted iterations then one weighted step.

        # Unweighted initial mean over demos:
        mu0 = self._karcher_mean_points(h_out_n, self.c, iters=3)          # [B,A,dh]

        # One weighted refinement step at mu0:
        mu0_exp = mu0.unsqueeze(2).expand(B, A, N, dh)                     # [B,A,N,dh]
        v_logs = log_map_x(mu0_exp, h_out_n, self.c)                       # [B,A,N,dh]
        v_bar  = torch.sum(gamma.unsqueeze(-1) * v_logs, dim=2, keepdim=False)  # [B,A,dh]
        h_out  = exp_map_x(mu0, v_bar, self.c)                             # [B,A,dh]

        # ---------- Project factors and combine ----------
        z_e = self.proj_e(e_out)                                           # [B,A,z_e]
        z_h = self.proj_h(h_out)                                           # [B,A,z_h]
        z = torch.cat([z_e, z_h], dim=-1)                                  # [B,A,z_dim]
        z = self.drop(self.norm(z))
        return z
        
        # scores: s = -(λ_H d_H^2 + λ_E ||.||^2) / τ
        # scores = -(self.lh * dH2 + self.le * e_diff_sq) / max(self.tau, 1e-8)  # [B, A, M]

        # # ---------- Attention weights ----------
        # alpha = torch.softmax(scores, dim=-1)  # [B, A, M]

        # # ---------- Euclidean aggregation (weighted mean) ----------
        # # demo_rho_flat: [B, M, A, de] → [B, A, M, de]
        # demo_e_for_agg = demo_rho_flat.permute(0, 2, 1, 3).contiguous()
        # e_out = torch.sum(alpha.unsqueeze(-1) * demo_e_for_agg, dim=2)  # [B, A, de]

        # # ---------- Hyperbolic aggregation (Karcher mean via log/exp at curr_h) ----------
        # # log_{x}(y_m): x = curr_hyp_emb; y_m = demo_hyp_flat
        # # Produce logs for each (B, M, dh), then weight per agent with alpha[B,A,M].
        # x = curr_hyp_emb  # [B, dh]
        # y = demo_hyp_flat # [B, M, dh]
        # # Compute log vectors once (no A), then combine with per-agent alpha
        # log_vecs = log_map_x(
        #     x.unsqueeze(1).expand(B, M, dh),  # [B,M,dh]
        #     y,                                # [B,M,dh]
        #     self.c
        # )  # [B, M, dh]
        # # Weight per-agent: alpha [B,A,M] -> [B,A,M,1]
        # v = torch.sum(alpha.unsqueeze(-1) * log_vecs.unsqueeze(1), dim=2)  # [B, A, dh]
        # # Exp back at x (per agent): base x is same across A; broadcast to [B,A,dh]
        # x_ba = x.unsqueeze(1).expand(B, A, dh)
        # h_out = exp_map_x(x_ba, v, self.c)  # [B, A, dh]

        # # ---------- Project factors and combine ----------
        # z_e = self.proj_e(e_out)            # [B, A, z_e]
        # z_h = self.proj_h(h_out)            # [B, A, z_h]
        # z = torch.cat([z_e, z_h], dim=-1)   # [B, A, z_dim]

        # z = self.drop(self.norm(z))         # optional LN/Dropout
        # return z
