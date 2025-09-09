# chatgpt helped me optimised so for-loops i used
import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import poincare_distance_sq, exp_map_x, log_map_x



class ProductManifoldAttention(nn.Module):
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
        B, N, L, A, de = demo_rho_batch.shape
        M = N * L
        demo_rho_flat = demo_rho_batch.view(B, M, A, de)
        demo_hyp_flat = demo_hyp_emb.reshape(B, M, -1)  
        return demo_rho_flat, demo_hyp_flat  # [B,M,A,de], [B,M,dh]

    def _ensure_demo_roots(self, curr_hyp_emb, N):
        if curr_hyp_emb.dim() == 2:
            B, dh = curr_hyp_emb.shape
            return curr_hyp_emb.unsqueeze(1).expand(B, N, dh)
        elif curr_hyp_emb.dim() == 3:
            return curr_hyp_emb
        else:
            raise ValueError(f"curr_hyp_emb must be [B,dh] or [B,N,dh], got {curr_hyp_emb.shape}")

    def _karcher_mean_points(self, pts, c, iters: int = 5, tol: float = 1e-6):
        B, A, N, dh = pts.shape
        mu = pts[:, :, :1, :].clone()   
        for _ in range(iters):
            v = log_map_x(mu.expand(B, A, N, dh), pts, c)         
            g = v.mean(dim=2, keepdim=True)                       
            mu = exp_map_x(mu, g, c)                              
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
        _, N2, L2, dh = demo_hyp_emb.shape
        assert N2 == N and L2 == L, "demo_hyp_emb shape mismatch"

        roots = self._ensure_demo_roots(curr_hyp_emb, N)  # [B,N,dh]

        # ---------- Product-metric attention scores (per demo) ----------
        # Euclidean: ||curr_e - demo_e||^2
        #   curr_e  [B,1,1,A,de]  vs  demo_e [B,N,L,A,de] → [B,N,L,A]
        curr_e = curr_rho_batch.unsqueeze(1).unsqueeze(1)                 
        e_diff_sq = ((curr_e - demo_rho_batch) ** 2).sum(dim=-1)          
        e_diff_sq = e_diff_sq.permute(0, 3, 1, 2).contiguous()            

        # Hyperbolic: d^2( root_n, y_{n,ℓ} )
        #   roots → [B,N,1,dh] → [B,N,L,dh] to match demo_hyp_emb
        roots_exp = roots.unsqueeze(2).expand(B, N, L, dh)                  
        dH2 = poincare_distance_sq(
            roots_exp.reshape(B, N*L, dh),
            demo_hyp_emb.reshape(B, N*L, dh),
            self.c
        ).view(B, N, L)  # [B,N,L]
        dH2 = dH2.unsqueeze(1).expand(B, A, N, L)                          

        # scores per demo/time: s = -(l_H d_H^2 + l_E ||.||^2) / tau
        scores = -(self.lh * dH2 + self.le * e_diff_sq) / max(self.tau, 1e-8)   

        # ---------- Token weights *within each demo* (softmax over L) ----------
        alpha_tok = torch.softmax(scores, dim=-1)   

        # ---------- Euclidean aggregation per demo ----------
        # demo_rho_batch: [B,N,L,A,de] -> [B,A,N,L,de]
        demo_e = demo_rho_batch.permute(0, 3, 1, 2, 4).contiguous()         
        e_out_n = torch.sum(alpha_tok.unsqueeze(-1) * demo_e, dim=-2)       

        # ---------- Hyperbolic aggregation per demo (log/exp at each root_n) ----------
        # logs: log_{root_n}( y_{n,ℓ} ) for all ℓ, then weight by alpha_tok
        log_vecs = log_map_x(
            roots_exp.reshape(B, N*L, dh),               
            demo_hyp_emb.reshape(B, N*L, dh),            
            self.c
        ).view(B, N, L, dh)                              
        # broadcast to [B,A,N,L,dh] to apply per-agent alpha
        log_vecs_ba = log_vecs.unsqueeze(1).expand(B, A, N, L, dh)        
        v_n = torch.sum(alpha_tok.unsqueeze(-1) * log_vecs_ba, dim=-2)    

        # exp back at each root_n, per agent
        roots_ba = roots.unsqueeze(1).expand(B, A, N, dh)                 
        h_out_n = exp_map_x(roots_ba, v_n, self.c)                        

        # ---------- Fuse across demos (learned gate) ----------
        # Demo-level gate from the scores averaged over time (higher = closer)
        demo_scores = scores.mean(dim=-1)                                 
        gamma = torch.softmax(demo_scores, dim=-1)                        

        # Euclidean fuse: weighted average across N
        e_out = torch.sum(gamma.unsqueeze(-1) * e_out_n, dim=2)           

        # Hyperbolic fuse: Karcher mean across {h_out_n}_n using gamma
        # Do a weighted Karcher mean by repeating points proportionally to gamma in tangent.

        # Unweighted initial mean over demos:
        mu0 = self._karcher_mean_points(h_out_n, self.c, iters=3)         

        # One weighted refinement step at mu0:
        mu0_exp = mu0.unsqueeze(2).expand(B, A, N, dh)                    
        v_logs = log_map_x(mu0_exp, h_out_n, self.c)                      
        v_bar  = torch.sum(gamma.unsqueeze(-1) * v_logs, dim=2, keepdim=False)  
        h_out  = exp_map_x(mu0, v_bar, self.c)                             

        # ---------- Project factors and combine ----------
        z_e = self.proj_e(e_out)                                           
        z_h = self.proj_h(h_out)                                           
        z = torch.cat([z_e, z_h], dim=-1)                                  
        z = self.drop(self.norm(z))
        return z
