import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from utilities import expmap0, logmap0

class CurrentHypTokenRefinerPA(nn.Module):
    """
    Current token per agent, conditioned on rho(G)[B,A,de] and demo_hyp[B,N,L,Dh].
    Outputs u[B,A,Dh] in the hyperbolic ball.

    time_mode:
      - "flatten":   treat all (n,l) as demo keys (M=N*L) shared by all agents.
      - "pool_agent":per agent, attend over L to get per-demo summaries => keys of size N per agent.
    """
    def __init__(
        self,
        Dh: int,        # hyperbolic dim
        de: int,        # rho(G) feature dim (per agent)
        K: int = 2,     # refinement steps
        c: float = 1.0, # curvature > 0 (ball radius = 1/sqrt(c))
        hidden: int = None,
        time_mode: str = "flatten",  # or "pool_agent"
    ):
        super().__init__()
        assert time_mode in ("flatten", "pool_agent")
        self.Dh, self.de, self.K, self.c = Dh, de, K, c
        self.time_mode = time_mode
        h = hidden or max(Dh, de)

        # --- Seeding ---
        self.seed_g = nn.Linear(de, Dh, bias=True)  # maps rho_g[a] -> Dh
        self.seed_d = nn.Linear(Dh, Dh, bias=False) # mixes pooled demo summary

        # --- Refinement ---
        self.q_from_u = nn.Linear(Dh, Dh, bias=False)
        self.q_from_g = nn.Linear(de, Dh, bias=True)

        self.kv_proj  = nn.Linear(Dh, Dh, bias=False)

        self.delta_mlp = nn.Sequential(
            nn.Linear(Dh, h),
            nn.GELU(),
            nn.Linear(h, Dh)
        )

        self.norm_tan = nn.LayerNorm(Dh)

        # Time pooling (only used for time_mode="pool_agent")
        self.time_pool_q = nn.Linear(de, Dh, bias=True)
        self.time_pool_scale = 1.0 / sqrt(Dh)

    # ---------- helpers ----------
    def _flatten_time(self, demo_hyp, mask):
        """
        demo_hyp: (B,N,L,Dh) -> demo_tan_2d: (B,M,Dh) with M=N*L
        mask    : (B,N,L)    -> mask_2d    : (B,M)
        """
        B, N, L, Dh = demo_hyp.shape
        demo_tan = logmap0(demo_hyp, self.c)               # (B,N,L,Dh)
        demo_tan_2d = demo_tan.reshape(B, N*L, Dh)         # (B,M,Dh)
        mask_2d = None
        if mask is not None:
            mask_2d = mask.reshape(B, N*L)                 # (B,M)
        return demo_tan_2d, mask_2d

    def _pool_time_per_agent(self, demo_hyp, rho_g, mask):
        """
        Per agent time pooling:
          q_time[b,a,:] attends over L within each demo n.
        Returns:
          demo_pa: (B, A, N, Dh) in tangent
        """
        B, N, L, Dh = demo_hyp.shape
        A = rho_g.size(1)

        demo_tan = logmap0(demo_hyp, self.c)               # (B,N,L,Dh)
        q_time = self.time_pool_q(rho_g)                   # (B,A,Dh)

        # scores[b,a,n,l] = <q_time[b,a], demo_tan[b,n,l]>
        scores = torch.einsum('bad,bnld->banl', q_time, demo_tan) * self.time_pool_scale  # (B,A,N,L)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).bool(), float('-inf'))         # broadcast (B,1,N,L)

        a_time = F.softmax(scores, dim=-1)                 # (B,A,N,L)
        # pooled over L: (B,A,N,Dh)
        demo_pa = torch.einsum('banl,bnld->band', a_time, demo_tan)
        return demo_pa  # (B,A,N,Dh)

    def _seed_from_demos(self, rho_g, demo_keys, demo_mask=None, per_agent=False):
        """
        Build per-agent seed via attention pooling over demos.

        Inputs:
          rho_g:     (B,A,de)
          demo_keys: (B,M,Dh) if per_agent=False
                     (B,A,N,Dh) if per_agent=True
          demo_mask: (B,M) or (B,A,N) or None
        Returns:
          pooled: (B,A,Dh)
        """
        B, A, de = rho_g.shape
        q = self.seed_g(rho_g)                              # (B,A,Dh)

        if not per_agent:
            # Keys shared across agents: demo_keys (B,M,Dh)
            K = self.kv_proj(demo_keys)                    # (B,M,Dh)
            scores = torch.einsum('bad,bmd->bam', q, K) / sqrt(self.Dh)  # (B,A,M)
            if demo_mask is not None:
                scores = scores.masked_fill(~demo_mask.unsqueeze(1).bool(), float('-inf'))
            a = F.softmax(scores, dim=-1)                  # (B,A,M)
            V = K                                          # (B,M,Dh)
            # Expand V to (B,A,M,Dh) to match 'a'
            V_exp = V.unsqueeze(1).expand(B, A, V.size(1), V.size(2)).contiguous()
            pooled = torch.einsum('bam,bamd->bad', a, V_exp)             # (B,A,Dh)
            return pooled
        else:
            # Per-agent keys: demo_keys (B,A,N,Dh)
            K = self.kv_proj(demo_keys)                    # (B,A,N,Dh)
            scores = torch.einsum('bad,band->ban', q, K) / sqrt(self.Dh) # (B,A,N)
            if demo_mask is not None:
                scores = scores.masked_fill(~demo_mask.bool(), float('-inf'))             # (B,A,N)
            a = F.softmax(scores, dim=-1)                  # (B,A,N)
            pooled = torch.einsum('ban,band->bad', a, K)   # (B,A,Dh)
            return pooled

    # ---------- forward ----------
    def forward(self, demo_hyp: torch.Tensor, rho_g: torch.Tensor, mask: torch.Tensor = None):
        """
        demo_hyp : (B, N, L, Dh)  hyperbolic
        rho_g    : (B, A, de)     Euclidean
        mask     : optional (B, N, L) boolean for valid demo steps
        returns  : u (B, A, Dh)   hyperbolic
        """
        B, N, L, Dh = demo_hyp.shape
        A = rho_g.size(1)
        assert Dh == self.Dh

        if self.time_mode == "flatten":
            demo_2d, mask_2d = self._flatten_time(demo_hyp, mask)         # (B,M,Dh), (B,M)
            # ---- Seed ----
            demo_pool = self._seed_from_demos(rho_g, demo_2d, mask_2d, per_agent=False)  # (B,A,Dh)
            seed_tan  = self.seed_g(rho_g) + self.seed_d(demo_pool)        # (B,A,Dh)
            u = expmap0(seed_tan, self.c)                                   # (B,A,Dh)

            # ---- K refinements ----
            Kmat = self.kv_proj(demo_2d)                                    # (B,M,Dh)
            if mask_2d is not None:
                mask_scores = ~mask_2d.unsqueeze(1).bool()                  # (B,1,M) -> broadcast

            for _ in range(self.K):
                u_tan = self.norm_tan(logmap0(u, self.c))                   # (B,A,Dh)
                q = self.q_from_u(u_tan) + self.q_from_g(rho_g)             # (B,A,Dh)

                scores = torch.einsum('bad,bmd->bam', q, Kmat) / sqrt(Dh)   # (B,A,M)
                if mask_2d is not None:
                    scores = scores.masked_fill(mask_scores, float('-inf'))
                a = F.softmax(scores, dim=-1)                                # (B,A,M)

                V = Kmat                                                    # (B,M,Dh)
                V_exp = V.unsqueeze(1).expand(B, A, V.size(1), V.size(2))   # (B,A,M,Dh)
                delta = torch.einsum('bam,bamd->bad', a, V_exp)             # (B,A,Dh)

                upd = self.delta_mlp(delta)                                 # (B,A,Dh)
                u = expmap0(u_tan + upd, self.c)                            # (B,A,Dh)

            return u

        # -------- time_mode == "pool_agent" --------
        demo_pa = self._pool_time_per_agent(demo_hyp, rho_g, mask)         # (B,A,N,Dh)
        # Optional per-demo mask after pooling: (B,N) -> (B,A,N)
        demo_pa_mask = None
        if mask is not None:
            demo_pa_mask = mask.any(dim=2, keepdim=False).unsqueeze(1).expand(B, A, N)  # (B,A,N)

        # ---- Seed ----
        demo_pool = self._seed_from_demos(rho_g, demo_pa, demo_pa_mask, per_agent=True)  # (B,A,Dh)
        seed_tan  = self.seed_g(rho_g) + self.seed_d(demo_pool)                           # (B,A,Dh)
        u = expmap0(seed_tan, self.c)                                                     # (B,A,Dh)

        # ---- K refinements ----
        Kmat = self.kv_proj(demo_pa)                                                      # (B,A,N,Dh)
        for _ in range(self.K):
            u_tan = self.norm_tan(logmap0(u, self.c))                                     # (B,A,Dh)
            q = self.q_from_u(u_tan) + self.q_from_g(rho_g)                               # (B,A,Dh)

            scores = torch.einsum('bad,band->ban', q, Kmat) / sqrt(Dh)                    # (B,A,N)
            if demo_pa_mask is not None:
                scores = scores.masked_fill(~demo_pa_mask.bool(), float('-inf'))
            a = F.softmax(scores, dim=-1)                                                 # (B,A,N)

            delta = torch.einsum('ban,band->bad', a, Kmat)                                # (B,A,Dh)
            upd = self.delta_mlp(delta)                                                   # (B,A,Dh)
            u = expmap0(u_tan + upd, self.c)                                              # (B,A,Dh)

        return u
