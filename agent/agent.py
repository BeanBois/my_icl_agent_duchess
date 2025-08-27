import torch 
import torch.nn as nn

from configs import ANGULAR_GRANULITIES
from .policy import Policy
class Agent(nn.Module):

    def __init__(self, 
                geometric_encoder,
                max_translation = 500,
                max_diff_timesteps = 1000,
                beta_start = 1e-4,
                beta_end = 0.02,
                num_att_heads = 4,
                euc_head_dim = 16,
                pred_horizon = 5,
                in_dim_agent = 9,
                angular_granulities = ANGULAR_GRANULITIES,               # list[float]
                curvature: float = 1.0,
                tau: float = 0.5):

        super().__init__()
        self.policy = Policy(
            geometric_encoder,
                num_att_heads,
                euc_head_dim,
                pred_horizon,
                in_dim_agent,
                angular_granulities,               # list[float]
                curvature,
                tau
                )
        self.max_translation = max_translation
        self.max_diff_timesteps = max_diff_timesteps
        betas = torch.linspace(beta_start, beta_end, max_diff_timesteps)  # linear; swap for cosine if you like
        self.register_buffer("betas", betas)                     # [K]
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))  # [K]
    
    def forward(self,
                curr_agent_info, # [B x self.num_agent_nodes x 6] x, y, theta, state, time, done
                curr_object_pos, # [B x M x 2] x,y
                demo_agent_info, # [B x N x L x self.num_agent_nodes x 6] x, y, theta, state, time, done
                demo_object_pos, # [B x N x L x M x 2]
                actions,         # [B, T, 4]
                ): 
        
        B, _, _ = actions.shape 
        device = actions.device 
        timesteps = torch.randint(0,self.max_diff_timesteps, (B,), device = device)
        noisy_actions, _, _ = self.add_action_noise(actions, timesteps) # [B, T, 4]

        denoising_directions_normalised = self.policy(
            curr_agent_info, # [B x self.num_agent_nodes x 6] x, y, theta, state, time, done
            curr_object_pos, # [B x M x 2] x,y
            demo_agent_info, # [B x N x L x self.num_agent_nodes x 6] x, y, theta, state, time, done
            demo_object_pos, # [B x N x L x M x 2]
            noisy_actions # [B, T, 4]
        ) # [B, T, 4, 5]
        denoising_directions = self._unnormalise_denoising_directions(denoising_directions_normalised)

        return denoising_directions, noisy_actions

    # get unnormalised noise and project to SE(2) to add noise 
    def add_action_noise(self, actions: torch.Tensor, t: torch.Tensor):
        """
        actions: [B,T,4] = (dx,dy,theta,state)
        t:       [B,T] int diffusion step
        returns: (noisy_actions [B,T,4], epsilon_target [B,T,3], state_eps [B,T])
                epsilon_target is the Lie noise xi you sampled (vx,vy,omega)
        """
        B,T,_ = actions.shape
        dev, dt = actions.device, actions.dtype

        transrot = actions[...,:3]
        state    = actions[...,-1]

        # clean SE(2)
        T_clean = self._se2_from_vec(transrot)         # [B,T,3,3]

        betas = self._betas_lookup(t).to(dev, dt)    # [B,T]
        sigma = torch.sqrt(betas).unsqueeze(-1)      # [B,T,1]

        # sample Lie noise ε ~ N(0, I) then scale by σ_t
        eps_body = torch.randn(B,T,3, device=dev, dtype=dt) * sigma  # [B,T,3]
        Xi = self._se2_exp(eps_body)                                       # [B,T,3,3]

        # left-invariant noising
        T_noisy = self._se2_compose(Xi, T_clean)                           # [B,T,3,3]

        noisy_vec = self._se2_to_vec(T_noisy)                               # [B,T,3]
        # wrap angle to (-pi,pi]
        noisy_vec[...,2] = ((noisy_vec[...,2] + torch.pi) % (2*torch.pi)) - torch.pi

        # state/gripper channel in R
        state_eps = torch.randn_like(state) * torch.sqrt(betas)       # [B,T]
        state_noisy = state + state_eps
        state_noisy = state_noisy.clamp(0.,1.)

        noisy_actions = torch.cat([noisy_vec, state_noisy.unsqueeze(-1)], dim=-1)
        return noisy_actions, eps_body, state_eps

    def _actions_vect_to_SE2_flat(self, actions):
        # (x,y,theta_rad,state) => SE(2).flatten() | state 
        B, T, _ = actions.shape
        device = actions.device 
        dtype = actions.dtype 

        dx = actions[..., 0]
        dy = actions[..., 1]
        th = actions[..., 2]
        st = actions[..., 3]

        c, s = torch.cos(th), torch.sin(th)
        # T_clean: [B,T,3,3]
        SE2 = torch.zeros(B, T, 3, 3, device=device, dtype=dtype)
        SE2[..., 0, 0] = c
        SE2[..., 0, 1] = -s
        SE2[..., 1, 0] = s
        SE2[..., 1, 1] = c
        SE2[..., 0, 2] = dx
        SE2[..., 1, 2] = dy
        SE2[..., 2, 2] = 1.

        SE2_flat = SE2.view(B,T,9)
        SE2_flat_final = torch.concat([SE2_flat, st], dim = -1)
        return SE2_flat_final

    def _se2_exp(self, xi: torch.Tensor):
        """
        xi: [...,3] -> (vx,vy,omega)  (body-frame twist)
        returns SE(2) matrix [...,3,3]
        """
        vx, vy, w = xi[...,0], xi[...,1], xi[...,2]
        eps = 1e-6
        sw, cw = torch.sin(w), torch.cos(w)
        w_safe  = torch.where(torch.abs(w) < eps, torch.ones_like(w), w)
        a = sw / w_safe
        b = (1. - cw) / w_safe

        # V @ v
        tx = a*vx - b*vy
        ty = b*vx + a*vy
        tx = torch.where(torch.abs(w) < eps, vx, tx)
        ty = torch.where(torch.abs(w) < eps, vy, ty)

        T = torch.zeros(*xi.shape[:-1], 3, 3, dtype=xi.dtype, device=xi.device)
        T[...,0,0] = cw;  T[...,0,1] = -sw; T[...,0,2] = tx
        T[...,1,0] = sw;  T[...,1,1] =  cw; T[...,1,2] = ty
        T[...,2,2] = 1.0
        return T

    def _se2_from_vec(self, vec: torch.Tensor):
        """
        vec: [...,3] -> (dx,dy,theta) to matrix
        """
        dx, dy, th = vec[...,0], vec[...,1], vec[...,2]
        c, s = torch.cos(th), torch.sin(th)
        T = torch.zeros(*vec.shape[:-1], 3, 3, dtype=vec.dtype, device=vec.device)
        T[...,0,0] = c;  T[...,0,1] = -s; T[...,0,2] = dx
        T[...,1,0] = s;  T[...,1,1] =  c; T[...,1,2] = dy
        T[...,2,2] = 1.0
        return T

    def _se2_to_vec(self, T: torch.Tensor):
        """
        T: [...,3,3] -> (dx,dy,theta)
        """
        dx = T[...,0,2]
        dy = T[...,1,2]
        th = torch.atan2(T[...,1,0], T[...,0,0])
        return torch.stack([dx,dy,th], dim=-1)

    def _se2_compose(self, A: torch.Tensor, B: torch.Tensor):
        """matrix compose with batch broadcasting: A @ B"""
        return A @ B

    def _betas_lookup(self, timesteps):
        return self.betas[timesteps]

    def _unnormalise_denoising_directions(self, denoising_directions_normalised):
        
        denoising_directions_normalised = denoising_directions_normalised[:4] * self.max_translation 
        return denoising_directions_normalised 
