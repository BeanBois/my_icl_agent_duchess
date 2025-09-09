import torch 
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from utilities import se2_exp, se2_log, _wrap_to_pi
import math 
from configs import ANGULAR_GRANULITIES
from .policy import Policy

class Agent(nn.Module):

    def __init__(self, 
                geometric_encoder,
                max_translation = 200,
                max_rotation = 30,
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
        self.max_rotation = math.radians(max_rotation)
        self.max_translation = max_translation
        self.pred_horizon = pred_horizon
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
        
        B, T, _ = actions.shape 
        device = actions.device 
        timesteps = torch.randint(0,self.max_diff_timesteps, (B,1), device = device).repeat(1,T)
        noisy_actions, _, _ = self.add_action_noise(actions, timesteps) # [B, T, 4]

        denoising_directions_normalised = self.policy(
            curr_agent_info, # [B x self.num_agent_nodes x 6] x, y, theta, state, time, done
            curr_object_pos, # [B x M x 2] x,y
            demo_agent_info, # [B x N x L x self.num_agent_nodes x 6] x, y, theta, state, time, done
            demo_object_pos, # [B x N x L x M x 2]
            noisy_actions # [B, T, 4]
        ) # [B, T, 4, 5]

        return denoising_directions_normalised, noisy_actions

    @torch.no_grad()
    def plan_actions(
        self,
        curr_agent_info: Tensor,   # [B, A, ...]
        curr_object_pos: Tensor,   # [B, M, 2]
        demo_agent_info: Tensor,   # [B, N, L, A, ...]
        demo_object_pos: Tensor,   # [B, N, L, M, 2]
        actions: Tensor = None,    # [B, T, 4] start guess; if None, zeros
        K: int = 5,                # number of refinement/DDIM steps
        keypoints: Tensor = None,
        ddim_steps: int = None     # override number of DDIM steps; default=K
    ) -> Tensor:
        device = curr_agent_info.device
        dtype  = curr_agent_info.dtype
        B = curr_agent_info.shape[0]

        # T (horizon) from actions or model attr
        if actions is None:
            T = getattr(self, "pred_horizon", None)
            assert T is not None, "Provide actions or set self.pred_horizon"
            actions = torch.zeros(B, T, 4, device=device, dtype=dtype)
            actions = torch.zeros(B, T, 4, device=device, dtype=dtype)
            actions[...,:2] = (torch.randn(B, T, 2, device=device) * 2 *  (self.max_translation / 10.0)) - (self.max_translation / 10.0)
            actions[...,2:3] = (torch.rand(B, T, 1, device=device) - 0.5) * (2*torch.pi/4)  # +/- 90°
            actions[...,3:4] = torch.full((B,T,1), 0.5, device=device)
        else:
            T = actions.shape[1]


        # DDIM mode
        ab = self.alphas_cumprod.to(device=device, dtype=dtype)    
        D  = ab.numel()
        steps = int(ddim_steps or K)
        assert steps >= 1, "DDIM requires at least 1 step"

        # Build t-schedule
        t_sched = torch.linspace(D - 1, 0, steps=steps + 1, device=device).round().long()   

        # Start from current actions 
        x_t = torch.cat([self._se2_log_cw(actions[..., :3]), actions[..., 3:4]], dim=-1)   

        for s in range(steps):
            t      = t_sched[s].item()
            t_prev = t_sched[s + 1].item()
            ab_t       = ab[t]
            ab_prev    = ab[t_prev]
            sqrt_ab_t  = ab_t.sqrt()
            sqrt1m_t   = (1 - ab_t).clamp_min(1e-12).sqrt()

            a_t = torch.cat([self._se2_exp_cw(x_t[..., :3]), x_t[..., 3:4]], dim=-1)   

            # one-step denoise to get a0_hat
            denoise = self.policy(curr_agent_info, curr_object_pos,
                                demo_agent_info, demo_object_pos, a_t, mode='eval')       
            a0_hat  = self._svd_refine_once(a_t, denoise, keypoints)          
            x0_hat  = torch.cat([self._se2_log_cw(a0_hat[..., :3]), a0_hat[..., 3:4]], dim=-1)   

            # eps and DDIM update
            eps_hat = (x_t - sqrt_ab_t * x0_hat) / (sqrt1m_t + 1e-12)
            x_t     = ab_prev.sqrt() * x0_hat + (1 - ab_prev).clamp_min(1e-12).sqrt() * eps_hat

        # back to vector actions
        a0 = self._se2_exp_cw(x_t[..., :3])
        s0 = x_t[..., 3:4].clamp(0.0, 1.0)
        out = torch.cat([a0, s0], dim=-1)
        out[..., 2] = _wrap_to_pi(out[..., 2])
        return out

    # get unnormalised noise and project to SE(2) to add noise 
    def add_action_noise(self, actions: torch.Tensor, t_int: torch.Tensor):
        B,T,_ = actions.shape
        dev, dt = actions.device, actions.dtype

        # schedules 
        ab_all = self.alphas_cumprod.to(dev, dt)                 
        ab_t   = ab_all.gather(0, t_int.reshape(-1)).reshape(B,T)   
        sqrt_ab_t  = ab_t.sqrt()                                  
        sqrt1m_abt = (1.0 - ab_t).clamp_min(1e-12).sqrt()        

        # split action channels 
        transrot0 = actions[...,:3]     
        state0    = actions[...,-1]    

        # go to tangent: x0 = log( SE2(transrot0) ) 
        x0 = self._se2_log_cw(transrot0)        
        # sample epsilons 
        eps_action = torch.randn(B,T,3, device=dev, dtype=dt)    
        eps_state  = torch.randn(B,T,  device=dev, dtype=dt)     

        # closed-form forward: x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps 
        x_t = sqrt_ab_t.unsqueeze(-1) * x0 + sqrt1m_abt.unsqueeze(-1) * eps_action   

        # map back to action space 
        transrot_t = self._se2_exp_cw(x_t)         
        # keep theta in (-pi,pi]
        transrot_t[...,2] = ((transrot_t[...,2] + torch.pi) % (2*torch.pi)) - torch.pi

        # state diffusion in R 
        state_t = sqrt_ab_t * state0 + sqrt1m_abt * eps_state    
        state_t = state_t.clamp(0.0, 1.0)

        noisy_actions = torch.cat([transrot_t, state_t.unsqueeze(-1)], dim=-1)   
        return noisy_actions, eps_action, eps_state

    def _unnormalise_denoising_directions(self, x, kp_norms):
        return torch.cat([x[..., :2] * self.max_translation,
                        x[..., 2:4] * (self.max_rotation * kp_norms[None,None,:,None]),
                        x[..., 4:5]], dim=-1)

    def _svd_refine_once(self, actions: Tensor, denoise: Tensor, keypoints: Tensor) -> Tensor:
        device, dtype = actions.device, actions.dtype
        B, T, _ = actions.shape
        A = denoise.shape[2]

        # keypoints
        kp0 = keypoints
        kp = kp0.view(1,1,A,2).expand(B,T,A,2)

        dxdy = actions[..., :2]           
        th   = actions[..., 2:3]          
        st   = actions[..., 3:4]          

        c = torch.cos(th); s = torch.sin(th)
        kx, ky = kp[..., 0], kp[..., 1]
        Rx = c * kx + s * ky
        Ry = - s * kx + c * ky

        P  = torch.stack([Rx, Ry], dim=-1) + dxdy.unsqueeze(2)   

        # denoise split
        kp_norms = kp0.norm(dim = -1)
        denoise = self._unnormalise_denoising_directions(denoise, kp_norms)  
        t_bias  = denoise[..., :2].mean(dim=2, keepdim=True)        
        disp    = denoise[..., 2:4]                                 
        Q       = P + disp + t_bias

        # Procrustes (vectorized)
        muP = P.mean(dim=2, keepdim=True)
        muQ = Q.mean(dim=2, keepdim=True)
        X, Y = P - muP, Q - muQ
        H = torch.einsum('btai,btaj->btij', X, Y)   

        U, S, Vh = torch.linalg.svd(H)
        # det correction
        Rtmp = U @ Vh
        det = torch.det(Rtmp)                              
        sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
        Sfix = torch.zeros_like(Rtmp)   
        Sfix[..., 0, 0] = 1.0
        Sfix[..., 1, 1] = sign
        R = U @ Sfix @ Vh

        dtheta = torch.atan2(R[...,0,1], R[...,0,0]).unsqueeze(-1)   

        Rp = torch.einsum('btij,btaj->btai', R, muP.expand_as(P))     
        t  = (muQ - Rp).mean(dim=2)                                   

        dxdy_hat = dxdy + t
        th_hat   = _wrap_to_pi(th.squeeze(-1) + dtheta.squeeze(-1)).unsqueeze(-1)
        s_hat    = (st + denoise[..., 4:5].mean(dim=2)).clamp(0.0, 1.0)
        return torch.cat([dxdy_hat, th_hat, s_hat], dim=-1)           

    def _se2_log_cw(self, vec: torch.Tensor) -> torch.Tensor:
        dx, dy, th_cw = vec[..., 0], vec[..., 1], vec[..., 2]
        vec_ccw = torch.stack([dx, dy, -th_cw], dim=-1)
        xi_ccw = se2_log(vec_ccw)           
        vx, vy, om_ccw = xi_ccw[..., 0], xi_ccw[..., 1], xi_ccw[..., 2]
        xi_cw = torch.stack([vx, vy, -om_ccw], dim=-1)
        return xi_cw

    def _se2_exp_cw(self, xi: torch.Tensor) -> torch.Tensor:
        vx, vy, om_cw = xi[..., 0], xi[..., 1], xi[..., 2]
        xi_ccw = torch.stack([vx, vy, -om_cw], dim=-1)
        vec_ccw = se2_exp(xi_ccw)           
        dx, dy, th_ccw = vec_ccw[..., 0], vec_ccw[..., 1], vec_ccw[..., 2]
        vec_cw = torch.stack([dx, dy, -th_ccw], dim=-1)
        return vec_cw

