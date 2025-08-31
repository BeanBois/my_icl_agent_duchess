import torch
import torch.nn as nn

from configs import ANGULAR_GRANULITIES
from utilities import * 

# aux
from .demo_handler import DemoHandler
from .my_local_graph import build_local_heterodata_batch
from .rho import Rho 
from .foresight import IntraActionSelfAttn, PsiForesight
from .pma import ProductManifoldAttention
from .action_head import SimpleActionHead


from .refiner import CurrentHypTokenRefinerPA
from .alignment import EuclidToHypAlign

class Policy(nn.Module):

    def __init__(self,
                geometric_encoder,
                num_att_heads,
                euc_head_dim,
                pred_horizon,
                in_dim_agent = 9,
                angular_granulities = ANGULAR_GRANULITIES,               # list[float]
                curvature: float = 1.0,
                tau: float = 0.5,):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.tau = tau
        self.curvature = curvature
        self.euc_dim = num_att_heads * euc_head_dim
        self.hyp_dim = 2
        self.z_dim = 2 * self.hyp_dim + self.euc_dim
        self.angular_granulities = angular_granulities
        

        self.geometric_encoder = geometric_encoder 
        self.rho = Rho(
            in_dim_agent = in_dim_agent, # default by construction, 4 onehot + 5 scalars (sin,cos,state,time,done)
            in_dim_scene = num_att_heads * euc_head_dim,     
            edge_dim     = num_att_heads * euc_head_dim,        
            hidden_dim   = num_att_heads * euc_head_dim,
            out_dim      = num_att_heads * euc_head_dim,
            num_layers   = 2,
            heads        = num_att_heads,
            dropout      = 0.1,
            use_agent_agent = False
        )
        self.demo_handler = DemoHandler(
            curvature=self.curvature,
            angular_granularities_deg=self.angular_granulities,
        )
        self.hyp_context = CurrentHypTokenRefinerPA(
            Dh=self.hyp_dim, de=self.euc_dim, K=2, c = self.curvature
        )
        self.agent_pool_q = nn.Linear(self.euc_dim, 1, bias=False)
        # self.hyp_context = EuclidToHypAlign(
        #     d_e = self.euc_dim, d_h = self.hyp_dim, heads = num_att_heads, 
        #     c = self.curvature, temperature=tau
        # )
        self.context_alignment = ProductManifoldAttention(
            de = self.euc_dim,
            dh = self.hyp_dim,
            z_dim = self.z_dim,
            curvature= self.curvature,
            tau = self.tau,
            dropout = 0.1,
            proj_hidden = 0, 
        )
        self.foresight_adjustment = PsiForesight(
            z_dim=self.z_dim,
            edge_dim=self.z_dim,
            n_heads = num_att_heads,
            dropout=0.1
        )
        self.intra_action = IntraActionSelfAttn(z_dim=self.z_dim, n_heads=4, ff_mult=4, dropout=0.0)

        self.action_head = SimpleActionHead(
            in_dim=self.z_dim,
            hidden_dim=self.z_dim // 2
        )

        self.curr_hyp_emb = nn.Parameter(torch.randn(self.hyp_dim))
              
    def forward(self,
                curr_agent_info, # [B x self.num_agent_nodes x 6] x, y, theta, state, time, done
                curr_object_pos, # [B x M x 2] x,y
                demo_agent_info, # [B x N x L x self.num_agent_nodes x 6] x, y, theta, state, time, done
                demo_object_pos, # [B x N x L x M x 2]
                actions # [B, T, 4]
                ):
        B, N, L, num_agent_nodes, agent_dim = demo_agent_info.shape
        _, _, _, num_object_nodes, obj_pos_dim = demo_object_pos.shape 

        ############################ First process demos into hyperbolic embeddings ############################
        demo_hyp_all = self.demo_handler(demo_agent_info)
        B_, N_, L_, dh = demo_hyp_all.shape
        assert (B_, N_, L_,) == (B, N, L), "SK shape mismatch."

        ### then process demo Euclidean embeddings 
        ### then get rho(G) for each demo
        flat_demo_agent_info = demo_agent_info.view(B * N * L, num_agent_nodes, agent_dim)
        flat_demo_object_pos = demo_object_pos.view(B * N * L, num_object_nodes, obj_pos_dim)
        
        ### first embed obj in demos
        F_list, C_list = [], []
        for i in range(B*N*L):
            # one cloud Pi: [M_raw, 2]  (use your raw per-frame points here)
            Pi = flat_demo_object_pos[i]           # [M_raw, 2]
            Fi, Ci = self.geometric_encoder(Pi)    # Fi: [M, D], Ci: [M, 2]
            F_list.append(Fi)
            C_list.append(Ci)
        flat_demo_scene_feats_batch = torch.stack(F_list, dim=0)  # [B*N*L, M, D]
        flat_demo_scene_pos_batch   = torch.stack(C_list, dim=0)  # [B*N*L, M, 2]

        flat_demo_local_graphs = build_local_heterodata_batch(
            agent_pos_b = flat_demo_agent_info,
            scene_pos_b=flat_demo_scene_pos_batch,
            scene_feats_b=flat_demo_scene_feats_batch,
            num_freqs=self.euc_dim // 4,
            include_agent_agent=False # no agent-agent edges 
        ) # returns HeteroBatch[B*N*L]

        ### get rho(G) for demos 
        demo_node_emb, _ = self.rho(flat_demo_local_graphs)
        ##### indv emb
        flat_demo_rho_batch = demo_node_emb['agent']                  # [B*N*L, A, euc_emb]        
        demo_rho_batch = flat_demo_rho_batch.view(B, N, L, num_agent_nodes, -1)    # [B,N,L,A,euc_emb]


        ############################ Now for current observation ###################################
        ### first get obj embeddings 
        F_list, C_list = [], []
        for i in range(B):
            Fi, Ci = self.geometric_encoder(curr_object_pos[i])  # [M,D], [M,2]
            F_list.append(Fi); C_list.append(Ci)
        curr_scene_feats = torch.stack(F_list, dim=0)  # [B, M, D]
        curr_scene_pos   = torch.stack(C_list, dim=0)  # [B, M, 2]

        ### build local graph
        curr_local_graph_batch = build_local_heterodata_batch(
            agent_pos_b = curr_agent_info,      # [B, A, 6]
            scene_pos_b = curr_scene_pos,       # [B, M, 2]
            scene_feats_b = curr_scene_feats,   # [B, M, D]
            num_freqs = self.euc_dim //4,
            include_agent_agent=False
        )

        ### get rho 
        curr_node_emb, _ = self.rho(curr_local_graph_batch)
        ##### indv emb 
        curr_rho_batch = curr_node_emb['agent'].view(B,num_agent_nodes,-1)     # [B, A, De]



        ### then get curr hyp emb
        ## choose 1 of 3. 

        ### 1 : learnt params
        # curr_hyp_emb = self.curr_hyp_emb.repeat(B,1) # [B,dh]

        ### 2: seed from rho(g) and demo_hyp
        u_agents_now = self.hyp_context(demo_hyp_all, curr_rho_batch, mask=None)   # (B, A, Dh)
        curr_hyp_emb = self._pool_agents_to_batch(u_agents_now, curr_rho_batch) 

        # 3 : rho(g) alignment (not that good)
        # curr_hyp_emb, _  = self.hyp_context(curr_rho_batch, demo_rho_batch, demo_hyp_all)
        
        ### 
        ############################ Then Context Align demos & curr obs (phi) ############################ 
        curr_latent_var = self.context_alignment(
            curr_rho_batch,
            curr_hyp_emb,
            demo_rho_batch,
            demo_hyp_all
        ) # [B, A, z_dim]
       
        # change from here... 
        ############################ Then Actions ############################ 
        pred_obj_info, pred_agent_info = self._perform_reverse_action(actions, curr_object_pos, curr_agent_info)
        

        # with pred info flatten, then make hetero graph 
        B,T,M, _ = pred_obj_info.shape  # T = self.pred_horizon
        flat_pred_obs_info = pred_obj_info.view(B*T, M, -1)
        flat_pred_agent_info = pred_agent_info.view(B*T, num_agent_nodes, -1)
        F_list, C_list = [], []
        for i in range(B*T):
            # one cloud Pi: [M_raw, 2]  (use your raw per-frame points here)
            Pi = flat_pred_obs_info[i]           # [M_raw, 2]
            Fi, Ci = self.geometric_encoder(Pi)    # Fi: [M, D], Ci: [M, 2]
            F_list.append(Fi)
            C_list.append(Ci)
        flat_pred_feats_batch = torch.stack(F_list, dim=0)  # [B*N*L, M, D]
        flat_pred_scene_pos_batch   = torch.stack(C_list, dim=0)  # [B*N*L, M, 2]


        flat_pred_local_graphs = build_local_heterodata_batch(
            agent_pos_b = flat_pred_agent_info,
            scene_pos_b = flat_pred_scene_pos_batch,
            scene_feats_b = flat_pred_feats_batch,
            num_freqs=self.euc_dim // 4,
            include_agent_agent=False 
        )

        ### get pred rho and hyp emb
        pred_node_emb, _ = self.rho(flat_pred_local_graphs)
        pred_rho_batch = pred_node_emb['agent'].view(B,T, num_agent_nodes,-1) # [B, T, A, de] 
        flat_pred_rho_batch = pred_rho_batch.view(B*T, num_agent_nodes, -1) # [B*T, num_agent_nodes, self.euc_dim]
        
        ## choose 1 of 3. like above 

        # 1 learnt param 
        # pred_hyp_emb = self.curr_hyp_emb.repeat(B,T,1) # [B, T, dh] (general approximation, not the best, if dont work well replace with nn.param)
        
        # 2 seed: vectorize over T: merge (B,T) -> BT
        rho_bt   = pred_rho_batch.reshape(B*T, num_agent_nodes, -1)              # (BT, A, de)
        demo_bt  = demo_hyp_all.repeat_interleave(T, dim=0)                       # (BT, N, L, Dh)
        u_agents_seq = self.hyp_context(demo_bt, rho_bt, mask=None)               # (BT, A, Dh)
        flat_pred_hyp_emb = self._pool_agents_to_batch(u_agents_seq, rho_bt)                  # (BT, Dh)  
            
        # 3 align
        # flat_pred_hyp_emb, _  = self.hyp_context(flat_pred_rho_batch, demo_rho_batch, demo_hyp_all)
        # pred_hyp_emb = flat_pred_hyp_emb.view(B,T,-1) 
        
        ####

        # then context allign 

        flat_pred_latent_variables = self.context_alignment(
            flat_pred_rho_batch, # [B*T, A, de ]
            flat_pred_hyp_emb, # [B*T, dh]
            demo_rho_batch,
            demo_hyp_all,
        ) # [B*T, A, z_dim]

        pred_latent_variables = flat_pred_latent_variables.view(B,T,num_agent_nodes,-1) # [B,T, A, z_dim]
        ############################ info Z_current => Z_predicted  ############################ 
        # like in IP 
        # Z_current <= curr_latent_var [B,A,z]
        # Z_predicted <= pred_latent_variables [B,T, A, z_dim] * T = self.pred_horizon
        
        pred_embd = self.foresight_adjustment( 
            curr_latent_var, 
            pred_latent_variables,
            actions,
        ) # [B, T, A, z_dim] propagate info from curr latent var to pred latent var like in IP 
        final_embd = self.intra_action(pred_embd) # let agent nodes within each timestep 'coordinate; amongst themselves

        denoising_direction = self.action_head(final_embd) # [B,T,5] tran_x, tran_y, rot_x, rot_y, state_change

        return denoising_direction

    def _pool_agents_to_batch(self, u_agents: torch.Tensor, rho_g: torch.Tensor) -> torch.Tensor:
        """
        u_agents: (B, A, Dh) in Poincaré ball
        rho_g:    (B, A, de) Euclidean features (weights)
        returns:  (B, Dh)    pooled token in the ball
        """
        # weights from rho(G)
        w = torch.softmax(self.agent_pool_q(rho_g).squeeze(-1), dim=1)   # (B, A)

        # pool in tangent for stability
        u_tan = logmap0(u_agents, self.curvature)                        # (B, A, Dh)
        pooled_tan = torch.einsum('ba,bad->bd', w, u_tan)                # (B, Dh)
        u_batch = expmap0(pooled_tan, self.curvature)                    # (B, Dh)
        return u_batch

    def _perform_reverse_action(self,
                                actions: torch.Tensor,         # [B, T, 4] -> (dx, dy, dtheta_rad, state_action)
                                curr_object_pos: torch.Tensor, # [B, M, 2]
                                curr_agent_info: torch.Tensor  # [B, A, 6]: [ax, ay, cos, sin, grip, state_gate?]
                                ):
        """
        Apply inverse SE(2) to OBJECTS sequentially over time to simulate agent motion backwards.
        Agents do not move here (ax, ay, cos, sin stay fixed); only the gripper/state can change
        *sequentially* based on the per-step desired command.
        """
        B, T, _ = actions.shape
        _, M, _ = curr_object_pos.shape
        _, A, C = curr_agent_info.shape
        device  = curr_object_pos.device
        dtype   = curr_object_pos.dtype

        # ---- split action channels ----
        dxdy   = actions[..., 0:2]            # [B,T,2]
        dtheta = actions[..., 2]              # [B,T]
        sa     = actions[..., 3].clamp(0, 1)  # [B,T] desired gripper/state command (0/1)

        # ---- Prepare outputs ----
        pred_object_pos  = torch.empty(B, T, M, 2, device=device, dtype=dtype)
        pred_agent_info  = torch.empty(B, T, A, 6, device=device, dtype=curr_agent_info.dtype)

        # ---- Static agent info (positions/orientations & optional gate) ----
        ax   = curr_agent_info[..., 0]  # [B,A]
        ay   = curr_agent_info[..., 1]
        cth  = curr_agent_info[..., 2]
        sth  = curr_agent_info[..., 3]
        grip = curr_agent_info[..., 4]  # this will be updated sequentially
        if C >= 6:
            gate = curr_agent_info[..., 5]
        else:
            gate = torch.ones(B, A, device=device, dtype=curr_agent_info.dtype)

        # ---- Sequential roll-back over time ----
        # Start from the current object positions and current grip, step T times
        pos_t   = curr_object_pos  # [B,M,2]
        grip_t  = grip             # [B,A]

        for t in range(T):
            # Objects: p_{t-1} = R(-dθ_t) @ (p_t - [dx_t, dy_t])
            dxdy_bt12 = dxdy[:, t].unsqueeze(1)          # [B,1,2] -> broadcast over M
            delta     = pos_t - dxdy_bt12                # [B,M,2]

            c = torch.cos(-dtheta[:, t]).view(B, 1, 1)   # [B,1,1]
            s = torch.sin(-dtheta[:, t]).view(B, 1, 1)

            x = delta[..., 0:1]                          # [B,M,1]
            y = delta[..., 1:2]                          # [B,M,1]
            x_rot =  c * x + s * y
            y_rot = -s * x + c * y
            pos_t = torch.cat([x_rot, y_rot], dim=-1)    # [B,M,2]

            pred_object_pos[:, t] = pos_t

            # Agents: sequential grip update vs last state
            sa_t = sa[:, t].unsqueeze(-1).expand(B, A)   # [B,A]
            change_mask = (sa_t.round() != grip_t.round())
            grip_t = torch.where(change_mask, sa_t, grip_t)

            # Pack per-t agent info (positions/orientations fixed, grip_t updated)
            pred_agent_info[:, t, :, 0] = ax
            pred_agent_info[:, t, :, 1] = ay
            pred_agent_info[:, t, :, 2] = cth
            pred_agent_info[:, t, :, 3] = sth
            pred_agent_info[:, t, :, 4] = grip_t
            pred_agent_info[:, t, :, 5] = gate

        return pred_object_pos, pred_agent_info