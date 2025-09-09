import torch
import torch.nn as nn

from configs import ANGULAR_GRANULITIES
from utilities import * 

# aux
from .demo_handler import DemoHandler
from .graphs import build_local_heterodata_batch
from .rho import Rho 
from .foresight import IntraActionSelfAttn, PsiForesight
from .pma import ProductManifoldAttention
from .action_head import SimpleActionHead

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
        self.z_dim = num_att_heads * self.hyp_dim + self.euc_dim
        self.angular_granulities = angular_granulities
        

        self.geometric_encoder = geometric_encoder 
        self.rho = Rho(
            in_dim_agent = in_dim_agent, # default by construction, 4 onehot + 5 scalars (sin,cos,state,time,done)
            in_dim_scene = num_att_heads * euc_head_dim,     
            edge_dim     = 10 * 4,        
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
              
    def forward(self,
                curr_agent_info, # [B x self.num_agent_nodes x 6] x, y, theta, state, time, done
                curr_object_pos, # [B x M x 2] x,y
                demo_agent_info, # [B x N x L x self.num_agent_nodes x 6] x, y, theta, state, time, done
                demo_object_pos, # [B x N x L x M x 2]
                actions, # [B, T, 4],
                mode = 'train'   
                ):
        B, N, L, num_agent_nodes, agent_dim = demo_agent_info.shape
        _, _, _, num_object_nodes, obj_pos_dim = demo_object_pos.shape 

        ############################ First process euclidean embeddings ############################
        # get rho(G) for each demo
        flat_demo_agent_info = demo_agent_info.view(B * N * L, num_agent_nodes, agent_dim)
        flat_demo_object_pos = demo_object_pos.view(B * N * L, num_object_nodes, obj_pos_dim)
        
        ### first build local hetero graph
        F_list, C_list = [], []
        for i in range(B*N*L):
            # one cloud Pi: [M_raw, 2]  
            Pi = flat_demo_object_pos[i]           # [M_raw, 2]
            Fi, Ci = self.geometric_encoder(Pi)    # Fi: [M, D], Ci: [M, 2]
            F_list.append(Fi)
            C_list.append(Ci)
        flat_demo_scene_feats_batch = torch.stack(F_list, dim=0)  # [B*N*L, M, D]
        flat_demo_scene_pos_batch   = torch.stack(C_list, dim=0)  # [B*N*L, M, 2]
        flat_demo_local_graphs = build_local_heterodata_batch(
            agent_pos_b = flat_demo_agent_info,
            scene_pos_b=flat_demo_scene_pos_batch,
            scene_feats_b=flat_demo_scene_feats_batch
        ) # returns HeteroBatch[B*N*L]

        ### then get rho(demo)
        demo_node_emb, _ = self.rho(flat_demo_local_graphs)
        ##### indv emb
        flat_demo_rho_batch = demo_node_emb['agent']                  # [B*N*L, A, euc_emb]        
        demo_rho_batch = flat_demo_rho_batch.view(B, N, L, num_agent_nodes, -1)    # [B,N,L,A,euc_emb]

        # then get rho(G) for current observation 
        ### first build local hetero graph 
        F_list, C_list = [], []
        for i in range(B):
            Fi, Ci = self.geometric_encoder(curr_object_pos[i])  # [M,D], [M,2]
            F_list.append(Fi); C_list.append(Ci)
        curr_scene_feats = torch.stack(F_list, dim=0)  # [B, M, D]
        curr_scene_pos   = torch.stack(C_list, dim=0)  # [B, M, 2]
        curr_local_graph_batch = build_local_heterodata_batch(
            agent_pos_b = curr_agent_info,      # [B, A, 6]
            scene_pos_b = curr_scene_pos,       # [B, M, 2]
            scene_feats_b = curr_scene_feats   # [B, M, D]
        )
        ### get rho(current)
        curr_node_emb, _ = self.rho(curr_local_graph_batch)
        curr_rho_batch = curr_node_emb['agent'].view(B,num_agent_nodes,-1)     # [B, A, De]

        # finally for actions 
        pred_agent_info = None
        pred_obj_info = None 
        if mode == 'eval':
            pred_obj_info, pred_agent_info = self._perform_reverse_action_seq(actions, curr_object_pos, curr_agent_info)
        else:
            pred_obj_info, pred_agent_info = self._perform_reverse_action(actions, curr_object_pos, curr_agent_info)
        
        # with pred info flatten, then make hetero graph 
        B,T,M, _ = pred_obj_info.shape  # T = self.pred_horizon
        flat_pred_obs_info = pred_obj_info.view(B*T, M, -1)
        flat_pred_agent_info = pred_agent_info.view(B*T, num_agent_nodes, -1)
        F_list, C_list = [], []
        for i in range(B*T):
            # one cloud Pi: [M_raw, 2]  
            Pi = flat_pred_obs_info[i]           # [M_raw, 2]
            Fi, Ci = self.geometric_encoder(Pi)    # Fi: [M, D], Ci: [M, 2]
            F_list.append(Fi)
            C_list.append(Ci)
        flat_pred_feats_batch = torch.stack(F_list, dim=0)  # [B*N*L, M, D]
        flat_pred_scene_pos_batch   = torch.stack(C_list, dim=0)  # [B*N*L, M, 2]
        flat_pred_local_graphs = build_local_heterodata_batch(
            agent_pos_b = flat_pred_agent_info,
            scene_pos_b = flat_pred_scene_pos_batch,
            scene_feats_b = flat_pred_feats_batch
        )

        ### get pred rho 
        pred_node_emb, _ = self.rho(flat_pred_local_graphs)
        pred_rho_batch = pred_node_emb['agent'].view(B,T, num_agent_nodes,-1) # [B, T, A, de] 
        flat_pred_rho_batch = pred_rho_batch.view(B*T, num_agent_nodes, -1) # [B*T, num_agent_nodes, self.euc_dim]
        
        ############################ Now for hyperbolic embeddings ###################################

        # now we have curr_agent_pos [B, A, agent_dim], pred_agent_pos [B,T,A,agent_dim] and demo_agent_pos [B,N,L,A,agent_dim]
        # we build SK tree by concatinating curr_agent_pos | pred_agent_pos | deo_agent_pos along L dimension, eseentially treating
        # curr_obs and pred_actions as starting states.  
        _buffer = torch.zeros(B,N,1,num_agent_nodes,agent_dim)
        _curr_agent_info_temp = curr_agent_info.view(B,1,1,num_agent_nodes, agent_dim).repeat(1,N,1,1,1)
        _final_data = torch.concat([_buffer, _curr_agent_info_temp, demo_agent_info], dim = 2) #(B,N,L+T+2,A,6)
        hyperbolic_embeddings = self.demo_handler(_final_data)[:,:,1:,:] # [B,N,L+T+2,dh]

        curr_hyp_emb = hyperbolic_embeddings[:,:,0,:].view(B,N,-1) # choose form first demo they will be the same 
        pred_hyp_emb = curr_hyp_emb.clone().view(B,1,-1).repeat(1,self.pred_horizon,1) # choose form first demo they will be the same smentically
        demo_hyp_all = hyperbolic_embeddings[:,:,1:,:].view(B,N,L,-1)


        ############################ Then 'project' to Product manifold space ############################ 
        curr_latent_var = self.context_alignment(
            curr_rho_batch,
            curr_hyp_emb,
            demo_rho_batch,
            demo_hyp_all
        ) # [B, A, z_dim]
       
        # 5) Future steps: flatten B and T together, but KEEP N
        #    - Pred queries:  [B*T, N, dh]
        #    - Demos:         need to be repeated across T -> [B*T, N, L, A, de], [B*T, N, L, dh]
        flat_pred_hyp_emb   = pred_hyp_emb.view(B*T, N, -1)  # [B*T, N, dh]
        demo_rho_bt         = demo_rho_batch.repeat_interleave(T, dim=0)  # [B*T, N, L, A, de]
        demo_hyp_bt         = demo_hyp_all.repeat_interleave(T, dim=0)    # [B*T, N, L, dh]

        # 6) Flattened pred Euclidean queries 
        flat_pred_latent_variables = self.context_alignment(
            flat_pred_rho_batch,   # [B*T, A, de]
            flat_pred_hyp_emb,     # [B*T, N, dh]  <-- KEEP N here
            demo_rho_bt,           # [B*T, N, L, A, de]
            demo_hyp_bt            # [B*T, N, L, dh]
        )  # -> [B*T, A, z_dim]
        pred_latent_variables = flat_pred_latent_variables.view(B, T, num_agent_nodes, -1) 
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
        if torch.all(torch.isnan(denoising_direction)):
            breakpoint()
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
        u_tan = poincare_log0(u_agents, self.curvature)                        # (B, A, Dh)
        pooled_tan = torch.einsum('ba,bad->bd', w, u_tan)                # (B, Dh)
        u_batch = poincare_exp0(pooled_tan, self.curvature)                    # (B, Dh)
        return u_batch

        
    # Apply inverse SE(2) to OBJECTS ONLY (non-cumulative per t).
    def _perform_reverse_action(
        self,
        actions: torch.Tensor,         # [B, T, 4] -> (dx, dy, dtheta_rad, state_action)
        curr_object_pos: torch.Tensor, # [B, M, 2]
        curr_agent_info: torch.Tensor  # [B, A, 6]: [ax, ay, cos, sin, grip, state_gate?]
    ):

        B, T, _ = actions.shape
        _, M, _ = curr_object_pos.shape
        _, A, C = curr_agent_info.shape
        device  = curr_object_pos.device

        # split action channels  
        dxdy   = actions[..., 0:2]            
        dtheta = actions[..., 2]              
        sa     = actions[..., 3].clamp(0, 1)  

        #  objects: inverse SE(2) per t, from the same base pose  
        dxdy_bt12 = dxdy.unsqueeze(2)                
        obj_b1m2  = curr_object_pos.unsqueeze(1)     
        delta     = obj_b1m2 - dxdy_bt12             

        c = torch.cos(dtheta).unsqueeze(-1).unsqueeze(-1)   
        s = torch.sin(dtheta).unsqueeze(-1).unsqueeze(-1)

        x = delta[..., 0:1]                            
        y = delta[..., 1:2]
        # R(theta)^T application:
        x_rot =  c * x + s * y
        y_rot = -s * x + c * y
        pred_object_pos = torch.cat([x_rot, y_rot], dim=-1)  

        # agent: positions/orientations fixed across time 
        ax   = curr_agent_info[..., 0].unsqueeze(1).expand(B, T, A) 
        ay   = curr_agent_info[..., 1].unsqueeze(1).expand(B, T, A)
        cth  = curr_agent_info[..., 2].unsqueeze(1).expand(B, T, A)
        sth  = curr_agent_info[..., 3].unsqueeze(1).expand(B, T, A)
        grip = curr_agent_info[..., 4].unsqueeze(1).expand(B, T, A)

        if C >= 6:
            gate = curr_agent_info[..., 5].unsqueeze(1).expand(B, T, A)
        else:
            gate = torch.ones(B, T, A, device=device, dtype=curr_agent_info.dtype)

        # Per-step grip output: set to command if different from current; otherwise keep current
        sa_bta = sa.unsqueeze(-1).expand(B, T, A)             
        change_mask = (sa_bta.round() != grip.round())        
        grip_out = torch.where(change_mask, sa_bta, grip)     

        pred_agent_info = torch.stack([ax, ay, cth, sth, grip_out, gate], dim=-1) 
        return pred_object_pos, pred_agent_info


    def _perform_reverse_action_seq(self,
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

        #  split action channels 
        dxdy   = actions[..., 0:2]             
        dtheta = actions[..., 2]               
        sa     = actions[..., 3].clamp(0, 1)   

        #  Prepare outputs 
        pred_object_pos  = torch.empty(B, T, M, 2, device=device, dtype=dtype)
        pred_agent_info  = torch.empty(B, T, A, 6, device=device, dtype=curr_agent_info.dtype)

        #  Static agent info (positions/orientations & optional gate) 
        ax   = curr_agent_info[..., 0]   
        ay   = curr_agent_info[..., 1]
        cth  = curr_agent_info[..., 2]
        sth  = curr_agent_info[..., 3]
        grip = curr_agent_info[..., 4]   
        if C >= 6:
            gate = curr_agent_info[..., 5]
        else:
            gate = torch.ones(B, A, device=device, dtype=curr_agent_info.dtype)

        #  Sequential roll-back over time 
        # Start from the current object positions and current grip, step T times
        pos_t   = curr_object_pos  
        grip_t  = grip             

        for t in range(T):
            # Objects: p_{t-1} = R(-dtheta_t) @ (p_t - [dx_t, dy_t])
            dxdy_bt12 = dxdy[:, t].unsqueeze(1)         
            delta     = pos_t - dxdy_bt12               

            c = torch.cos(-dtheta[:, t]).view(B, 1, 1)  
            s = torch.sin(-dtheta[:, t]).view(B, 1, 1)

            x = delta[..., 0:1]                         
            y = delta[..., 1:2]                         
            x_rot =  c * x + s * y
            y_rot = -s * x + c * y
            pos_t = torch.cat([x_rot, y_rot], dim=-1)   

            pred_object_pos[:, t] = pos_t

            # Agents: sequential grip update vs last state
            sa_t = sa[:, t].unsqueeze(-1).expand(B, A)   
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
