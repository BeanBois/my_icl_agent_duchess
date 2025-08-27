import torch
import torch.nn as nn

from configs import ANGULAR_GRANULITIES
from utilities import * 

# aux
from .demo_handler import DemoHandler
from .my_local_graph import build_local_heterodata_batch
from .rho import Rho 
from .alignment import EuclidToHypAlign
from utilities import next_proceeding_strict, geodesic_segment, time_posterior
from .dynamic import HyperbolicDynamics
from .prod_cross_attn import ProductCrossAttention
from .action_head import ProductManifoldGPHead

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
        self.emb_dim_euc = num_att_heads * euc_head_dim
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

        self.context_alignment = EuclidToHypAlign(
            d_e = num_att_heads * euc_head_dim,
            d_h = 2,
            heads = num_att_heads
        )
        
        self.dynamic = HyperbolicDynamics(
            d_h = 2,
            hidden = 256,
            c = self.curvature
        )

        self.pca = ProductCrossAttention(
            num_heads=num_att_heads,
            euc_head_dim= euc_head_dim,
            hyp_dim = 2,
            z_dim = num_att_heads * euc_head_dim,
            curvature=curvature,
            tau =tau,
        )

        self.trajectory_predictor = ProductManifoldGPHead(
            action_dim=4,
            z_dim = num_att_heads * euc_head_dim,
            M = 128,
        )
        
    def forward(self,
                curr_agent_info, # [B x self.num_agent_nodes x 6] x, y, theta, state, time, done
                curr_object_pos, # [B x M x 2] x,y
                demo_agent_info, # [B x N x L x self.num_agent_nodes x 6] x, y, theta, state, time, done
                demo_object_pos, # [B x N x L x M x 2]
                ):

        # First process demos into hyperbolic embeddings
        B, N, L, num_agent_nodes, agent_dim = demo_agent_info.shape
        _, _, _, num_object_nodes, obj_pos_dim = demo_object_pos.shape 

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
            num_freqs=self.emb_dim_euc // 4,
            include_agent_agent=False # no agent-agent edges 
        ) # returns HeteroBatch[B*N*L]

        ### get rho(G) for demos 
        demo_node_emb, _ = self.rho(flat_demo_local_graphs)
        ##### indv emb
        flat_demo_rho_batch = demo_node_emb['agent']                  # [B*N*L, A, euc_emb]
        demo_rho_batch = flat_demo_rho_batch.view(B, N, L, num_agent_nodes, -1)    # [B,N,L,A,euc_emb]


        # Now for current observation and action

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
            num_freqs = self.emb_dim_euc //4,
            include_agent_agent=False
        )

        ### get rho 
        curr_node_emb, _ = self.rho(curr_local_graph_batch)
        ##### indv emb 
        curr_rho_batch = curr_node_emb['agent']     # [B, A, De]
        if len(curr_rho_batch.shape) == 2:
            temp = curr_rho_batch.shape
            curr_rho_batch = curr_rho_batch.view(1,*temp)
        ### then do context alignment by searching for curr_hyp_embedding within SK tree with Euc embeddings 
        ### euclidean keys/values from demos:
        curr_hyp_est, attn = self.context_alignment(
            curr_rho_batch, 
            demo_rho_batch,
            demo_hyp_all
        ) # [B, hyp_dim], [B, num_agent_nodes, num_demos, demo_length]


        # Decode actions from curr emb to next emb
        ### If using learnt dynamics
        
        h_seq = [curr_hyp_est]
        for t in range(self.pred_horizon):
            h_seq.append(self.dynamic(h_seq[-1]))
        path_hyp = torch.stack(h_seq, dim = 1)

        # Product Cross Attn to create latent variables
        # 1) path_hyp is of shape (B, self.pred_horizon + 1, 2)
        # 2) curr_rho(.) is of shape(B, A, euc_dim = self.num_attn_head * self.euc_head_dim)
        # 3) ProCrosAttn (PCA) takes in these are combine them to form meaningful representation like in paper 
        # 4) output should be (B, self.pred_horizon + 1,  z_dim (parameter)
        z_seq, attn_seq = self.pca(
            hyp_seq = path_hyp,
            rho_ctx = curr_rho_batch
        )
        z_t      = z_seq[:, :-1, :]        # [B, T, Z]
        h_tp1    = path_hyp[:, 1:, :]      # [B, T, 2]
        v_tp1    = logmap0(h_tp1, self.curvature)  # [B, T, 2]  (optional, keeps things Euclidean)

        # x_t = torch.cat([z_t, v_tp1], dim=-1)      # [B, T, Z+2]
        actions = self.trajectory_predictor(h_tp1, z_t)   # define this -> [B, T, action_dim]

        # v_tan = logmap0(path_hyp, self.curvature) # [B, T + 1, hyp_dim]
        # vel = v_tan[:, 1:, :] - v_tan[:, :-1,:] # [B, self.pred_horizon, hyp_dim]
        # actions = self.low_level_action_head(vel) # [B, self.pred_horizon, 3]

        # return actions
        return {
            'curr_rho' : curr_rho_batch,
            'demo_rho' : demo_rho_batch,
            'curr_hyp' : curr_hyp_est,
            'demo_hyp' : demo_hyp_all,
            'attn_e2e' : attn,
            'pred_actions' : actions
        }

    def predict(self):
        pass 