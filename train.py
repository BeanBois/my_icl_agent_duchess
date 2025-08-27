import os
import math
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np


from agent import Policy  


# ---------------------------
# Small hyperbolic utilities
# (Poincaré ball; curvature c>0)
# ---------------------------
class Poincare:
    @staticmethod
    def mobius_add(x, y, c):
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        den = 1 + 2*c*xy + c*c*x2*y2
        return num / torch.clamp(den, min=1e-15)

    @staticmethod
    def lambda_x(x, c):
        return 2.0 / torch.clamp(1 - c*(x*x).sum(dim=-1, keepdim=True), min=1e-15)

    @staticmethod
    def poincare_dist(x, y, c, eps=1e-15):
        # d_c(x,y) = arcosh(1 + 2c||x-y||^2 / ((1 - c||x||^2)(1 - c||y||^2)))
        x2 = torch.clamp((x*x).sum(dim=-1), max=(1.0 - 1e-6)/c)  # keep inside ball
        y2 = torch.clamp((y*y).sum(dim=-1), max=(1.0 - 1e-6)/c)
        diff2 = ((x - y)**2).sum(dim=-1)
        num = 2*c*diff2
        den = (1 - c*x2) * (1 - c*y2)
        z = 1 + torch.clamp(num / torch.clamp(den, min=eps), min=1+1e-7)
        return torch.acosh(z)

    @staticmethod
    def project_to_ball(x, c, eps=1e-5):
        # keep inside radius 1/sqrt(c)
        r = 1.0 / math.sqrt(c)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
        scale = torch.clamp((r - eps)/norm, max=1.0)
        return x * scale


# ---------------------------
# Data interface (stub)
# Replace with your real dataset that returns tensors
# ---------------------------
@dataclass
class Item:
    # Shapes must match your policy.forward signature
    curr_agent_info: torch.Tensor       # [B, A, 6]
    curr_object_pos: torch.Tensor       # [B, M, 2]
    clean_actions: torch.Tensor         # [B, T, 3] (or your action dim)
    demo_agent_info: torch.Tensor       # [B, N, L, A, 6]
    demo_object_pos: torch.Tensor       # [B, N, L, M, 2]
    demo_agent_action: torch.Tensor     # [B, N, L-1, 3]
    # Optional/time channels you may already export:
    demo_time: Optional[torch.Tensor] = None  # [B, N, L] monotonically increasing
    curr_time: Optional[torch.Tensor] = None  # [B] or [B, A] if per-node
 
from data import PseudoDemoGenerator

class PseudoDemoDataset(Dataset):
    def __init__(self, length=10000, device="cpu",
                 B=4, A=4, M=64, N=2, L=10, T=8, action_dim=3):
        self.length = length
        self.device = device
        self.B, self.A, self.M, self.N, self.L, self.T = B, A, M, N, L, T
        self.action_dim = action_dim
        self.data_gen = PseudoDemoGenerator(device, num_demos = self.N + 1, demo_length = self.L, pred_horizon = self.T)

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> Item:
        B, A, M, N, L, T, ad = self.B, self.A, self.M, self.N, self.L, self.T, self.action_dim
        # Dummy random sample — replace with your real loading logic.
        curr_agent_info = torch.randn(B, A, 6)
        curr_object_pos = torch.randn(B, M, 2)
        clean_actions   = torch.randn(B, T, ad)

        demo_agent_info   = torch.randn(B, N, L, A, 6)
        demo_object_pos   = torch.randn(B, N, L, M, 2)
        demo_agent_action = torch.randn(B, N, L-1, ad)

        curr_obs, context, _clean_actions = self.data_gen.get_batch_samples(self.B)
        curr_agent_info, curr_object_pos = self._process_obs(curr_obs)
        demo_agent_info, demo_object_pos, demo_agent_action = self._process_context(context)
        clean_actions = self._process_actions(_clean_actions)

        # Monotone times for each demo traj
        base = torch.arange(L).float()[None, None, :].expand(B, N, L)
        noise = 0.01*torch.randn_like(base)
        demo_time = base + noise
        curr_time = (L-1) * torch.ones(B)  # treat current as “after” the last demo waypoint if you prefer

        return Item(
            curr_agent_info, curr_object_pos, clean_actions,
            demo_agent_info, demo_object_pos, demo_agent_action,
            demo_time=demo_time, curr_time=curr_time
        )

    def _process_obs(self, curr_obs: List[Dict]):
        """
        curr_obs: list length B. Each element is a list of observation dicts
                  (from PDGen._get_ground_truth: 'curr_obs_set').
        We take the FIRST obs of each sample as the 'current' one and turn it
        into tensors:
          - curr_agent_info: [B, A=4, 6] with [x,y,orientation,state,time,done]
          - curr_object_pos: [B, M, 2]  sampled coords
        """
        B, A, M = self.B, self.A, self.M
        device = self.device

        # fixed keypoint order (matches 4 nodes expected by A=4)
        kp_order = ["front", "back-left", "back-right", "center"]
        kp_local = [PseudoDemoGenerator.agent_keypoints[k] for k in kp_order]  # local-frame offsets
        kp_local = torch.tensor(kp_local, dtype=torch.float32, device=device)  # [4,2]

        agent_infos = []
        obj_coords_all = []

        for b in range(B):
            # Use the first "current" obs for this sample
            ob = curr_obs[b][0] if isinstance(curr_obs[b], list) else curr_obs[b]

            # Scalars
            cx, cy = float(ob["agent-pos"][0][0]), float(ob["agent-pos"][0][1])
            ori_deg = float(ob["agent-orientation"])
            ori_rad = math.radians(ori_deg)
            st = ob["agent-state"]
            st_val = float(getattr(st, "value", st))  # enum -> int if needed
            t_val = float(ob["time"])
            done_val = float(bool(ob["done"]))

            # Rotate local KPs to world and translate by agent center
            c, s = math.cos(ori_rad), math.sin(ori_rad)
            R = torch.tensor([[c, -s],
                              [s,  c]], dtype=torch.float32, device=device)     # [2,2]
            kp_world = (kp_local @ R.T) + torch.tensor([cx, cy], device=device)  # [4,2]

            # Pack [x,y,orientation,state,time,done] per keypoint
            o = torch.full((A, 1), ori_deg, dtype=torch.float32, device=device)
            stt = torch.full((A, 1), st_val, dtype=torch.float32, device=device)
            tt = torch.full((A, 1), t_val, dtype=torch.float32, device=device)
            dd = torch.full((A, 1), done_val, dtype=torch.float32, device=device)
            agent_info = torch.cat([kp_world, o, stt, tt, dd], dim=1)  # [4,6]
            agent_infos.append(agent_info)

            # Object coords → pick exactly M 2D points
            coords_np = ob["coords"]  # numpy array [K,2] (possibly K != M)
            K = int(coords_np.shape[0])
            if K == 0:
                # nothing detected → zeros
                sel = torch.zeros((M, 2), dtype=torch.float32, device=device)
            elif K >= M:
                idx = np.random.choice(K, size=M, replace=False)
                sel = torch.tensor(coords_np[idx], dtype=torch.float32, device=device)
            else:
                # not enough points → repeat with replacement
                idx = np.random.choice(K, size=M, replace=True)
                sel = torch.tensor(coords_np[idx], dtype=torch.float32, device=device)
            obj_coords_all.append(sel)

        curr_agent_info = torch.stack(agent_infos, dim=0)  # [B,4,6]
        curr_object_pos = torch.stack(obj_coords_all, dim=0)  # [B,M,2]
        return curr_agent_info, curr_object_pos

    def _process_actions(self, _clean_actions: List[List[torch.Tensor]]):
        """
        _clean_actions: list length B; each element is a LIST of length >=1,
                        where each entry is a [T, 10] tensor:
                          9 numbers = row-major SE(2) (3x3), then 1 gripper/state.
        We return [B, T, 3] with (tx, ty, theta) for the FIRST horizon sequence.
        """
        B, T = self.B, self.T
        device = self.device

        def mat_to_vec(m9: torch.Tensor) -> torch.Tensor:
            # m9 [9] row-major -> tx,ty,theta(rad)
            M = m9.view(3, 3)
            tx = M[0, 2]
            ty = M[1, 2]
            theta = torch.atan2(M[1, 0], M[0, 0])
            return torch.stack([tx, ty, theta], dim=0)  # [3]

        out = []
        for b in range(B):
            # take the first pred-horizon sequence for this sample
            seq = _clean_actions[b][0]  # [T, 10] on same device as generator set
            # Robustness: pad/truncate to T if needed
            Tb = seq.shape[0]
            if Tb < T:
                pad = torch.zeros((T - Tb, seq.shape[1]), dtype=seq.dtype, device=seq.device)
                seq = torch.cat([seq, pad], dim=0)
            elif Tb > T:
                seq = seq[:T]

            # Convert each step
            vecs = []
            for t in range(T):
                m9 = seq[t, :9]  # first 9 entries are SE(2)
                state_action = seq[t,-1].view(1)
                _vec = mat_to_vec(m9)
                vec = torch.concat([_vec,state_action])
                vecs.append(vec)
            vecs = torch.stack(vecs, dim=0).to(device)  # [T,3]
            out.append(vecs)

        return torch.stack(out, dim=0)  # [B,T,3]

    def _process_context(self, context: List[Tuple]):
        """
        context: list length B; each element is a LIST of N demos.
                 Each demo is (observations, actions) where:
                   - observations: list length L of obs dicts (already downsampled in PDGen)
                   - actions: tensor [L-1, 10] (accumulated, already downsampled in PDGen)
        Returns:
          demo_agent_info  : [B, N, L, A=4, 6]
          demo_object_pos  : [B, N, L, M, 2]
          demo_agent_action: [B, N, L-1, 3]  (tx, ty, theta)
        """
        B, N, L, A, M = self.B, self.N, self.L, self.A, self.M
        device = self.device

        kp_order = ["front", "back-left", "back-right", "center"]
        kp_local = [PseudoDemoGenerator.agent_keypoints[k] for k in kp_order]
        kp_local = torch.tensor(kp_local, dtype=torch.float32, device=device)  # [4,2]

        # Containers
        all_demo_agent_info = []
        all_demo_obj = []
        all_demo_act = []

        def obs_to_agent_info(ob):
            cx, cy = float(ob["agent-pos"][0][0]), float(ob["agent-pos"][0][1])
            ori_deg = float(ob["agent-orientation"])
            ori_rad = math.radians(ori_deg)
            st = ob["agent-state"]
            st_val = float(getattr(st, "value", st))
            t_val = float(ob["time"])
            done_val = float(bool(ob["done"]))

            c, s = math.cos(ori_rad), math.sin(ori_rad)
            R = torch.tensor([[c, -s],
                              [s,  c]], dtype=torch.float32, device=device)
            kp_world = (kp_local @ R.T) + torch.tensor([cx, cy], dtype=torch.float32, device=device)  # [4,2]

            o = torch.full((A, 1), ori_deg, dtype=torch.float32, device=device)
            stt = torch.full((A, 1), st_val, dtype=torch.float32, device=device)
            tt = torch.full((A, 1), t_val, dtype=torch.float32, device=device)
            dd = torch.full((A, 1), done_val, dtype=torch.float32, device=device)
            return torch.cat([kp_world, o, stt, tt, dd], dim=1)  # [4,6]

        def mat_to_vec(m9: torch.Tensor) -> torch.Tensor:
            M = m9.view(3, 3)
            tx = M[0, 2]
            ty = M[1, 2]
            theta = torch.atan2(M[1, 0], M[0, 0])
            return torch.stack([tx, ty, theta], dim=0)  # [3]

        for b in range(B):
            demos = context[b]  # list of N demos
            assert len(demos) == N, f"Expected {N} demos, got {len(demos)}"

            demo_infos = []
            demo_objs = []
            demo_acts = []

            for n in range(N):
                observations, actions = demos[n]  # observations: list L; actions: [L-1,10] torch
                # Observations → [L, A, 6] and [L, M, 2]
                agent_info_steps = []
                obj_steps = []
                for l in range(L):
                    if l >= len(observations):
                        agent_info_steps.append(torch.zeros((A, 6), dtype=torch.float32, device=device))
                        obj_steps.append(torch.zeros((M, 2), dtype=torch.float32, device=device))
                        continue
                    ob = observations[l]
                    agent_info_steps.append(obs_to_agent_info(ob))

                    coords_np = ob["coords"]
                    K = int(coords_np.shape[0])
                    if K == 0:
                        sel = torch.zeros((M, 2), dtype=torch.float32, device=device)
                    elif K >= M:
                        idx = np.random.choice(K, size=M, replace=False)
                        sel = torch.tensor(coords_np[idx], dtype=torch.float32, device=device)
                    else:
                        idx = np.random.choice(K, size=M, replace=True)
                        sel = torch.tensor(coords_np[idx], dtype=torch.float32, device=device)
                    obj_steps.append(sel)

                demo_infos.append(torch.stack(agent_info_steps, dim=0))  # [L,4,6]
                demo_objs.append(torch.stack(obj_steps, dim=0))          # [L,M,2]

                # Actions → [L-1, 3]
                act = actions  # [L-1,10]
                # robust pad/truncate to L-1 in case
                if act.shape[0] < L - 1:
                    pad = torch.zeros((L - 1 - act.shape[0], act.shape[1]),
                                      dtype=act.dtype, device=act.device)
                    act = torch.cat([act, pad], dim=0)
                elif act.shape[0] > L - 1:
                    act = act[:L - 1]

                vecs = []
                for i in range(act.shape[0]):
                    vecs.append(mat_to_vec(act[i, :9].to(device)))  # [3]
                demo_acts.append(torch.stack(vecs, dim=0))  # [L-1,3]

            all_demo_agent_info.append(torch.stack(demo_infos, dim=0))  # [N,L,4,6]
            all_demo_obj.append(torch.stack(demo_objs, dim=0))          # [N,L,M,2]
            all_demo_act.append(torch.stack(demo_acts, dim=0))          # [N,L-1,3]

        demo_agent_info = torch.stack(all_demo_agent_info, dim=0)  # [B,N,L,4,6]
        demo_object_pos = torch.stack(all_demo_obj, dim=0)         # [B,N,L,M,2]
        demo_agent_action = torch.stack(all_demo_act, dim=0)       # [B,N,L-1,3]
        return demo_agent_info, demo_object_pos, demo_agent_action



def collate_items(batch: List[Item]) -> Item:
    # Here each dataset __getitem__ already returns batched B samples;
    # if yours returns single samples, stack along dim 0 here.
    assert len(batch) == 1, "This stub returns batch-already tensors; adjust as needed."
    return batch[0]

# ---------------------------
# Losses
# ---------------------------
class AlignmentLoss(nn.Module):
    """
    InfoNCE-style product-space alignment between (curr_euc, demo_euc)
    with a *time-consistency mask* restricting positives to valid phases.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self,
                curr_euc: torch.Tensor,       # [B, A, de]
                demo_euc: torch.Tensor,       # [B, N, L, A, de]
                curr_time: Optional[torch.Tensor], # [B] or [B, A]
                demo_time: Optional[torch.Tensor], # [B, N, L]
                time_window: float = 1.5):
        # Inside AlignmentLoss.forward(...)
        B, A, de = curr_euc.shape
        N, L = demo_euc.shape[1], demo_euc.shape[2]

        qn = nn.functional.normalize(curr_euc, dim=-1)                    # [B, A, de]
        kn = nn.functional.normalize(demo_euc.view(B, N*L*A, de), dim=-1) # [B, NLA, de]
        # kn^T: [B, de, NLA]; qn: [B, A, de] -> sims: [B, A, NLA]
        sims = torch.einsum('bad,bdk->bak', qn, kn.transpose(1, 2))
        sims = sims / self.tau


        # Build positives via nearest in time (|t_demo - t_curr| <= time_window).
        if (curr_time is not None) and (demo_time is not None):
            # shape handling
            if curr_time.ndim == 1:
                curr_time = curr_time[:, None].expand(B, A)  # [B, A]
            demo_time_flat = demo_time[:, :, :, None].expand(B, N, L, A).reshape(B, N*L*A)  # [B, NLA]

            time_diff = demo_time_flat[:, None, :] - curr_time[:, :, None]  # [B, A, NLA]
            pos_mask = (time_diff.abs() <= time_window).float()             # [B, A, NLA]
        else:
            # fallback: same demo index if you track that; otherwise allow all (weak)
            pos_mask = torch.ones_like(sims)

        # Avoid degenerate all-zero mask — if no valid positives, select the max-sim as pseudo-positive
        needs_fallback = (pos_mask.sum(-1) == 0)
        if needs_fallback.any():
            idx = sims.argmax(dim=-1, keepdim=True)  # [B, A, 1]
            pos_mask = pos_mask.scatter(-1, idx, 1.0)

        # InfoNCE with masked positives (multi-positive)
        log_probs = sims - torch.logsumexp(sims, dim=-1, keepdim=True)  # [B, A, NLA]
        # average over positives
        pos_logprob = (log_probs * pos_mask).sum(-1) / torch.clamp(pos_mask.sum(-1), min=1.0)
        loss = -(pos_logprob).mean()
        return loss


class NextStateHyperbolicLoss(nn.Module):
    """
    h_curr: [B, dh]  (one current latent per sample)
    demo_h: [B, N, L, dh]
    align_idx: [B, A, 2] -> per-agent (n_idx, l_idx) anchors
    We pull h_curr[b] toward each agent's next target demo_h[b, n, l+lookahead]
    and optionally push away later steps on the same (b, n).
    """
    def __init__(self, curvature=1.0, margin=0.2, neg_weight=0.5):
        super().__init__()
        self.c = curvature
        self.margin = margin
        self.neg_weight = neg_weight

    def forward(
        self,
        h_curr: torch.Tensor,        # [B, dh]
        demo_h: torch.Tensor,        # [B, N, L, dh]
        align_idx: torch.Tensor,     # [B, A, 2] -> (n_idx, l_idx)
        lookahead: int = 2
    ) -> torch.Tensor:
        device = h_curr.device
        B, dh = h_curr.shape
        _, N, L, _ = demo_h.shape
        A = align_idx.shape[1]
        # Extract per-agent indices
        n0 = align_idx[..., 0].long()                # [B, A]
        l0 = align_idx[..., 1].long()                # [B, A]
        lt = l0 + int(lookahead)                     # [B, A]
        valid = (lt >= 0) & (lt < L)                 # [B, A]

        if valid.sum().item() == 0:
            return h_curr.new_tensor(0.0)

        # Build (b,a) grids and filter valid
        b_grid = torch.arange(B, device=device).unsqueeze(1).expand(B, A)  # [B, A]
        b_v = b_grid[valid]          # [V]
        n_v = n0[valid]              # [V]
        lt_v = lt[valid]             # [V]

        # Gather positives: target next-step embeddings
        h_tgt = demo_h[b_v, n_v, lt_v, :]            # [V, dh]
        h_now = h_curr[b_v, :]                       # [V, dh]

        # Geodesic positive distances per (b,a)
        d_pos = Poincare.poincare_dist(h_now, h_tgt, self.c)  # [V]

        # Average positives over all valid (b,a)
        pos_term = d_pos.mean()

        # Negatives: for each valid (b,a), later steps on same (b, n)
        neg_terms = []
        m = self.margin
        # We’ll iterate over valid pairs (variable K per pair).
        # This is cheap for typical L.
        for i in range(b_v.shape[0]):
            b = int(b_v[i].item())
            n = int(n_v[i].item())
            start = int(lt_v[i].item()) + 1
            if start < L:
                negs = demo_h[b, n, start:, :]  # [K, dh]
                # Compare same h_curr[b] against all negs
                d_negs = Poincare.poincare_dist(h_curr[b].expand_as(negs), negs, self.c)  # [K]
                # margin ranking: max(0, m - d_neg + d_pos_i)
                # neg_terms.append(torch.relu(m - d_negs + d_pos[i].detach()).mean())
                neg_terms.append(torch.relu(m - d_negs + d_pos[i]).mean())

        neg_term = torch.stack(neg_terms).mean() if len(neg_terms) > 0 else h_curr.new_tensor(0.0)

        return pos_term + self.neg_weight * neg_term


class ActionTrajectoryLoss(nn.Module):
    """
    Supervised action loss (e.g., MSE on low-level deltas).
    Replace with SE(3) geodesic + gripper BCE if that’s your format.
    """
    def __init__(self, w_pos=1.0):
        super().__init__()
        self.w_pos = w_pos
        self.mse = nn.MSELoss()

    def forward(self, pred_actions: torch.Tensor, gt_actions: torch.Tensor, max_translation):
        gt_norm = gt_actions.clone()
        gt_norm[:, :, 0]  = gt_actions[:, :, 0]/max_translation
        gt_norm[:, :, 1]  = gt_actions[:, :, 1]/max_translation
        gt_norm[:, :, 2]  = gt_actions[:, :, 2]/(torch.pi)

        # pred_actions: [B, T, da], gt_actions: [B, T, da]
        return self.mse(pred_actions, gt_norm) * self.w_pos


# ---------------------------
# Trainer
# ---------------------------
@dataclass
class TrainConfig:
    device: str = "cpu"
    batch_size: int = 1      # Each dataset item already contains an internal B; keep 1 here for the stub
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_steps: int = 50000
    log_every: int = 50
    ckpt_every: int = 1000
    out_dir: str = "./checkpoints"
    grad_clip: float = 1.0
    amp: bool = True
    align_temp: float = 0.07
    hyp_curvature: float = 1.0
    hyp_margin: float = 0.2
    hyp_neg_w: float = 0.5
    time_window: float = 1.5   # for alignment positives
    lookahead: int = 1      # next-search step size
    num_sampled_pc = 8
    num_att_heads = 4
    euc_head_dim = 16
    hyp_dim = 2
    in_dim_agent = 9
    tau=0.5
    pred_horizon = 5
    demo_length = 20
    max_translation = 1000
    

    # flags
    train_geo_encoder = False


def select_time_consistent_indices(sim_scores: torch.Tensor,
                                   demo_time: torch.Tensor,
                                   curr_time: torch.Tensor) -> torch.Tensor:
    """
    Pick (n_idx, l_idx) per (b,a) as the *best* demo step with |Δt| <= window *and* l must be <= last valid.
    sim_scores: [B, A, N, L] attention scores (higher is better)
    demo_time:  [B, N, L]
    curr_time:  [B] or [B, A]
    Returns: idx [B, A, 2] (n_idx, l_idx)
    """
    B, A, N, L = sim_scores.shape
    if curr_time.ndim == 1:
        curr_time = curr_time[:, None].expand(B, A)
    # Score mask by time proximity (<= window); if all invalid, fallback to argmax over all.
    # Here we don’t apply a numeric window; instead we simply argmax (you can add a finite window if you prefer).
    # For your earlier request, the *training loss* enforces forward-only next step; alignment can be soft.
    idx_out = torch.zeros(B, A, 2, dtype=torch.long, device=sim_scores.device)
    with torch.no_grad():
        # Flatten (N,L)
        sim_flat = sim_scores.view(B, A, N*L)
        best = sim_flat.argmax(dim=-1)  # [B,A]
        n_idx = best // L
        l_idx = best % L
        idx_out[..., 0] = n_idx
        idx_out[..., 1] = l_idx
    return idx_out


if __name__ == "__main__":
    from agent import GeometryEncoder, fulltrain_geo_enc2d
    import torch
    from contextlib import nullcontext

    
    cfg = TrainConfig()
    geometry_encoder = GeometryEncoder(M = cfg.num_sampled_pc, out_dim=cfg.num_att_heads * cfg.euc_head_dim)
    if cfg.train_geo_encoder:  
        geometry_encoder.impl = fulltrain_geo_enc2d(feat_dim=cfg.num_att_heads * cfg.euc_head_dim, num_sampled_pc= cfg.num_sampled_pc, save_path=f"geometry_encoder_2d")
    else:
        state = torch.load("geometry_encoder_2d_frozen.pth", map_location="cpu")
        geometry_encoder.impl.load_state_dict(state)
    os.makedirs(cfg.out_dir, exist_ok=True)
    

    # --- Data
    ds = PseudoDemoDataset(B=cfg.batch_size, T=cfg.pred_horizon, L = cfg.demo_length)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_items, num_workers=0)

    # --- Model
    policy = Policy(
        geometric_encoder=geometry_encoder,
        num_att_heads=cfg.num_att_heads,
        euc_head_dim=cfg.euc_head_dim,
        pred_horizon=cfg.pred_horizon,
        in_dim_agent=cfg.in_dim_agent,
        curvature=cfg.hyp_curvature,
        tau=cfg.tau

    ).to(cfg.device)  # your policy encapsulates rho, PCA alignment, and dynamics

    # --- Losses
    loss_align = AlignmentLoss(temperature=cfg.align_temp)
    loss_next  = NextStateHyperbolicLoss(curvature=cfg.hyp_curvature,
                                         margin=cfg.hyp_margin,
                                         neg_weight=cfg.hyp_neg_w)
    loss_traj  = ActionTrajectoryLoss()

    # --- Optim
    optim = AdamW([p for p in policy.parameters() if p.requires_grad],
                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cpu.amp.GradScaler(enabled=cfg.amp)

    step = 0
    policy.train()

    while step < cfg.max_steps:
        for item in dl:
            step += 1
            if step > cfg.max_steps:
                break

            # Move to device
            def dev(x): return None if x is None else x.to(cfg.device)
            curr_agent_info = dev(item.curr_agent_info)
            curr_object_pos = dev(item.curr_object_pos)
            clean_actions   = dev(item.clean_actions)
            demo_agent_info = dev(item.demo_agent_info)
            demo_object_pos = dev(item.demo_object_pos)
            demo_agent_action = dev(item.demo_agent_action)
            demo_time = dev(item.demo_time)
            curr_time = dev(item.curr_time)

            optim.zero_grad(set_to_none=True)
            use_amp = cfg.amp and torch.cuda.is_available()
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # ---- Forward: call YOUR policy.forward
                # Expect it to return:
                #   curr_rho: [B, A, de]
                #   demo_rho: [B, N, L, A, de]
                #   curr_hyp: [B, A, dh]   (estimated via PCA using demo alignments)
                #   demo_hyp: [B, N, L, A, dh]
                #   attn_e2e: [B, A, N, L] (optional: euc alignment scores)
                #   pred_actions: [B, T, da] (if dynamics head present)
                out = policy.forward(
                    curr_agent_info,
                    curr_object_pos,
                    demo_agent_info,
                    demo_object_pos,
                )

                curr_rho = out['curr_rho']          # [B, A, de]
                demo_rho = out['demo_rho']          # [B, N, L, A, de]
                curr_hyp = out['curr_hyp']          # [B, A, dh]
                demo_hyp = out['demo_hyp']          # [B, N, L, A, dh]
                attn_e2e = out.get('attn_e2e', None) # [B, A, N, L] (optional)
                pred_actions = out.get('pred_actions', None)
                # ---- Alignment loss (product-space InfoNCE) on the EUC channel
                L_align = loss_align(curr_rho, demo_rho, curr_time, demo_time, time_window=cfg.time_window)

                # ---- Pick time-consistent alignment indices for “current phase”
                if attn_e2e is None:
                    # basic similarity via cosine over rho if your policy doesn't output attn
                    B, A, de = curr_rho.shape
                    N, L = demo_rho.shape[1], demo_rho.shape[2]
                    # sims fallback path
                    qn = nn.functional.normalize(curr_rho, dim=-1).view(B, A, 1, 1, de)
                    kn = nn.functional.normalize(demo_rho, dim=-1)  # [B, N, L, A, de]
                    sims = (qn * kn).sum(-1).permute(0, 3, 1, 2).contiguous()  # [B, A, N, L]

                else:
                    sims = attn_e2e

                align_idx = select_time_consistent_indices(sims.detach(), demo_time, curr_time)  # [B, A, 2]

                # ---- Next-state hyperbolic loss (enforce forward-in-time “next search”)
                L_next = loss_next(curr_hyp, demo_hyp, align_idx, lookahead=cfg.lookahead)

                # ---- Action/trajectory supervision if you provide GT actions
                if pred_actions is not None and clean_actions is not None:
                    L_traj = loss_traj(pred_actions, clean_actions, cfg.max_translation)
                else:
                    L_traj = curr_rho.new_tensor(0.0)

                loss = L_align + L_next + L_traj

            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            if step % cfg.log_every == 0:
                print(f"[step {step:6d}] loss={loss.item():.4f} "
                      f"L_align={L_align.item():.4f} L_next={L_next.item():.4f} L_traj={L_traj.item():.4f}")

            if step % cfg.ckpt_every == 0:
                ckpt = {
                    "step": step,
                    "model": policy.state_dict(),
                    "optim": optim.state_dict(),
                    "cfg": cfg.__dict__,
                }
                path = os.path.join(cfg.out_dir, f"ckpt_{step:07d}.pt")
                torch.save(ckpt, path)
                print(f"Saved checkpoint to {path}")



