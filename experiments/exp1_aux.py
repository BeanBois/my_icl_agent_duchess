# From chatgpt 
from __future__ import annotations
import os, sys, math, random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

# Ensure local imports work (expects demo_handler.py in the same folder)
HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)

try:
    from utilities import logmap0 as poincare_log0, expmap0 as poincare_exp0
except Exception as e:
    raise RuntimeError("Could not import log0/exp0 from demo_handler.py. "
                       "Make sure demo_handler.py is in the same folder.") from e


# ==============================
# Tokenisation (y-down aware)
# ==============================

@dataclass
class TokeniserParams:
    move_thresh: float        # tau_move (pixels per step)
    rot_thresh_deg: float     # tau_theta (degrees per step)
    hysteresis_deg: float = 10.0
    n_dirs: int = 8
    prefer_translation: bool = True  # when both move/rot exceed thresholds, prefer direction


# Vocabulary and mapping
DIR_LABELS = ["E","NE","N","NW","W","SW","S","SE"]  # world-up compass labels (0° = E, +CCW)
ROT_PLUS, ROT_MINUS, STOP, EOS = "ROT+", "ROT-", "STOP", "<EOS>"
VOCAB = DIR_LABELS + [ROT_PLUS, ROT_MINUS, STOP, EOS]
VOCAB2ID = {tok:i for i,tok in enumerate(VOCAB)}

def vocab_size() -> int:
    return len(VOCAB)

def _angle_deg_world_up(dx: float, dy_world_up: float) -> float:
    """Angle in degrees for world-up convention (atan2 with y-up)."""
    return math.degrees(math.atan2(dy_world_up, dx))  # (-180,180]

def _quantise_dir(theta_deg: float, n_dirs: int) -> str:
    bin_width = 360.0 / n_dirs
    a = (theta_deg % 360.0 + 360.0) % 360.0  # [0,360)
    idx = int((a + bin_width/2.0) // bin_width) % n_dirs
    return DIR_LABELS[idx]

def _circ_dist_deg(a: float, b: float) -> float:
    """Smallest signed distance on the circle in degrees."""
    return abs(((a - b + 180.0) % 360.0) - 180.0)

def _rle_with_boundaries(tokens: List[str], steps: List[int]) -> Tuple[List[str], List[int]]:
    """
    Run-length encode while keeping the *last* step index for each run.
    tokens/steps are length L (one per time step). Returns:
      rle_tokens [P], end_steps [P] (indices into original timeline)
    """
    if not tokens:
        return [], []
    out_tok = [tokens[0]]
    out_end = [steps[0]]
    for t in range(1, len(tokens)):
        if tokens[t] != out_tok[-1]:
            out_tok.append(tokens[t])
            out_end.append(steps[t])
        else:
            out_end[-1] = steps[t]
    return out_tok, out_end

def auto_calibrate_thresholds_y_down(
    deltas_xy_y_down: np.ndarray, dtheta_rad: np.ndarray
) -> Tuple[float, float]:
    """
    Compute thresholds tau_move and tau_theta from y-down deltas.
    We use robust quantiles to separate jitter from true motion/rotation.
    """
    # Convert to world-up just for magnitude estimation
    dx = deltas_xy_y_down[:,0]
    dy_up = -deltas_xy_y_down[:,1]  # flip sign (image->world)
    r = np.hypot(dx, dy_up).astype(np.float64)  # move magnitudes
    a_deg = np.abs(np.degrees(dtheta_rad).astype(np.float64))

    r_sorted = np.sort(r)
    a_sorted = np.sort(a_deg)

    # Move threshold: 90th percentile within lower half
    half = r_sorted[:max(1, len(r_sorted)//2)]
    tau_move = float(np.quantile(half, 0.90)) if len(half) > 20 else float(np.median(r_sorted)*0.3)
    tau_move = max(2.0, tau_move)  # clamp to at least 2 pixels

    # Rotation threshold: consider steps with tiny translation
    tiny = r < max(1.0, 0.5 * tau_move)
    if tiny.sum() >= 30:
        tau_theta = float(max(5.0, np.quantile(a_deg[tiny], 0.80)))
    else:
        tau_theta = float(max(5.0, np.median(a_sorted)*0.5))

    return tau_move, tau_theta

def tokenise_actions_y_down(
    deltas_xy_y_down: np.ndarray, dtheta_rad: np.ndarray, params: TokeniserParams
) -> Tuple[List[str], List[int]]:
    """
    Tokenise per-step deltas measured in *image coords* (y-down). We adapt targets so:
      - "N" means screen-up (dy < 0)
      - "S" means screen-down (dy > 0)
    Returns:
      tokens_rle: List[str] of RLE tokens
      token_end_steps: List[int] indices where each token ends (inclusive)
    """
    assert deltas_xy_y_down.shape[0] == dtheta_rad.shape[0]
    L = deltas_xy_y_down.shape[0]
    step_ids = list(range(L))

    out_step_tokens: List[str] = []
    prev_dir_theta: Optional[float] = None  # last chosen direction center (deg)

    for t in range(L):
        dx = float(deltas_xy_y_down[t,0])
        dy_up = -float(deltas_xy_y_down[t,1])  # convert to world-up for angular binning

        r = math.hypot(dx, dy_up)
        rot_deg = abs(math.degrees(float(dtheta_rad[t])))
        has_move = r >= params.move_thresh
        has_rot  = rot_deg >= params.rot_thresh_deg

        if has_move and (params.prefer_translation or not has_rot):
            theta = _angle_deg_world_up(dx, dy_up)  # deg in world-up
            # Hysteresis: suppress small bin flips
            if prev_dir_theta is None or _circ_dist_deg(prev_dir_theta, theta) >= (180.0/params.n_dirs/1.0 + params.hysteresis_deg):
                label = _quantise_dir(theta, params.n_dirs)
                prev_dir_theta = theta
            else:
                label = _quantise_dir(prev_dir_theta, params.n_dirs)
            out_step_tokens.append(label)
        elif has_rot:
            label = ROT_PLUS if float(dtheta_rad[t]) > 0 else ROT_MINUS
            out_step_tokens.append(label)
            # Do not update prev_dir_theta on pure rotations
        else:
            out_step_tokens.append(STOP)

    tokens_rle, end_steps = _rle_with_boundaries(out_step_tokens, step_ids)
    return tokens_rle, end_steps


# ==============================
# DemoHandler adapter
# ==============================

class DemoEmbeddingsAdapter:
    """
    Wraps frozen DemoHandler to output time-series embeddings [L, dh],
    a single rollout embedding [dh], and prefix embeddings at token boundaries.
    Assumes handler.forward expects [B,N,L,A,6] with fields (x, y, theta, state, time, done)
    in **y-down** coordinates (image frame).
    """
    def __init__(self, demo_handler: torch.nn.Module, curvature: Optional[float] = None, device: str = "cpu"):
        self.h = demo_handler.to(device).eval()
        for p in self.h.parameters(): p.requires_grad = False
        # Try to read curvature from handler if present
        c = None
        for name in ["curvature", "c", "k"]:
            if hasattr(demo_handler, name):
                try:
                    c = float(getattr(demo_handler, name))
                except Exception:
                    pass
        self.c = float(curvature) if curvature is not None else (float(c) if c is not None else 1.0)
        self.device = device

    @torch.no_grad()
    def time_series(self, demo_agent_info_y_down: torch.Tensor) -> torch.Tensor:
        """
        demo_agent_info_y_down: [L, A, 6] OR [1,1,L,A,6], where coords are y-down (image frame).
        Returns: E[L, dh] (hyperbolic embeddings per time step, on the Poincaré ball)
        """
        x = demo_agent_info_y_down
        if x.dim() == 5:
            pass
        elif x.dim() == 2:
            # Not supported
            raise ValueError("demo_agent_info must be [L,A,6] or [1,1,L,A,6]")
        elif x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(0)  # [1,1,L,A,6]
        else:
            raise ValueError(f"Unexpected shape {tuple(x.shape)}")
        X = x.to(self.device)
        E = self.h(X)  # [1,1,L,dh]
        return E.squeeze(0).squeeze(0)  # [L, dh]

    @torch.no_grad()
    def rollout_embedding(self, demo_agent_info_y_down: torch.Tensor, mode: str = "tmean") -> torch.Tensor:
        E = self.time_series(demo_agent_info_y_down)  # [L, dh]
        if mode == "last":
            return E[-1]
        V = poincare_log0(E, c=self.c)          # [L, dh]
        vbar = V.mean(dim=0, keepdim=True)      # [1, dh]
        z = poincare_exp0(vbar, c=self.c).squeeze(0)  # [dh]
        return z

    @torch.no_grad()
    def prefix_embeddings_from_endsteps(
        self,
        demo_agent_info_y_down: torch.Tensor,
        token_end_steps: List[int],
        next_token_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        E = self.time_series(demo_agent_info_y_down)  # [L, dh]
        idx = torch.as_tensor(token_end_steps, dtype=torch.long, device=E.device)
        Zp = E.index_select(dim=0, index=idx)         # [P, dh] on Poincaré ball
        y  = torch.as_tensor(next_token_ids, dtype=torch.long, device=E.device)
        return Zp, y


# ==============================
# Retrieval utilities
# ==============================

def d_poincare(x: torch.Tensor, Y: torch.Tensor, curvature: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """One-vs-many Poincaré distance. x: [dh], Y: [R,dh]"""
    x = x.unsqueeze(0)
    diff2 = ((x - Y)**2).sum(-1).clamp_min(eps)
    x2 = (x**2).sum(-1).clamp_max((1.0/curvature)-eps)
    y2 = (Y**2).sum(-1).clamp_max((1.0/curvature)-eps)
    z = 1.0 + (2.0 * diff2) / ((1 - curvature*x2)*(1 - curvature*y2)).clamp_min(eps)
    return torch.acosh(z)

@torch.no_grad()
def frechet_mean_poincare(Z_ball: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Tangent mean at origin (log0 → mean → exp0)"""
    V = poincare_log0(Z_ball, c=curvature).mean(0, keepdim=True)
    return poincare_exp0(V, c=curvature).squeeze(0)

@torch.no_grad()
def build_references(z_rollout: torch.Tensor, base_ids: List[str], curvature: float = 1.0) -> Tuple[torch.Tensor, List[str]]:
    """
    Groups rollout embeddings by base sequence ID and returns Karcher means per ID.
    z_rollout: [N, dh] on Poincaré ball
    base_ids:  list[str], length N
    """
    from collections import defaultdict
    buckets: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for z, bid in zip(z_rollout, base_ids):
        buckets[bid].append(z)
    refs, ref_ids = [], []
    for bid, zs in buckets.items():
        Z = torch.stack(zs, dim=0)
        mu = frechet_mean_poincare(Z, curvature=curvature)
        refs.append(mu); ref_ids.append(bid)
    return torch.stack(refs, dim=0), ref_ids

@torch.no_grad()
def topk_retrieval_acc(
    z_query: torch.Tensor, query_base_ids: List[str],
    ref_Z: torch.Tensor, ref_ids: List[str],
    k: int = 1, curvature: float = 1.0
) -> float:
    correct = 0
    for z, true_id in zip(z_query, query_base_ids):
        d = d_poincare(z, ref_Z, curvature=curvature)  # [R]
        topk = d.topk(k, largest=False).indices.tolist()
        cand = [ref_ids[i] for i in topk]
        correct += int(true_id in cand)
    return correct / len(query_base_ids)


# ==============================
# Next-token linear probe
# ==============================

class LinearProbe(nn.Module):
    def __init__(self, d: int, K: int):
        super().__init__()
        self.fc = nn.Linear(d, K)
    def forward(self, z): return self.fc(z)

@torch.no_grad()
def to_tangent(Z_ball: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    return poincare_log0(Z_ball, c=curvature)

def train_linear_probe(
    Zp_ball_train: torch.Tensor, y_train: torch.Tensor,
    Zp_ball_val: torch.Tensor,   y_val: torch.Tensor,
    K: int, curvature: float = 1.0, epochs: int = 60, lr: float = 1e-3, wd: float = 1e-3
) -> Tuple[LinearProbe, float]:
    Ztr = to_tangent(Zp_ball_train, curvature=curvature)
    Zva = to_tangent(Zp_ball_val,   curvature=curvature)
    model = LinearProbe(Ztr.size(1), K)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    best_acc, best_w = -1.0, None
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(Ztr), y_train); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Zva).argmax(-1) == y_val).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_w = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for p, w in zip(model.parameters(), best_w): p.copy_(w)
    return model, best_acc


# ==============================
# Dataset builder (from raw arrays)
# ==============================

def build_demo_agent_info_y_down(
    x_img: np.ndarray, y_img: np.ndarray, theta_rad: np.ndarray,
    state: Optional[np.ndarray] = None, done: Optional[np.ndarray] = None,
    time_norm: str = "index", A: int = 1
) -> torch.Tensor:
    """
    Assemble [1,1,L,A,6] tensor for DemoHandler forward. Image (y-down) coords expected.
    theta_rad: heading, +CCW in radians. 
    state: optional int/float flags; defaults to zeros.
    done:  optional 0/1 flags; defaults to zeros with last=1.
    time_norm: 'index' (0..L-1), or 'lin' (0..1)
    """
    L = int(len(x_img))
    assert len(y_img) == L and len(theta_rad) == L
    if state is None: state = np.zeros(L, dtype=np.float32)
    if done is None:
        done = np.zeros(L, dtype=np.float32); done[-1] = 1.0
    if time_norm == "index":
        t = np.arange(L, dtype=np.float32)
    else:
        t = np.linspace(0.0, 1.0, L, dtype=np.float32)

    # Build [L, A, 6] for A agents; we populate agent 0 as per user's handler
    arr = np.zeros((L, A, 6), dtype=np.float32)
    arr[:, 0, 0] = x_img               # field 0 = x (y-down frame)
    arr[:, 0, 1] = y_img               # field 1 = y (y-down frame)
    arr[:, 0, 2] = theta_rad           # field 2 = theta in user's code snippet? (They used [...,0,2] for theta)
    arr[:, 0, 3] = state               # field 3 = state
    arr[:, 0, 4] = t                   # field 4 = time
    arr[:, 0, 5] = done                # field 5 = done


    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,L,A,6]

def tokens_to_base_id(tokens: List[str]) -> str:
    return "|".join(tokens)

def build_prefix_labels(tokens_rle: List[str]) -> List[int]:
    """
    For each token position p (end of token p), the target is token p+1, else EOS for the last.
    Returns a list of class ids (length == len(tokens_rle)).
    """
    ids = [VOCAB2ID.get(tok, VOCAB2ID[STOP]) for tok in tokens_rle[1:]] + [VOCAB2ID[EOS]]
    return ids


# ==============================
# High-level runner
# ==============================

class SequenceProbes:
    def __init__(self, demo_handler: torch.nn.Module, curvature: Optional[float] = None, device: str = "cpu"):
        self.adapter = DemoEmbeddingsAdapter(demo_handler, curvature=curvature, device=device)
        self.c = self.adapter.c
        self.device = device

    def autocalibrate_from_rollouts(self, rollouts: List[Dict]) -> TokeniserParams:
        """
        rollouts: list of dicts with keys: x_img, y_img, theta_rad (each np.ndarray [L])
        Returns TokeniserParams calibrated over pooled deltas.
        """
        all_dxdy = []
        all_dth  = []
        for R in rollouts:
            x = np.asarray(R["x_img"], dtype=np.float32)
            y = np.asarray(R["y_img"], dtype=np.float32)
            th = np.asarray(R["theta_rad"], dtype=np.float32)
            dx = np.diff(x, prepend=x[:1])
            dy = np.diff(y, prepend=y[:1])
            dth = np.diff(th, prepend=th[:1])
            # Wrap angles to (-pi, pi]
            dth = (dth + np.pi) % (2*np.pi) - np.pi
            all_dxdy.append(np.stack([dx, dy], axis=1))
            all_dth.append(dth)
        deltas_xy = np.concatenate(all_dxdy, axis=0)
        dtheta    = np.concatenate(all_dth, axis=0)
        tau_move, tau_theta = auto_calibrate_thresholds_y_down(deltas_xy, dtheta)
        # return TokeniserParams(move_thresh=tau_move, rot_thresh_deg=tau_theta, hysteresis_deg=10.0)
        return TokeniserParams(move_thresh=20, rot_thresh_deg=20, hysteresis_deg=10)

    def build_dataset(
        self, rollouts: List[Dict], params: TokeniserParams, prefer_last_step: bool = False
    ) -> Dict[str, List]:
        """
        Given raw rollouts (arrays), compute tokens, end-steps, embeddings, and labels.
        Returns a dict of lists (parallel, same length per key).
        Keys:
          - base_id     : str (joined tokens)
          - tokens      : List[str]
          - token_end   : List[int]
          - z_rollout   : torch.Tensor [dh] on ball
          - Zp          : torch.Tensor [P, dh] on ball (prefix embeddings at end-steps)
          - y_next      : torch.Tensor [P] (class ids for next token, EOS at end)
        """
        data: Dict[str, List] = {"base_id":[], "tokens":[], "token_end":[],
                                 "z_rollout":[], "Zp":[], "y_next":[]}
        for R in rollouts:
            x = np.asarray(R["x_img"], dtype=np.float32)
            y = np.asarray(R["y_img"], dtype=np.float32)
            th = np.asarray(R["theta_rad"], dtype=np.float32)
            state = np.asarray(R.get("state", np.zeros_like(x)), dtype=np.float32)
            done  = np.asarray(R.get("done",  np.zeros_like(x)), dtype=np.float32)
            L = len(x)
            dx = np.diff(x, prepend=x[:1]); dy = np.diff(y, prepend=y[:1])
            dth = np.diff(th, prepend=th[:1]); dth = (dth + np.pi) % (2*np.pi) - np.pi

            # Tokens from y-down:
            tokens_rle, end_steps = tokenise_actions_y_down(np.stack([dx,dy],axis=1), dth, params)
            base = tokens_to_base_id(tokens_rle)

            # Next-token labels
            y_next_ids = build_prefix_labels(tokens_rle)

            # Build DemoHandler input (y-down) and get embeddings
            demo_info = build_demo_agent_info_y_down(x, y, th, state=state, done=done)  # [1,1,L,1,6]
            z_roll = self.adapter.rollout_embedding(demo_info, mode=("last" if prefer_last_step else "tmean"))  # [dh]
            Zp, y_next = self.adapter.prefix_embeddings_from_endsteps(demo_info, end_steps, y_next_ids)         # [P,dh], [P]

            # Append
            data["base_id"].append(base)
            data["tokens"].append(tokens_rle)
            data["token_end"].append(end_steps)
            data["z_rollout"].append(z_roll.detach().cpu())
            data["Zp"].append(Zp.detach().cpu())
            data["y_next"].append(y_next.detach().cpu())

        # Stack tensors where possible
        data["z_rollout"] = torch.stack(data["z_rollout"], dim=0)  # [N, dh]
        # For prefixes, concat variable P across rollouts
        data["Zp"] = torch.cat(data["Zp"], dim=0)                  # [∑P, dh]
        data["y_next"] = torch.cat(data["y_next"], dim=0).long()    # [∑P]
        return data

    def vocab_size(self) -> int:
        return vocab_size()


# ==============================
# Splits by base sequence ID
# ==============================

def split_by_base_id(dataset: Dict[str, List], train_frac=0.7, val_frac=0.15, seed=0) -> Dict[str, Dict[str, object]]:
    """
    Deterministic split by unique base sequence strings so no leakage between splits.
    Returns dict with keys 'train','val','test' each mapping to shallow copies of dataset with subsetted items.
    """
    rng = random.Random(seed)
    base_ids = list(sorted(set(dataset["base_id"])))
    rng.shuffle(base_ids)
    n = len(base_ids)
    n_train = int(round(train_frac * n))
    n_val   = int(round(val_frac * n))
    train_ids = set(base_ids[:n_train])
    val_ids   = set(base_ids[n_train:n_train+n_val])
    test_ids  = set(base_ids[n_train+n_val:])

    def mask_from_ids(ids: set) -> List[int]:
        return [i for i,b in enumerate(dataset["base_id"]) if b in ids]

    def subset(idx: List[int]) -> Dict[str, object]:
        out = {}
        for k,v in dataset.items():
            if k in ("z_rollout","Zp","y_next"):
                # For tensors that are concatenated, we need to filter per-rollout for z_rollout; for Zp/y_next it's trickier (variable P).
                # We'll rebuild Zp/y_next by concatenating only the selected rollouts (we kept per-rollout lists 'tokens' and 'token_end' but not per-rollout Zp,y).
                # Simpler approach: store per-rollout prefix tensors initially (not concatenated). Here we approximate by returning the original full tensors.
                out[k] = v
            else:
                out[k] = [v[i] for i in idx]
        # For Zp/y_next, rebuild from the original per-rollout structures if available

        return out

    # To keep it simple, we will also create index arrays and rebuild Zp/y_next properly:
    # Build cumulative prefix counts per rollout to slice Zp/y_next
    P_counts = [len(te) for te in dataset["token_end"]]
    cumP = np.cumsum([0] + P_counts)
    def slice_prefix(idx: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        pieces_Z, pieces_y = [], []
        for i in idx:
            s, e = int(cumP[i]), int(cumP[i+1])
            pieces_Z.append(dataset["Zp"][s:e])
            pieces_y.append(dataset["y_next"][s:e])
        return torch.cat(pieces_Z, dim=0) if pieces_Z else torch.empty(0, dataset["Zp"].size(1)), \
               torch.cat(pieces_y, dim=0) if pieces_y else torch.empty(0, dtype=torch.long)

    idx_tr = mask_from_ids(train_ids)
    idx_va = mask_from_ids(val_ids)
    idx_te = mask_from_ids(test_ids)

    splits: Dict[str, Dict[str, object]] = {}
    for name, idx in [("train", idx_tr), ("val", idx_va), ("test", idx_te)]:
        part = {
            "base_id": [dataset["base_id"][i] for i in idx],
            "tokens":  [dataset["tokens"][i]   for i in idx],
            "token_end":[dataset["token_end"][i] for i in idx],
            "z_rollout": torch.stack([dataset["z_rollout"][i] for i in idx], dim=0) if idx else torch.empty(0, dataset["z_rollout"].size(1)),
        }
        Zp, y = slice_prefix(idx)
        part["Zp"] = Zp
        part["y_next"] = y
        splits[name] = part

    return splits


# Data generator aux 
from data import PseudoDemoGenerator
from torch.utils.data import Dataset




@dataclass
class Experiment1Item:
    demo_agent_info: torch.Tensor       # [B, N, L, A, 6] (x,y,theta,state,time,done)


 

class Experiment1Dataset(Dataset):
    agent_kp = PseudoDemoGenerator.agent_keypoints
    kp_order = ["front", "back-left", "back-right", "center"]

    def __init__(self,  biased_odds, augmented_odds, length=10000, device="cpu",
                 B=4, A=4, M=64, N=2, L=10, T=8, action_dim=3):
        self.biased_odds = biased_odds
        self.augmented_odds = augmented_odds
        self.length = length
        self.device = device
        self.B, self.A, self.M, self.N, self.L, self.T = B, A, M, N, L, T
        self.action_dim = action_dim
        self.data_gen = PseudoDemoGenerator(device, num_demos = self.N + 1, demo_length = self.L, pred_horizon = self.T,
                                            biased_odds=biased_odds, augmented_odds=augmented_odds)

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> Experiment1Item:
        B, A, M, N, L, T, ad = self.B, self.A, self.M, self.N, self.L, self.T, self.action_dim
        demo_agent_info   = torch.randn(B, N, L, A, 6)
        _, context, _ = self.data_gen.get_batch_samples(self.B)
        demo_agent_info, _, _ = self._process_context(context)
    
        return Experiment1Item(
            demo_agent_info = demo_agent_info
        )

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

        kp_local = [PseudoDemoGenerator.agent_keypoints[k] for k in Experiment1Dataset.kp_order]
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
            # theta = torch.atan2(M[1, 0], M[0, 0])
            theta = torch.atan2(M[0, 1], M[0, 0])          # CW 
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

def collate_items(batch: List[Experiment1Item]) -> Experiment1Item:
    # Here each dataset __getitem__ already returns batched B samples;
    # if yours returns single samples, stack along dim 0 here.
    assert len(batch) == 1, "This stub returns batch-already tensors; adjust as needed."
    return batch[0]


def retrieval_curve(z_test, id_test, ref_Z, ref_ids, ks, curvature, dist_fn=None):
    """
    Returns list of (k, acc_at_k).
    """
    accs = []
    for k in ks:
        acc = topk_retrieval_acc(z_test, id_test, ref_Z, ref_ids, k=k, curvature=curvature)
        accs.append((k, float(acc)))
    return accs
def confusion_matrix_from_predictions(y_true, y_pred, K):
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < K and 0 <= p < K:
            cm[t, p] += 1
    return cm