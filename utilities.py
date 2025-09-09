import torch
from torch import Tensor
from typing import List, Tuple
import math
import numpy as np

######## SE2 Helpers ########
def _wrap_to_pi(theta: Tensor) -> Tensor:
    pi = math.pi
    return (theta + pi) % (2 * pi) - pi

def _taylor_A(w: Tensor) -> Tensor:   
    w2 = w * w
    return torch.where(w.abs() < 1e-4, 1 - w2/6 + w2*w2/120, torch.sin(w) / w)

def _taylor_B(w: Tensor) -> Tensor:  
    w2 = w * w
    return torch.where(w.abs() < 1e-4, 0.5 - w2/24 + w2*w2/720, (1 - torch.cos(w)) / w)

def se2_exp(y: Tensor) -> Tensor:
    vx, vy, w = y.unbind(dim=-1)
    A, B = _taylor_A(w), _taylor_B(w)
    tx = A * vx - B * vy
    ty = B * vx + A * vy
    theta = _wrap_to_pi(w)
    return torch.stack([tx, ty, theta], dim=-1)

def se2_log(a: Tensor) -> Tensor:
    dx, dy, w = a.unbind(dim=-1)
    A, B = _taylor_A(w), _taylor_B(w)
    denom = A*A + B*B
    invA, invB = A/denom, B/denom
    vx =  invA * dx + invB * dy
    vy = -invB * dx + invA * dy
    return torch.stack([vx, vy, _wrap_to_pi(w)], dim=-1)

######## action aux ########
def se2_from_kp(P, Q, eps=1e-6):
    B, K, _ = P.shape
    Pc = P.mean(dim=1, keepdim=True)
    Qc = Q.mean(dim=1, keepdim=True)
    P0, Q0 = P - Pc, Q - Qc
    H = torch.matmul(P0.transpose(1,2), Q0)    
    U, S, Vt = torch.linalg.svd(H)
    R = torch.matmul(Vt.transpose(1,2), U.transpose(1,2))
    # enforce det=+1
    det = torch.linalg.det(R).unsqueeze(-1).unsqueeze(-1)
    Vt_adj = torch.cat([Vt[:,:,:1], Vt[:,:,1:]*det], dim=2)
    R = torch.matmul(Vt_adj.transpose(1,2), U.transpose(1,2))
    t = (Qc - torch.matmul(Pc, R.transpose(1,2))).squeeze(1) 
    dtheta = torch.atan2(R[:,1,0], R[:,0,0])
    return t[:,0], t[:,1], dtheta

######### point care aux ########
def artanh(x, eps=1e-15):
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def poincare_exp0(v, c):
    norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)
    factor = torch.tanh(torch.sqrt(c) * norm) / (torch.sqrt(c) * norm)
    return factor * v

def poincare_log0(x, c):
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-9)
    factor = artanh(torch.sqrt(c) * norm) / (torch.sqrt(c) * norm)
    return factor * x

def mobius_add(x, y, c):
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c**2) * x2 * y2
    return num / den.clamp_min(1e-9)

def log_map_x(x, y, c):
    if type(c) != torch.Tensor:
        c = torch.tensor(c)
    w = mobius_add(-x, y, c)                              
    v0 = poincare_log0(w, c)                              
    lam = _lambda_x(x, c)                                 
    return (2.0 / lam) * v0

def exp_map_x(x, v, c):
    if type(c) != torch.Tensor:
        c = torch.tensor(c)
    lam = _lambda_x(x, c)
    step = (lam * 0.5) * v
    return mobius_add(x, poincare_exp0(step, c), c)

def poincare_dist(x, y, c, eps=1e-9):
    diff = mobius_add(-x, y, c)
    norm = torch.norm(diff, dim=-1).clamp_min(eps)
    return (2.0 / torch.sqrt(c)) * artanh(torch.sqrt(c) * norm)

def poincare_distance_sq(x, y, c):
    sqrt_c = c ** 0.5
    x = _project_to_ball(x, c)
    y = _project_to_ball(y, c)
    diff = mobius_add(-x, y, c)              
    norm = _safe_norm(diff, dim=-1)
    arg = torch.clamp(sqrt_c * norm, max=1 - 1e-7)
    dist = (2.0 / sqrt_c) * torch.atanh(arg)
    return dist * dist

def _lambda_x(x, c):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 - c * x2).clamp_min(1e-15)

def _safe_norm(x, dim=-1, keepdim=True, eps=1e-15):
    return torch.clamp(torch.norm(x, dim=dim, keepdim=keepdim), min=eps)

def _project_to_ball(x, c, eps=1e-5):
    sqrt_c = c ** 0.5
    norm = _safe_norm(x, dim=-1, keepdim=True)
    max_norm = (1. - eps) / sqrt_c
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale

######### Geometric Encoder Aux #########
def furthest_point_sampling_2d(P: torch.Tensor, M: int) -> torch.Tensor:
    device = P.device
    N = P.shape[0]
    M = min(M, N)
    idxs = torch.zeros(M, dtype=torch.long, device=device)

    # Pick a random start
    idxs[0] = torch.randint(0, N, (1,), device=device)
    dist = torch.full((N,), float("inf"), device=device)

    last = P[idxs[0]]
    for i in range(1, M):
        d = torch.sum((P - last) ** 2, dim=-1)
        dist = torch.minimum(dist, d)
        idxs[i] = torch.argmax(dist)
        last = P[idxs[i]]
    return idxs[:M]

def knn_indices(P: torch.Tensor, q: torch.Tensor, k: int) -> torch.Tensor:
    N = P.shape[0]
    k = min(k, N)
    d2 = torch.sum((P - q) ** 2, dim=-1)
    return torch.topk(d2, k, largest=False).indices

def fourier_embed_2d(delta: Tensor, num_freqs: int = 10) -> Tensor:
    delta = delta.to(torch.float32)
    freqs = (2.0 ** torch.arange(num_freqs, device=delta.device, dtype=delta.dtype)) * torch.pi
    ang = delta.unsqueeze(-1) * freqs  # [E,2,F]
    sin_x, cos_x = torch.sin(ang[:, 0, :]), torch.cos(ang[:, 0, :])
    sin_y, cos_y = torch.sin(ang[:, 1, :]), torch.cos(ang[:, 1, :])
    
    return torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)  


######### batching aux #########
def split_by_batch(x: Tensor, batch_vec: Tensor) -> List[Tensor]:
    B = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1
    outs: List[Tensor] = []
    for b in range(B):
        outs.append(x[batch_vec == b])
    return outs

def reshape_fixed_count(x: Tensor, batch_vec: Tensor, count_per_graph: int) -> Tensor:
    N, D = x.shape
    B = int(N // count_per_graph)
    counts = torch.bincount(batch_vec, minlength=B)
    if not torch.all(counts == count_per_graph):
        raise ValueError(f"Not all graphs have {count_per_graph} nodes: counts={counts.tolist()}")
    return x.view(B, count_per_graph, D)

def pad_by_batch(x: Tensor, batch_vec: Tensor) -> Tuple[Tensor, Tensor]:
    parts = split_by_batch(x, batch_vec)        
    B = len(parts)
    D = x.shape[-1]
    max_len = max(p.shape[0] for p in parts) if parts else 0
    padded = x.new_zeros(B, max_len, D)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=x.device)
    for b, p in enumerate(parts):
        n = p.shape[0]
        if n > 0:
            padded[b, :n] = p
            mask[b, :n] = True
    return padded, mask

def split_into_horizons(seq, pred_horizon: int):
    B, L = seq.shape[0], seq.shape[1]
    T = pred_horizon
    num = (L - 1) // T   # -1 so we always have next frames
    chunks = []
    for i in range(num):
        s = i*T
        e = s + T       
        chunks.append(seq[:, s:e, ...])
    return chunks