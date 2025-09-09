# Boilerplat codes by chatgpt
from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import Dataset, DataLoader

import numpy as np

from utilities import furthest_point_sampling_2d, knn_indices, fourier_embed_2d


class Geo2DEncoder(nn.Module):
    def __init__(
        self,
        M: int = 16,
        k: int = 64,
        fourier_L: int = 6,
        out_dim: int = 128,
    ):
        super().__init__()
        self.M = M
        self.k = k
        self.fourier_L = fourier_L

        # Per-neighbor descriptor size:
        in_ch = 2 + 1 + 1 + 1 + (4 * fourier_L)

        self.per_point = nn.Sequential(
            nn.Linear(in_ch, 64), nn.GELU(),
            nn.Linear(64, 128),   nn.GELU(),
        )

        # concat max-pooled [128] + mean-pooled [128] + local stats [4]
        self.out_head = nn.Sequential(
            nn.Linear(128 * 2 + 4, 128), nn.GELU(),
            nn.Linear(128, out_dim),
        )

    @torch.no_grad()
    def _centroids(self, P: torch.Tensor) -> torch.Tensor:
        return furthest_point_sampling_2d(P, self.M)

    def _encode_patch(self, P: torch.Tensor, c_xy: torch.Tensor) -> torch.Tensor:
        # kNN
        nidx = knn_indices(P, c_xy, self.k)
        patch = P[nidx]                    
        delta = patch - c_xy[None, :]      

        # polar bits
        r = torch.linalg.norm(delta, dim=-1, keepdim=True) 
        theta = torch.atan2(delta[:, 1:2], delta[:, 0:1])  
        st, ct = torch.sin(theta), torch.cos(theta)        

        # fourier on delta
        ff = fourier_embed_2d(delta, self.fourier_L)       

        per_pt = torch.cat([delta, r, st, ct, ff], dim=-1) 
        h = self.per_point(per_pt)                         
        h_max = h.max(dim=0).values
        h_mean = h.mean(dim=0)

        # simple stats for extra stability
        stats = torch.stack([
            r.mean().squeeze(),
            r.max().squeeze(),
            r.min().squeeze(),
            (r.std().squeeze() if r.numel() > 1 else torch.tensor(0.0, device=P.device)),
        ], dim=0)  

        feat = self.out_head(torch.cat([h_max, h_mean, stats], dim=0))  
        return feat

    def forward(self, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert P.ndim == 2 and P.shape[-1] == 2, "Geo2DEncoder expects P of shape [N,2]"
        N = P.shape[0]
        if N == 0:
            # Edge case: no points -> emit zeros
            device = P.device
            return torch.zeros(self.M, 128, device=device), torch.zeros(self.M, 2, device=device)

        c_idx = self._centroids(P) 
        C = P[c_idx]               

        feats = []
        for i in range(C.shape[0]):
            feats.append(self._encode_patch(P, C[i]))
        F_out = torch.stack(feats, dim=0)  
        return F_out, C


# Public wrapper
class GeometryEncoder(nn.Module):

    def __init__(
        self,
        M: int = 16,
        k: int = 64,
        fourier_L: int = 6,
        out_dim: int = 128
    ):
        super().__init__()
        self.impl = Geo2DEncoder(M=M, k=k, fourier_L=fourier_L, out_dim=out_dim)

    def forward(self, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.impl(P)


# Patch decoder for self-supervised pretraining 
class PatchDecoder(nn.Module):
    def __init__(self, in_dim=128, h=32, w=32):
        super().__init__()
        self.h, self.w = h, w
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, h * w)
        )
    def forward(self, F_nodes):                
        logits = self.net(F_nodes)             
        return logits.view(-1, self.h, self.w) 


# Config
M_NODES        = 8      # nodes per cloud 
K_NEIGHBORS    = 64      # per-patch kNN
FOURIER_L      = 6
FEAT_DIM       = 256
PATCH_H        = 32
PATCH_W        = 32
BATCH_CLOUDS   = 16      # clouds per batch
EPOCHS         = 20
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
JITTER_STD     = 0.01    # data aug: coord jitter (relative to unit square)
ROT_AUG_MAX    = math.radians(10)  # small random rotation
SAVE_PATH      = "geo2d_pretrained.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"


# Training data
class PointCloud2DDataset(Dataset):

    def __init__(self, clouds):
        self.clouds = []
        for P in clouds:
            P = np.asarray(P, dtype=np.float32)
            if P.ndim != 2 or P.shape[1] != 2:
                continue
            # Normalise each cloud to roughly unit box for stable training
            P = P - P.mean(0, keepdims=True)
            scale = (np.abs(P).max() + 1e-8)
            P = P / scale
            self.clouds.append(P)
        assert len(self.clouds) > 0, "No valid 2D clouds provided."

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, idx):
        P = self.clouds[idx].copy()
        # jitter
        P += np.random.normal(0.0, JITTER_STD, size=P.shape).astype(np.float32)
        # small random rotation
        ang = np.random.uniform(-ROT_AUG_MAX, ROT_AUG_MAX)
        c, s = math.cos(ang), math.sin(ang)
        R = np.array([[c, -s],[s, c]], dtype=np.float32)
        P = (R @ P.T).T
        return torch.from_numpy(P)  # [N,2]

def collate_clouds(batch):
    return batch



@torch.no_grad()
def _knn(P, q, k):
    N = P.shape[0]
    k = min(k, N)
    d2 = torch.sum((P - q)**2, dim=-1)
    return torch.topk(d2, k, largest=False).indices

@torch.no_grad()
def rasterize_patch(P, center, k=K_NEIGHBORS, H=PATCH_H, W=PATCH_W, pad=1.1):

    idx = _knn(P, center, k)               
    patch = P[idx]                         
    delta = patch - center[None, :]        

    # Scale to grid: fit to [-1,1] box with a small padding
    max_abs = torch.max(torch.abs(delta)) + 1e-6
    scale = pad * max_abs
    norm = delta / scale                   

    # Map to pixel coords
    xs = ((norm[:, 0] + 1) * 0.5) * (W - 1)
    ys = ((norm[:, 1] + 1) * 0.5) * (H - 1)
    xs = xs.round().long().clamp(0, W - 1)
    ys = ys.round().long().clamp(0, H - 1)

    grid = torch.zeros(H, W, device=P.device, dtype=torch.float32)
    grid[ys, xs] = 1.0
    return grid  # [H,W]


# Training step
def train_epoch(encoder, decoder, loader, optim):
    encoder.train(); decoder.train()
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss()

    for clouds in loader:
        # Build a flat batch of node features and their raster targets
        feats_all = []
        targs_all = []

        for P in clouds:
            P = P.to(device)
            # 1) Get M nodes (features + positions)
            F_nodes, C = encoder(P)   

            # 2) Build raster target per centroid (no grad)
            grids = []
            for i in range(C.shape[0]):
                g = rasterize_patch(P, C[i], k=encoder.k, H=PATCH_H, W=PATCH_W)
                grids.append(g)
            T = torch.stack(grids, dim=0)     

            feats_all.append(F_nodes)         
            targs_all.append(T)               

        F_batch = torch.cat(feats_all, dim=0) 
        Y_batch = torch.cat(targs_all, dim=0) 

        # 3) Predict grids
        logits = decoder(F_batch)             
        loss_rec = bce(logits, Y_batch)
        loss = loss_rec

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        optim.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


# data aux 
def get_pseudogame_clouds(num_clouds = 10):
    from data import PseudoGame as PseudoDemo
    clouds = []
    for _ in range(num_clouds):
        pseudo_demo = PseudoDemo()
        obs = pseudo_demo.get_obs()
        pc = obs['coords']
        clouds.append(pc)
    return clouds


# Training func 
def fulltrain_geo_enc2d(get_clouds_fn = get_pseudogame_clouds, num_samples = 1000, num_epochs = EPOCHS ,batch_size = BATCH_CLOUDS, num_sampled_pc = M_NODES, k_neighbours = K_NEIGHBORS, 
                        save_path = SAVE_PATH, fourier_L = FOURIER_L, feat_dim = FEAT_DIM, h = PATCH_H, w = PATCH_W, lr = LR, weight_decay = WEIGHT_DECAY):

    clouds = get_clouds_fn(num_samples)
    ds = PointCloud2DDataset(clouds)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_clouds, drop_last=True)

    encoder = Geo2DEncoder(M=num_sampled_pc, k=k_neighbours, fourier_L=fourier_L, out_dim=feat_dim).to(device)
    decoder = PatchDecoder(in_dim=feat_dim, h=h, w=w).to(device)

    optim = torch.optim.AdamW(
        [{"params": encoder.parameters(), "lr": lr},
         {"params": decoder.parameters(), "lr": lr}],
        lr=lr, weight_decay=weight_decay
    )

    best = float("inf")
    for ep in range(1, num_epochs + 1):
        loss = train_epoch(encoder, decoder, dl, optim)
        print(f"[{ep:02d}/{num_epochs}] train_loss={loss:.4f}")
        if loss < best:
            best = loss
            torch.save({"encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict()}, f'{save_path}.pth')

    print(f"Saved best to {save_path} (loss={best:.4f})")

    # Freeze encoder for policy training
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    torch.save(encoder.state_dict(), f"{save_path}_frozen.pth")
    print(f"Exported frozen encoder weights to {save_path}_frozen.pth")
    return encoder