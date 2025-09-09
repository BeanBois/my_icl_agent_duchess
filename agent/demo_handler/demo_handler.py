import math
from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import poincare_log0, poincare_exp0, exp_map_x, log_map_x

# ---------- tree utilities ----------
# clustering is done based on 2 conditions
    # if next proceeding agent_info state has changed, then the next_proceeding obs becomes new parent
    # else if abs(angle_parent - angle_next) > angular_granulity, then it becomes new parent
# an example tree is as such : 
#                   empty root 
#                  /         \
#                 1           7
#                / \         / \
#               2   4        8  10
#              /.  /  \.     |.  | \
#             3.  5.   6.    9.  11 12

def _cluster(
        cluster_idxs,
        theta,
        state,
        gran
):
    earliest_index = cluster_idxs.pop(0)
    clusters = []
    children = []
    for idx in cluster_idxs:
        if (state[earliest_index] != state[idx] and False) or \
         abs(math.radians(theta[earliest_index] - theta[idx])) > gran:
            clusters.append((earliest_index, children))
            earliest_index = idx
            children = []
            continue
        children.append(idx)
    clusters.append((earliest_index, children))
    return clusters

def build_temporal_tree_multigran_K(
    theta: torch.Tensor,            # [T]
    state: torch.Tensor,            # [T]
    grans: List[float],
    use_degrees: bool = True,
) -> Tuple[List[int], List[List[int]]]:
    assert theta.ndim == 1 and state.ndim == 1 and theta.shape[0] == state.shape[0]
    T = theta.shape[0]
    N = T + 1  # include root at index 0; items are 1..T

    children_of = [-1 for _ in range(T)]
    queue_clusters = [(-1 , [i for i in range(T)])]

    for gran in grans:
        new_cluster = []
        while len(queue_clusters) > 0:
            parent, cluster = queue_clusters.pop(0)
            if len(cluster) <= 0:
                continue
            _new_cluster = _cluster(cluster, theta, state, gran)
            new_cluster = new_cluster + _new_cluster
        for cluster in new_cluster:
            parent, children = cluster
            for child in children:
                children_of[child] = parent
        queue_clusters = new_cluster

    parent = [-1] * T
    children = [[] for _ in range(T)]
    ROOT = 0
    parent[ROOT] = -1
    for c, p in enumerate(children_of):
        if p != -1:
            children[p].append(c)
        parent[c] = p
    if T == 0:
        return parent, children
    return parent, children

# ---------- SK-style hyperbolic constructor in 2D ----------
class SKConstructor2D(nn.Module):
    def __init__(self, curvature: float = 1.0, base_cone_deg: float = 30.0, min_edge_dist: float = 0.5):
        super().__init__()
        self.register_buffer("c", torch.tensor(float(curvature)))
        self.base_cone = math.radians(base_cone_deg)
        self.min_edge_dist = min_edge_dist  # hyperbolic distance floor

    def forward(self, children: List[List[int]], parent: List[int], L: int) -> torch.Tensor:
        device = self.c.device
        emb = torch.zeros(L, 2, device=device)
        node_angle = torch.zeros(L, device=device)
        node_radius = torch.zeros(L, device=device)
        for p in range(L):
            ch = children[p]
            if len(ch) == 0:
                continue
            total_span = max(self.base_cone * len(ch), self.base_cone)
            start = node_angle[p] - total_span / 2.0
            for i, kid in enumerate(ch):
                node_angle[kid] = start + (i + 0.5) * (total_span / len(ch))
                node_radius[kid] = node_radius[p] + self.min_edge_dist
        e_x = torch.stack([torch.cos(node_angle), torch.sin(node_angle)], dim=-1)
        v = e_x * node_radius.unsqueeze(-1)
        emb = poincare_exp0(v, self.c)
        return emb  # [L,2]

# ---------- Inter-/Intra-level hyperbolic mixer ----------
class HyperbolicTemporalMixer(nn.Module):
    def __init__(self, curvature: float = 1.0):
        super().__init__()
        self.register_buffer("c", torch.tensor(float(curvature)))
        self.max_depth = 512
        self.depth_scale = nn.Embedding(self.max_depth, 1)
        nn.init.ones_(self.depth_scale.weight)
        self.depth_theta = nn.Embedding(self.max_depth, 1)
        nn.init.zeros_(self.depth_theta.weight)

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        d = depth.clamp(max=self.max_depth - 1)
        v = poincare_log0(x, self.c)  # [L,2]
        k = self.depth_scale(d).view(-1, 1)
        v = k * v
        ang = self.depth_theta(d).view(-1)
        cos_a, sin_a = torch.cos(ang), torch.sin(ang)
        R = torch.stack([torch.stack([cos_a, -sin_a], dim=-1),
                         torch.stack([sin_a,  cos_a], dim=-1)], dim=-2)  # [L,2,2]
        v = torch.einsum('lij,lj->li', R, v)
        return poincare_exp0(v, self.c)

# ---------- segment-softmax (no external deps) ----------
def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: Optional[int] = None) -> torch.Tensor:
    """
    Softmax over variable-length segments given by `index` (e.g., per target node i in edge list).
    scores: [E], index: [E] with values in [0..V-1]
    """
    if num_segments is None:
        num_segments = int(index.max().item()) + 1 if index.numel() > 0 else 0
    # subtract per-segment max for stability
    max_buf = torch.full((num_segments,), -1e30, device=scores.device, dtype=scores.dtype)
    max_buf = max_buf.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
    normed = scores - max_buf.gather(0, index)
    exp = torch.exp(normed)
    denom = torch.zeros(num_segments, device=scores.device, dtype=scores.dtype)
    denom = denom.scatter_reduce(0, index, exp, reduce='sum')
    return exp / denom.gather(0, index).clamp_min(1e-15)

# ---------- Option A: Tangent-at-node hyperbolic message passing ----------
class HyperbolicTreeLayer(nn.Module):
    """
    For each edge i <- j:
      v_ij = log_xi(x_j) in T_{x_i}; score -> α_ij (softmax per i); aggregate m_i = Σ α_ij v_ij;
      optional depth-gate (your mixer params) in T_{x_i}; update x_i' = exp_xi(η m_i).
    """
    def __init__(self, curvature: float = 1.0, att_dim: int = 16, use_bias: bool = True):
        super().__init__()
        self.register_buffer("c", torch.tensor(float(curvature)))
        self.scorer = nn.Sequential(
            nn.Linear(2, att_dim, bias=use_bias), nn.GELU(), nn.Linear(att_dim, 1, bias=False)
        )
        self.eta = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,            # [L,2]
        edge_index: torch.Tensor,   # [2,E], row=i (target), col=j (source)
        depth: torch.Tensor,        # [L]
        mixer: Optional[HyperbolicTemporalMixer] = None,
    ) -> torch.Tensor:
        assert x.dim() == 2 and x.size(-1) == 2
        assert edge_index.dim() == 2 and edge_index.size(0) == 2
        L = x.size(0)
        row, col = edge_index[0], edge_index[1]    # i <- j
        xi = x[row]
        xj = x[col]
        c = self.c

        # 1) log at xi (consistent with your exp0/log0)
        v_ij = log_map_x(xi, xj, c)               # [E,2]

        # 2) attention in T_{xi}
        scores = self.scorer(v_ij).squeeze(-1)    # [E]
        alpha = segment_softmax(scores, row, num_segments=L)  # [E]
        v_ij = alpha.unsqueeze(-1) * v_ij

        # 3) aggregate per target i
        m_i = torch.zeros(L, 2, device=x.device, dtype=x.dtype)
        m_i.index_add_(0, row, v_ij)

        # 4) optional depth-gate (reuse your mixer params in tangent)
        if mixer is not None:
            d = depth.clamp(max=mixer.max_depth - 1)
            k = mixer.depth_scale(d).view(-1, 1)
            ang = mixer.depth_theta(d).view(-1)
            ca, sa = torch.cos(ang), torch.sin(ang)
            R = torch.stack([torch.stack([ca, -sa], dim=-1),
                             torch.stack([sa,  ca], dim=-1)], dim=-2)  # [L,2,2]
            m_i = k * torch.einsum('lij,lj->li', R, m_i)

        # 5) manifold update at xi
        x_new = exp_map_x(x, self.eta * m_i, c)
        return x_new

# ---------- edge builder for local neighbourhood ----------
def build_local_edge_index(parent: List[int], children: List[List[int]], L: int) -> torch.Tensor:
    """
    Returns [2,E] with ONLY:
      - temporal edges (t-1 <-> t),
      - parent <-> child edges.
    """
    edges = []
    # temporal chain
    for t in range(L - 1):
        edges.append((t, t + 1)); edges.append((t + 1, t))
    # tree edges
    for p in range(L):
        for ch in children[p]:
            edges.append((p, ch)); edges.append((ch, p))
    if len(edges) == 0:
        return torch.empty(2, 0, dtype=torch.long)
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return ei

# ---------- Full demo handler ----------
class DemoHandler(nn.Module):
    """
    Input: demo_agent_info [B, N, L, A, 6]  (x,y,theta,state,time,done)
    Output: embeddings [B, N, L, 2]
    Pipeline:
      1) Build temporal tree via multi-granularity clustering.
      2) SK-style initial embedding in the Poincaré ball (2D).
      3) Option-A tree message passing (tangent at node) using ONLY local edges.
      4) Depth-gated mixer (your original) for per-level rotate+scale.
    """
    def __init__(self,
                 curvature: float = 1.0,
                 angular_granularities_deg: List[float] = [90.0, 60.0, 30.0],
                 base_cone_deg: float = 30.0,
                 min_edge_dist: float = 0.5,
                 att_dim: int = 16):
        super().__init__()
        self.c = torch.tensor(float(curvature))  # as Tensor for sqrt
        self.ang_grans = [math.radians(g) for g in angular_granularities_deg]
        self.sk = SKConstructor2D(curvature=float(curvature),
                                  base_cone_deg=base_cone_deg,
                                  min_edge_dist=min_edge_dist)
        self.mixer = HyperbolicTemporalMixer(curvature=float(curvature))
        self.tree_layer = HyperbolicTreeLayer(curvature=float(curvature), att_dim=att_dim)

    @staticmethod
    def _compute_depth(parent: List[int]) -> torch.Tensor:
        L = len(parent)
        depth = torch.zeros(L, dtype=torch.long)
        for i in range(L):
            d, p = 0, parent[i]
            while p != -1 and p is not None:
                d += 1
                p = parent[p]
            depth[i] = d
        return depth

    def forward(self, demo_agent_info: torch.Tensor) -> torch.Tensor:
        """
        demo_agent_info: [B, N, L, A, 6]  -> returns [B, N, L, 2]
        Uses agent index 0 as the proceeding agent for θ/state.
        """
        B, N, L, A, D = demo_agent_info.shape
        device = demo_agent_info.device
        assert D >= 6, "last dim must contain at least (x,y,theta,state,time,done)"

        theta = demo_agent_info[..., 0, 2]  # [B,N,L]
        state = demo_agent_info[..., 0, 3]  # [B,N,L]

        out = torch.zeros(B, N, L, 2, device=device, dtype=demo_agent_info.dtype)

        for b in range(B):
            for n in range(N):
                th_seq = theta[b, n]  # [L]
                st_seq = state[b, n]  # [L]

                # 1) temporal tree
                parent, children = build_temporal_tree_multigran_K(th_seq, st_seq, self.ang_grans)
                depth = self._compute_depth(parent).to(device)  # [L]

                # 2) SK initial embedding
                emb = self.sk(children, parent, L).to(device)   # [L,2]

                # 3) local edges (temporal ±1 and parent↔child)
                edge_index = build_local_edge_index(parent, children, L).to(device)

                # 4) Option-A tangent-at-node message passing (with depth gating inside)
                emb = self.tree_layer(emb, edge_index, depth, mixer=self.mixer)

                # 5) (optional) extra global gate — if you still want it, uncomment:
                # emb = self.mixer(emb, depth)

                out[b, n] = emb

        return out  # [B,N,L,2]
