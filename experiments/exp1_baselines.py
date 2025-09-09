#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp1_baselines.py

Add-on utilities:
- MLPProbe for next-token classification on tangent embeddings
- Class-weighted CE training loop for stability
- Last-token and bigram token-only baselines
- Per-base support/query split for fair retrieval evaluation
- Simple Levenshtein distance for token-string retrieval baseline
"""

import math, random, numpy as np, torch, torch.nn as nn
from typing import List, Tuple, Dict
from collections import defaultdict, Counter

from exp1_aux import poincare_log0, VOCAB

# ---------- MLP probe ----------

class MLPProbe(nn.Module):
    def __init__(self, d: int, K: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, K)
        )
    def forward(self, z): return self.net(z)

def class_weights_from_labels(y: torch.Tensor, K: int) -> torch.Tensor:
    cnt = Counter(y.tolist())
    w = torch.tensor([1.0 / max(1, cnt.get(i, 0)) for i in range(K)], dtype=torch.float32, device=y.device)
    w = (K * w / w.sum()).float()
    return w

def train_mlp_probe(Zp_ball_train: torch.Tensor, y_train: torch.Tensor,
                    Zp_ball_val: torch.Tensor, y_val: torch.Tensor,
                    K: int, curvature: float = 1.0,
                    hidden: int = 128, epochs: int = 60, lr: float = 1e-3, wd: float = 1e-3,
                    use_class_weights: bool = True) -> Tuple[MLPProbe, float]:
    # Map to tangent
    Ztr = poincare_log0(Zp_ball_train, c=curvature)
    Zva = poincare_log0(Zp_ball_val,   c=curvature)

    model = MLPProbe(Ztr.size(1), K, hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w = class_weights_from_labels(y_train, K) if use_class_weights else None
    crit = nn.CrossEntropyLoss(weight=w)

    best_acc, best_state = -1.0, None
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(Ztr), y_train); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Zva).argmax(-1) == y_val).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc

# ---------- Token-only baselines ----------

def last_token_baseline_train(tokens_train: List[List[str]]) -> Dict[str, int]:
    """Return mapping last_token (str) -> most frequent next_token_id (int) on train."""
    K = len(VOCAB)
    tok2id = {t:i for i,t in enumerate(VOCAB)}
    counts = defaultdict(Counter)
    for toks in tokens_train:
        for i in range(len(toks)-1):
            counts[toks[i]][tok2id[toks[i+1]]] += 1
        counts[toks[-1]][tok2id["<EOS>"]] += 1
    mapping = {}
    for last_tok, c in counts.items():
        mapping[last_tok] = c.most_common(1)[0][0]
    return mapping

def last_token_baseline_eval(mapping: Dict[str,int], tokens: List[List[str]]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return y_true, y_pred, and accuracy for a set of rollouts (prefixes concatenated)."""
    tok2id = {t:i for i,t in enumerate(VOCAB)}
    y_true, y_pred = [], []
    for toks in tokens:
        for i in range(len(toks)-1):
            y_true.append(tok2id[toks[i+1]])
            y_pred.append(mapping.get(toks[i], tok2id["<EOS>"]))
        y_true.append(tok2id["<EOS>"]); y_pred.append(mapping.get(toks[-1], tok2id["<EOS>"]))
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    acc = (y_true == y_pred).mean().item()
    return y_true, y_pred, acc

def bigram_table_train(tokens_train: List[List[str]]) -> Dict[Tuple[str,str], int]:
    """(prev2, prev1) -> next_id; falls back to last-token if unseen."""
    K = len(VOCAB)
    tok2id = {t:i for i,t in enumerate(VOCAB)}
    counts2 = defaultdict(Counter)
    counts1 = defaultdict(Counter)
    for toks in tokens_train:
        for i in range(len(toks)-2):
            counts2[(toks[i], toks[i+1])][tok2id[toks[i+2]]] += 1
        # tails
        if len(toks) >= 2:
            counts2[(toks[-2], toks[-1])][tok2id["<EOS>"]] += 1
        for i in range(len(toks)-1):
            counts1[toks[i]][tok2id[toks[i+1]]] += 1
        counts1[toks[-1]][tok2id["<EOS>"]] += 1
    # choose argmax
    table2 = {}
    for k, c in counts2.items():
        table2[k] = c.most_common(1)[0][0]
    # also return last-token fallback
    fallback = {}
    for k, c in counts1.items():
        fallback[k] = c.most_common(1)[0][0]
    table2["_fallback"] = fallback
    return table2

def bigram_table_eval(table2: Dict[Tuple[str,str], int], tokens: List[List[str]]) -> Tuple[np.ndarray, np.ndarray, float]:
    tok2id = {t:i for i,t in enumerate(VOCAB)}
    y_true, y_pred = [], []
    fallback = table2.get("_fallback", {})
    for toks in tokens:
        if len(toks) == 1:
            y_true.append(tok2id["<EOS>"]); y_pred.append(fallback.get(toks[0], tok2id["<EOS>"]))
            continue
        for i in range(len(toks)-2):
            y_true.append(tok2id[toks[i+2]])
            y_pred.append(table2.get((toks[i], toks[i+1]), fallback.get(toks[i+1], tok2id["<EOS>"])))
        # tail
        y_true.append(tok2id["<EOS>"])
        y_pred.append(table2.get((toks[-2], toks[-1]), fallback.get(toks[-1], tok2id["<EOS>"])))
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    acc = (y_true == y_pred).mean().item()
    return y_true, y_pred, acc

# ---------- Retrieval: per-base support/query + token-string baseline ----------

def per_base_support_query_indices(base_ids: List[str], support_frac: float = 0.5, seed: int = 0) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for i,b in enumerate(base_ids): buckets[b].append(i)
    sup_idx, qry_idx = [], []
    for b, idxs in buckets.items():
        rng.shuffle(idxs)
        k = max(1, int(round(len(idxs)*support_frac)))
        sup_idx += idxs[:k]
        qry_idx += idxs[k:]
    return sup_idx, qry_idx

def levenshtein(a: List[str], b: List[str]) -> int:
    # Simple DP edit distance (cost 1 per insert/delete/substitute)
    n, m = len(a), len(b)
    dp = [list(range(m+1))] + [[i+1]+[0]*m for i in range(n)]
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m]

def token_string_retrieval_acc(tokens: List[List[str]], base_ids: List[str], support_idx: List[int], query_idx: List[int]) -> float:
    # Build canonical string per base from support
    canon = {}
    for i in support_idx:
        canon[base_ids[i]] = tokens[i]  # bases are already unique strings; pick any
    # For each query, nearest base by edit distance (should be exact match)
    correct = 0
    for i in query_idx:
        t = tokens[i]
        # find nearest canon
        best, best_b = 1e9, None
        for b,s in canon.items():
            d = levenshtein(t, s)
            if d < best:
                best, best_b = d, b
        correct += int(best_b == base_ids[i])
    return correct / max(1, len(query_idx))
