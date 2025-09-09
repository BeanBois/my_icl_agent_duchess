import numpy as np
import torch
import os

from .plot_aux import (
    plot_retrieval_curve,
    plot_confusion_matrix
)
from .exp1_aux import poincare_log0, VOCAB
from .exp1_baselines import train_mlp_probe
from .exp1_baselines import last_token_baseline_train, last_token_baseline_eval, bigram_table_train, bigram_table_eval
from .exp1_baselines import per_base_support_query_indices
from .exp1_aux import build_references, topk_retrieval_acc
from .exp1_baselines import token_string_retrieval_acc

from .exp1_aux import (
    SequenceProbes,
    split_by_base_id,
    build_references,
    topk_retrieval_acc,
    train_linear_probe,
    vocab_size,
    retrieval_curve,
    confusion_matrix_from_predictions,
    VOCAB,
)

plot = True

def _make_rollouts_from_xytheta(xs, ys, thetas_rad, states=None, dones=None):
    """Pack lists of arrays into the rollouts format SequenceProbes expects."""
    assert len(xs) == len(ys) == len(thetas_rad)
    rollouts = []
    for i in range(len(xs)):
        x = np.asarray(xs[i], dtype=np.float32)
        y = np.asarray(ys[i], dtype=np.float32)        # y-down image frame
        th = np.asarray(thetas_rad[i], dtype=np.float32)
        L = len(x)
        st = np.zeros(L, dtype=np.float32) if states is None or (states[i] is None) else np.asarray(states[i], np.float32)
        dn = np.zeros(L, dtype=np.float32) if dones  is None or (dones[i]  is None) else np.asarray(dones[i],  np.float32)
        rollouts.append({"x_img": x, "y_img": y, "theta_rad": th, "state": st, "done": dn})
    return rollouts

def _integrate_actions_to_xytheta(actions, theta0=0.0):
    """
    actions: [T, >=3] in IMAGE frame (dx, dy, dtheta) with y-down convention.
    → returns x, y, theta arrays of length T.
    """
    a = np.asarray(actions, dtype=np.float32)
    assert a.ndim == 2 and a.shape[1] >= 3
    dx, dy, dth = a[:, 0], a[:, 1], a[:, 2]
    T = a.shape[0]
    x = np.zeros(T, dtype=np.float32)
    y = np.zeros(T, dtype=np.float32)
    th = np.zeros(T, dtype=np.float32); th[0] = float(theta0)
    for t in range(1, T):
        th[t] = th[t-1] + dth[t]
        x[t]  = x[t-1] + dx[t]
        y[t]  = y[t-1] + dy[t]
    return x, y, th



# def _adapt_exp1_item_to_rollout(sample):
#     """
#     Flexible adapter for Experiment1Dataset item → rollout dict.
#     Supported keys (any one path is enough):
#       1) 'x_img','y_img','theta_rad'          (preferred; y-down)
#       2) 'pos' [T,2] + 'theta' [T]
#       3) 'actions' [T,3+] (dx,dy,dtheta in y-down)  → integrate
#     Optional: 'state', 'done'
#     """
#     # Path 1: already in image frame
#     if all(k in sample for k in ("x_img", "y_img", "theta_rad")):
#         return {
#             "x_img": np.asarray(sample["x_img"], np.float32),
#             "y_img": np.asarray(sample["y_img"], np.float32),
#             "theta_rad": np.asarray(sample["theta_rad"], np.float32),
#             "state": np.asarray(sample.get("state", None), np.float32) if "state" in sample else None,
#             "done":  np.asarray(sample.get("done",  None), np.float32) if "done"  in sample else None,
#         }
#     # Path 2: pos+theta
#     if "pos" in sample and "theta" in sample:
#         pos = np.asarray(sample["pos"], np.float32)
#         assert pos.ndim == 2 and pos.shape[1] == 2, "pos must be [T,2]"
#         return {
#             "x_img": pos[:, 0],
#             "y_img": pos[:, 1],
#             "theta_rad": np.asarray(sample["theta"], np.float32),
#             "state": np.asarray(sample.get("state", None), np.float32) if "state" in sample else None,
#             "done":  np.asarray(sample.get("done",  None), np.float32) if "done"  in sample else None,
#         }
#     # Path 3: integrate actions
#     if "actions" in sample:
#         x, y, th = _integrate_actions_to_xytheta(sample["actions"], theta0=float(sample.get("theta0", 0.0)))
#         return {
#             "x_img": x, "y_img": y, "theta_rad": th,
#             "state": np.asarray(sample.get("state", None), np.float32) if "state" in sample else None,
#             "done":  np.asarray(sample.get("done",  None), np.float32) if "done"  in sample else None,
#         }
#     # Nested common alt
#     if "trajectory" in sample and isinstance(sample["trajectory"], dict):
#         return _adapt_exp1_item_to_rollout(sample["trajectory"])

#     raise ValueError("Experiment1Dataset item missing required keys. Provide x_img/y_img/theta_rad OR pos/theta OR actions.")



def _maybe_deg_to_rad(theta_arr: np.ndarray) -> np.ndarray:
    # Convert to radians if values look like degrees (|theta| > 2pi somewhere)
    if np.max(np.abs(theta_arr)) > (2 * np.pi + 1e-6):
        return np.deg2rad(theta_arr)
    return theta_arr

def _truncate_at_first_done(x, y, th, st, dn):
    if dn is None:
        return x, y, th, st, None
    dn_bool = dn.astype(bool)
    if dn_bool.any():
        idx = int(np.argmax(dn_bool))
        end = idx + 1
        return x[:end], y[:end], th[:end], (None if st is None else st[:end]), dn[:end]
    return x, y, th, st, dn

def _adapt_exp1_item_to_rollout(
    item, *,
    select_demo: int | None = None,      # pick a specific N; None => all N
    select_agent: int | None = None,     # pick a specific A; None => all A
    flip_y: bool = False,                # set True if your y is world-up and you want image y-down
    screen_height: float | None = None,  # required if flip_y=True (y_img = H-1 - y_world)
    drop_empty: bool = True              # skip sequences with L==0 after truncation
) -> list[dict]:
    """
    Convert Experiment1Item → list of rollout dicts:
      {"x_img": [T], "y_img": [T], "theta_rad": [T], "state": [T], "done": [T]}
    Input:
      item.demo_agent_info: torch.Tensor [B, N, L, A, 6] with fields (x,y,theta,state,time,done)
    Output:
      One rollout per (b, n, a) unless select_* restricts; sequences are time-sorted and
      truncated at first done==True if present. theta auto-converted to radians if needed.
    """
    assert hasattr(item, "demo_agent_info"), "item must have 'demo_agent_info'"
    t = item.demo_agent_info
    assert isinstance(t, torch.Tensor) and t.ndim == 5 and t.shape[-1] >= 6, \
        "demo_agent_info must be [B, N, L, A, 6]"

    # Move to CPU numpy once
    arr = t.detach().cpu().numpy()  # shape [B, N, L, A, 6]
    B, N, L, A, F = arr.shape
    assert F >= 6, "last dim must contain (x,y,theta,state,time,done)"

    demos = range(N) if select_demo is None else [select_demo]
    agents = range(A) if select_agent is None else [select_agent]

    rollouts: list[dict] = []
    for b in range(B):
        for n in demos:
            for a in agents:
                seq = arr[b, n, :, a, :]           # [L, 6]
                if seq.size == 0:
                    continue
                x     = seq[:, 0].astype(np.float32)
                y     = seq[:, 1].astype(np.float32)
                theta = seq[:, 2].astype(np.float32)
                state = seq[:, 3].astype(np.float32)
                time  = seq[:, 4].astype(np.float32)
                done  = seq[:, 5].astype(np.float32)

                # Sort by time if not strictly increasing (robust)
                if not (np.all(np.diff(time) >= 0)):
                    order = np.argsort(time)
                    x, y, theta, state, time, done = x[order], y[order], theta[order], state[order], time[order], done[order]

                # Truncate at first done==True (if any)
                x, y, theta, state, done = _truncate_at_first_done(x, y, theta, state, done)

                # Optionally flip Y (world-up → image-down)
                if flip_y:
                    if screen_height is None:
                        raise ValueError("flip_y=True requires screen_height.")
                    y = (float(screen_height) - 1.0 - y).astype(np.float32)

                # Ensure radians
                theta = _maybe_deg_to_rad(theta).astype(np.float32)

                # Guard: empty after truncation?
                if drop_empty and (x.size == 0):
                    continue

                rollouts.append({
                    "x_img":    x,
                    "y_img":    y,
                    "theta_rad":theta,
                    "state":    state,
                    "done":     (done != 0).astype(np.float32),
                    # Optional: you can carry IDs for grouping if you want
                    # "meta": {"b": b, "n": n, "a": a}
                })
    return rollouts

def validate_split_alignment(split, K=5):
    import numpy as np, random
    P_counts = [len(te) for te in split["token_end"]]
    cumP = np.cumsum([0] + P_counts)
    ok = True
    idxs = random.sample(range(len(P_counts)), k=min(K, len(P_counts)))
    for i in idxs:
        s, e = int(cumP[i]), int(cumP[i+1])
        toks = split["tokens"][i]
        labels = [VOCAB[j] for j in split["y_next"][s:e].tolist()]
        expected = toks[1:] + ['<EOS>']
        if labels != expected:
            ok = False
            print(f"[MISMATCH] i={i}")
            print("  tokens  :", toks)
            print("  labels  :", labels)
            print("  expected:", expected)
    print("All good!" if ok else "Found mismatches.")
    return ok
import random 
def validate_prefix_alignment(dataset, K=5):
    P_counts = [len(te) for te in dataset["token_end"]]
    cumP = np.cumsum([0] + P_counts)
    idxs = random.sample(range(len(P_counts)), k=min(K, len(P_counts)))
    ok = True
    for i in idxs:
        s, e = int(cumP[i]), int(cumP[i+1])
        toks = dataset["tokens"][i]
        labels = [VOCAB[j] for j in dataset["y_next"][s:e].tolist()]
        expected = toks[1:] + ['<EOS>']
        if labels != expected:
            ok = False
            print(f"[MISMATCH] i={i}")
            print("  tokens  :", toks)
            print("  labels  :", labels)
            print("  expected:", expected)
    print("All good!" if ok else "Found mismatches.")
    return ok


def run_experiment1_with_handler(
    experiment1_dataset,                 
    demo_handler,       
    max_episodes: int | None = None,     
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    1) Uses *frozen* demo_handler (state dict).
    2) Pulls trajectories from Experiment1Dataset.
    3) Tokenises/embeds via SequenceProbes.
    4) Evaluates sequence retrieval + linear probe (next-token).

    Returns dict with metrics.
    """

    # 1) Wrap frozen handler
    runner = SequenceProbes(demo_handler=demo_handler, device=device)

    # 2) Gather rollouts from Experiment1Dataset
    N = len(experiment1_dataset)
    use_N = N if max_episodes is None else min(N, int(max_episodes))
    xs, ys, ths, sts, dns = [], [], [], [], []
    for i in range(use_N):
        item = experiment1_dataset[i]
        rollouts_i = _adapt_exp1_item_to_rollout(item)
        
        for r in rollouts_i:
            xs.append(r["x_img"])
            ys.append(r["y_img"])
            ths.append(r["theta_rad"])
            sts.append(r.get("state", None))
            dns.append(r.get("done", None))
    rollouts = _make_rollouts_from_xytheta(xs, ys, ths, sts, dns)

    # 3) Tokeniser calibration + dataset build
    params  = runner.autocalibrate_from_rollouts(rollouts)
    dataset = runner.build_dataset(rollouts, params, prefer_last_step=False)
    validate_prefix_alignment(dataset)

    ## deb
    # from collections import Counter

    # counts = Counter(dataset["y_next"].tolist())
    # total  = sum(counts.values())
    # for i in range(len(VOCAB)):
    #     print(i, VOCAB[i], counts.get(i,0), f"{100*counts.get(i,0)/total:.1f}%")
        
    # i = 0  # some rollout index
    # print(dataset["tokens"][i])
    # print(dataset["token_end"][i][:10])
    # print([VOCAB[j] for j in dataset["y_next"][:10].tolist()])
    ##
    
    # 4) Split by base_id to avoid leakage
    splits = split_by_base_id(dataset, train_frac=train_frac, val_frac=val_frac, seed=0)
    ## db
    # validate_split_alignment(splits["val"])
    # from .exp1_aux import poincare_log0
    # Z = splits["train"]["Zp"][:200]         # prefix embeddings on the ball
    # V = poincare_log0(Z, c=runner.c)        # tangent
    # print("tangent std per dim:", V.std(0))

    # Zr = splits["train"]["z_rollout"]
    # print("pairwise mean distance:", torch.pdist(poincare_log0(Zr, c=runner.c)).mean())
    # Prefix embeddings (ball) -> tangent; check variance
    ###

    # baselines
    # K = len(VOCAB)
    # Ztr, ytr = splits["train"]["Zp"], splits["train"]["y_next"]
    # Zva, yva = splits["val"]["Zp"],   splits["val"]["y_next"]
    # Zte, yte = splits["test"]["Zp"],  splits["test"]["y_next"]

    # mlp, best_val = train_mlp_probe(
    #     Ztr, ytr, Zva, yva,
    #     K=K, curvature=runner.c,
    #     hidden=128, epochs=60, lr=1e-3, wd=1e-3, use_class_weights=True
    # )

    # tokens_tr = splits["train"]["tokens"]
    # tokens_va = splits["val"]["tokens"]
    # tokens_te = splits["test"]["tokens"]

    # # Last-token baseline
    # lt_map = last_token_baseline_train(tokens_tr)
    # y_true_va, y_pred_va, lt_val = last_token_baseline_eval(lt_map, tokens_va)
    # y_true_te, y_pred_te, lt_test = last_token_baseline_eval(lt_map, tokens_te)
    # print(f"[Baseline] Last-token — Val: {lt_val:.3f}  Test: {lt_test:.3f}")

    # # Bigram baseline
    # bi_tbl = bigram_table_train(tokens_tr)
    # y_true_va, y_pred_va, bi_val = bigram_table_eval(bi_tbl, tokens_va)
    # y_true_te, y_pred_te, bi_test = bigram_table_eval(bi_tbl, tokens_te)
    # print(f"[Baseline] Bigram — Val: {bi_val:.3f}  Test: {bi_test:.3f}")

    # with torch.no_grad():
    #     Zte_tan = poincare_log0(Zte, c=runner.c)
    #     test_acc = (mlp(Zte_tan).argmax(-1) == yte).float().mean().item()

    # print(f"[Next-token] MLP (2x128) — Val: {best_val:.3f}  Test: {test_acc:.3f}")


    # # 5) Sequence retrieval (z_rollout)
    # ref_Z, ref_ids = build_references(
    #     splits["train"]["z_rollout"], splits["train"]["base_id"], curvature=runner.c
    # )
    # acc1 = topk_retrieval_acc(
    #     splits["test"]["z_rollout"], splits["test"]["base_id"],
    #     ref_Z, ref_ids, k=1, curvature=runner.c
    # )
    # if plot :
    #     ks = [1, 2, 3, 5, 10]
    #     curve = retrieval_curve(
    #         splits["test"]["z_rollout"], splits["test"]["base_id"],
    #         ref_Z, ref_ids, ks, curvature=runner.c
    #     )

    #     # Save figure + CSV
    #     plot_dir = "./probe_plots"
    #     os.makedirs(plot_dir, exist_ok=True)
    #     plot_retrieval_curve(curve, out_png=os.path.join(plot_dir, "retrieval_curve.png"))

    #     # Optional CSV for the paper’s SI
    #     with open(os.path.join(plot_dir, "retrieval_curve.csv"), "w") as f:
    #         f.write("k,accuracy\n")
    #         for k,a in curve:
    #             f.write(f"{k},{a:.6f}\n")

    # baselines
    base_ids = dataset["base_id"]
    sup_idx, qry_idx = per_base_support_query_indices(base_ids, support_frac=0.5, seed=0)

    Zsup = torch.stack([dataset["z_rollout"][i] for i in sup_idx])
    sup_ids = [base_ids[i] for i in sup_idx]
    Zqry = torch.stack([dataset["z_rollout"][i] for i in qry_idx])
    qry_ids = [base_ids[i] for i in qry_idx]

    ref_Z, ref_ids = build_references(Zsup, sup_ids, curvature=runner.c)
    acc1 = topk_retrieval_acc(Zqry, qry_ids, ref_Z, ref_ids, k=1, curvature=runner.c)
    acc5 = topk_retrieval_acc(Zqry, qry_ids, ref_Z, ref_ids, k=5, curvature=runner.c)
    print(f"[Retrieval] Embedding — Top-1: {acc1:.3f}  Top-5: {acc5:.3f}")
    tok_acc = token_string_retrieval_acc(dataset["tokens"], base_ids, sup_idx, qry_idx)
    print(f"[Retrieval] Token-string — Top-1: {tok_acc:.3f}")
    ###
    # 6) Linear probe (next token from per-step embeddings)
    K = vocab_size()
    probe , val_acc = train_linear_probe(
        splits["train"]["Zp"], splits["train"]["y_next"],
        splits["val"]["Zp"],   splits["val"]["y_next"],
        K=K, curvature=runner.c
    )   

    if plot:
        # 1) Get predictions on the test split
        K = vocab_size()
        probe.eval()
        with torch.no_grad():
            logits = probe(splits["test"]["Zp"])    # shape [N, K]
            y_pred = torch.argmax(logits, dim=-1).cpu().numpy()
            # What column is bright?
            # After computing logits on val/test:
            pred = logits.argmax(-1)
            print("Most predicted class:", pred.mode().values.item())
            print("VOCAB order:", list(enumerate(VOCAB)))

        y_true = splits["test"]["y_next"].cpu().numpy()

        # 2) Build and save CM
        cm = confusion_matrix_from_predictions(y_true, y_pred, K)
        plot_dir = "./probe_plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_confusion_matrix(cm, out_png=os.path.join(plot_dir, "confusion_matrix.png"))

        # Optional CSV
        np.savetxt(os.path.join(plot_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")


    # 7) Report
    print("\n=== Experiment1 • Frozen demo_handler • Results ===")
    print(f"retrieval@1:             {acc1:.3f}")
    print(f"linear probe (val acc):  {val_acc:.3f}")
    print(f"episodes used:           {use_N}")
    return {"retrieval@1": float(acc1), "linear_probe_val_acc": float(val_acc)}


if __name__ == "__main__":
    # Load frozen demo_handler state dict
    from agent import DemoHandler
    from configs import ANGULAR_GRANULITIES
    cwd = os.curdir
    fp = cwd + '/experiments/demo_handler.pth'
    handler_sd = torch.load(fp, map_location="cpu")


    from exp1_aux import Experiment1Dataset
    ds = Experiment1Dataset(0.5, 0.1, B = 1)  # fill in ctor args
    handler = DemoHandler(curvature=1, angular_granularities_deg=ANGULAR_GRANULITIES)
    handler.load_state_dict(handler_sd)
    run_experiment1_with_handler(ds, handler, max_episodes=5000, device="cpu")


