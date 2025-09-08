import numpy as np
import torch
import os

from plot_aux import (
    plot_retrieval_curve,
    plot_confusion_matrix
)

from exp1_aux import (
    SequenceProbes,
    split_by_base_id,
    build_references,
    topk_retrieval_acc,
    train_linear_probe,
    vocab_size,
    retrieval_curve,
    confusion_matrix_from_predictions
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

def _adapt_exp1_item_to_rollout(sample):
    """
    Flexible adapter for Experiment1Dataset item → rollout dict.
    Supported keys (any one path is enough):
      1) 'x_img','y_img','theta_rad'          (preferred; y-down)
      2) 'pos' [T,2] + 'theta' [T]
      3) 'actions' [T,3+] (dx,dy,dtheta in y-down)  → integrate
    Optional: 'state', 'done'
    """
    # Path 1: already in image frame
    if all(k in sample for k in ("x_img", "y_img", "theta_rad")):
        return {
            "x_img": np.asarray(sample["x_img"], np.float32),
            "y_img": np.asarray(sample["y_img"], np.float32),
            "theta_rad": np.asarray(sample["theta_rad"], np.float32),
            "state": np.asarray(sample.get("state", None), np.float32) if "state" in sample else None,
            "done":  np.asarray(sample.get("done",  None), np.float32) if "done"  in sample else None,
        }
    # Path 2: pos+theta
    if "pos" in sample and "theta" in sample:
        pos = np.asarray(sample["pos"], np.float32)
        assert pos.ndim == 2 and pos.shape[1] == 2, "pos must be [T,2]"
        return {
            "x_img": pos[:, 0],
            "y_img": pos[:, 1],
            "theta_rad": np.asarray(sample["theta"], np.float32),
            "state": np.asarray(sample.get("state", None), np.float32) if "state" in sample else None,
            "done":  np.asarray(sample.get("done",  None), np.float32) if "done"  in sample else None,
        }
    # Path 3: integrate actions
    if "actions" in sample:
        x, y, th = _integrate_actions_to_xytheta(sample["actions"], theta0=float(sample.get("theta0", 0.0)))
        return {
            "x_img": x, "y_img": y, "theta_rad": th,
            "state": np.asarray(sample.get("state", None), np.float32) if "state" in sample else None,
            "done":  np.asarray(sample.get("done",  None), np.float32) if "done"  in sample else None,
        }
    # Nested common alt
    if "trajectory" in sample and isinstance(sample["trajectory"], dict):
        return _adapt_exp1_item_to_rollout(sample["trajectory"])

    raise ValueError("Experiment1Dataset item missing required keys. Provide x_img/y_img/theta_rad OR pos/theta OR actions.")



def run_experiment1_with_handler(
    experiment1_dataset,                 
    demo_handler_state_dict: dict,       
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
    runner = SequenceProbes(handler_state_dict=demo_handler_state_dict, device=device)

    # 2) Gather rollouts from Experiment1Dataset
    N = len(experiment1_dataset)
    use_N = N if max_episodes is None else min(N, int(max_episodes))
    xs, ys, ths, sts, dns = [], [], [], [], []
    for i in range(use_N):
        item = experiment1_dataset[i]
        r = _adapt_exp1_item_to_rollout(item)
        xs.append(r["x_img"]); ys.append(r["y_img"]); ths.append(r["theta_rad"])
        sts.append(r.get("state", None)); dns.append(r.get("done", None))
    rollouts = _make_rollouts_from_xytheta(xs, ys, ths, sts, dns)

    # 3) Tokeniser calibration + dataset build
    params  = runner.autocalibrate_from_rollouts(rollouts)
    dataset = runner.build_dataset(rollouts, params, prefer_last_step=False)

    # 4) Split by base_id to avoid leakage
    splits = split_by_base_id(dataset, train_frac=train_frac, val_frac=val_frac, seed=0)

    # 5) Sequence retrieval (z_rollout)
    ref_Z, ref_ids = build_references(
        splits["train"]["z_rollout"], splits["train"]["base_id"], curvature=runner.c
    )
    acc1 = topk_retrieval_acc(
        splits["test"]["z_rollout"], splits["test"]["base_id"],
        ref_Z, ref_ids, k=1, curvature=runner.c
    )
    if plot :
        ks = [1, 2, 3, 5, 10]
        curve = retrieval_curve(
            splits["test"]["z_rollout"], splits["test"]["base_id"],
            ref_Z, ref_ids, ks, curvature=runner.c
        )

        # Save figure + CSV
        plot_dir = "./probe_plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_retrieval_curve(curve, out_png=os.path.join(plot_dir, "retrieval_curve.png"))

        # Optional CSV for the paper’s SI
        with open(os.path.join(plot_dir, "retrieval_curve.csv"), "w") as f:
            f.write("k,accuracy\n")
            for k,a in curve:
                f.write(f"{k},{a:.6f}\n")


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
    ckpt = torch.load("path/to/ckpt.pt", map_location="cpu")
    if "demo_handler" in ckpt and isinstance(ckpt["demo_handler"], dict):
        handler_sd = ckpt["demo_handler"]
    elif "policy" in ckpt and "demo_handler" in ckpt["policy"]:
        handler_sd = ckpt["policy"]["demo_handler"]
    else:
        handler_sd = ckpt  # plain state_dict

    # Build your Experiment1Dataset
    from exp1_aux import Experiment1Dataset
    ds = Experiment1Dataset(0.5, 0.1, B = 1)  # fill in ctor args

    run_experiment1_with_handler(ds, handler_sd, max_episodes=256, device="cuda")


