import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def plot_retrieval_curve(curve, out_png=None, title="Sequence retrieval accuracy"):
    ks = [k for k,_ in curve]
    accs = [a for _,a in curve]
    plt.figure(figsize=(4.2,3.2))
    plt.plot(ks, accs, marker="o")
    plt.xlabel("k")
    plt.ylabel("accuracy@k")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
    plt.close()



def plot_confusion_matrix(cm, out_png=None, title="Next-token confusion matrix"):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted token")
    plt.ylabel("True token")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
    plt.close()
