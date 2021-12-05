import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_history(path="logs/history", metric="Losses"):
    hist_path = list(filter(lambda x: metric.lower() in x, os.listdir(path)))

    for p in hist_path:
        if p.startswith(metric.lower()):
            path = os.path.join(path, p)

    history = np.load(path)
    train_losses = history["train"]
    val_losses = history["valid"]

    return train_losses, val_losses


def plot_history(train_loss, val_loss, save_dir, metric="loss", verbose=None):
    """Returns the plots of the training and validation of specified metrics.
    Plots of loss is returned if metric is `loss` or plots of accuracy is returned
    if metrics is `acc`.

    Parameters:
        train_loss: (pd.DataFrame) training losses.
        val_loss: (pd.DataFrame) validation losses.
        save_dir: (str) Directory to save output plot.
        metric: (str) metric to monitor. Default is "loss".
        verbose: (bool) to display the plot. Default is None.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # axes parameters
    font = {"family": "sans-serif", "color": "black", "weight": "normal", "size": 13}

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(train_loss)), train_loss, color="#ffa500", label="Training")
    plt.plot(range(len(val_loss)), val_loss, color="green", label="Validation")
    plt.ylabel(f"{metric}", fontdict=font)
    plt.xlabel("Epochs", fontdict=font)
    plt.xlim([0, len(train_loss)])
    plt.legend(loc="best", frameon=False, prop={"size": 12})
    if verbose:
        plt.savefig(f"{save_dir}/{metric}.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    losses = load_history(metric="acc")

    plot_history(*losses, metric="Accuracy", save_dir="logs/plots", verbose=True)
