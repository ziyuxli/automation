import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PASSIVE_LOG = "output_passive/fracturemnist3d/resnet50passive-model1/260331_182508/fracturemnist3d_passive_log.txt"
GAUSSIAN_LOG = "output_gaussian/fracturemnist3d/resnet50_passive-model1/260418_212655/fracturemnist3d_entropy_log.txt"

SAVE_PATH = "comparison_4panel_clean.png"


# -----------------------------
# Parse
# -----------------------------
def parse_log(log_text, method_name):
    round_header = re.compile(r"\[Round\s+(\d+)\]\s+labeled=(\d+)")
    val_pattern = re.compile(
        r"epoch\s+\d+\s+(?:train loss:\s+[0-9.]+\s+)?val loss:\s+([0-9.]+)\s+auc:\s+([0-9.]+)"
    )
    test_pattern = re.compile(
        r"test\s+loss:\s+([0-9.]+)\s+auc:\s+([0-9.]+)"
    )

    lines = log_text.splitlines()

    results = []
    cur_round, cur_labeled = None, None
    val_losses, val_aucs = [], []

    def flush(test_match):
        results.append({
            "method": method_name,
            "round": cur_round,
            "labeled": cur_labeled,

            "val_loss_mean": np.mean(val_losses),
            "val_loss_std": np.std(val_losses),

            "val_auc_mean": np.mean(val_aucs),
            "val_auc_std": np.std(val_aucs),

            "test_loss": float(test_match.group(1)),
            "test_auc": float(test_match.group(2)),
        })

    for line in lines:
        line = line.strip()

        h = round_header.search(line)
        if h:
            cur_round = int(h.group(1))
            cur_labeled = int(h.group(2))
            val_losses, val_aucs = [], []
            continue

        v = val_pattern.search(line)
        if v:
            val_losses.append(float(v.group(1)))
            val_aucs.append(float(v.group(2)))
            continue

        t = test_pattern.search(line)
        if t:
            flush(t)

    return pd.DataFrame(results)


# -----------------------------
# Plot helpers
# -----------------------------
def plot_with_variance(ax, x, mean, std, label, color):
    ax.plot(x, mean, lw=2.5, marker="o", label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def plot_simple(ax, x, y, label, color):
    ax.plot(x, y, lw=2.5, marker="o", label=label, color=color)


# -----------------------------
# Plot
# -----------------------------
def plot_all(df):
    passive = df[df["method"] == "Passive"].sort_values("labeled")
    gaussian = df[df["method"] == "Gaussian Dropout"].sort_values("labeled")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # -----------------------------
    # Validation Loss (NOW SHADED)
    # -----------------------------
    ax = axes[0, 0]
    plot_with_variance(ax,
                       passive["labeled"],
                       passive["val_loss_mean"],
                       passive["val_loss_std"],
                       "Passive", "tab:blue")

    plot_with_variance(ax,
                       gaussian["labeled"],
                       gaussian["val_loss_mean"],
                       gaussian["val_loss_std"],
                       "Gaussian", "tab:orange")

    ax.set_title("Validation Loss")
    ax.set_ylabel("Loss")

    # -----------------------------
    # Validation AUC (SHADED)
    # -----------------------------
    ax = axes[0, 1]
    plot_with_variance(ax,
                       passive["labeled"],
                       passive["val_auc_mean"],
                       passive["val_auc_std"],
                       "Passive", "tab:blue")

    plot_with_variance(ax,
                       gaussian["labeled"],
                       gaussian["val_auc_mean"],
                       gaussian["val_auc_std"],
                       "Gaussian", "tab:orange")

    ax.set_title("Validation AUC")
    ax.set_ylabel("AUC")

    # final val AUC
    ax.text(0.98, 0.95,
            f"P: {passive.iloc[-1]['val_auc_mean']:.3f}\nG: {gaussian.iloc[-1]['val_auc_mean']:.3f}",
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(fc="white", alpha=0.8))

    # -----------------------------
    # Test Loss
    # -----------------------------
    ax = axes[1, 0]
    plot_simple(ax,
                passive["labeled"],
                passive["test_loss"],
                "Passive", "tab:blue")

    plot_simple(ax,
                gaussian["labeled"],
                gaussian["test_loss"],
                "Gaussian", "tab:orange")

    ax.set_title("Test Loss")
    ax.set_ylabel("Loss")

    # -----------------------------
    # Test AUC
    # -----------------------------
    ax = axes[1, 1]
    plot_simple(ax,
                passive["labeled"],
                passive["test_auc"],
                "Passive", "tab:blue")

    plot_simple(ax,
                gaussian["labeled"],
                gaussian["test_auc"],
                "Gaussian", "tab:orange")

    ax.set_title("Test AUC")
    ax.set_ylabel("AUC")

    # final test AUC
    ax.text(0.98, 0.95,
            f"P: {passive.iloc[-1]['test_auc']:.3f}\nG: {gaussian.iloc[-1]['test_auc']:.3f}",
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(fc="white", alpha=0.8))

    # -----------------------------
    # Styling
    # -----------------------------
    for ax in axes.flatten():
        ax.set_xlabel("Labeled Set Size")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(["Passive", "Gaussian Dropout"],
               loc="upper center",
               ncol=2,
               frameon=False)

    fig.suptitle("Passive vs Gaussian Dropout", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(SAVE_PATH, dpi=300)
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    with open(PASSIVE_LOG) as f:
        passive_text = f.read()

    with open(GAUSSIAN_LOG) as f:
        gaussian_text = f.read()

    passive_df = parse_log(passive_text, "Passive")
    gaussian_df = parse_log(gaussian_text, "Gaussian Dropout")

    df = pd.concat([passive_df, gaussian_df], ignore_index=True)

    plot_all(df)


if __name__ == "__main__":
    main()