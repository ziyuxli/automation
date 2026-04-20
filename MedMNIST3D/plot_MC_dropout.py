import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PASSIVE_LOG = "output/fracturemnist3d_passive_log.txt"
BERNOULLI_MC_LOG = "output/fracturemnist3d_bernoulliMC_entropy_log.txt"

SAVE_PATH = "passive_bernoulli.png"


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


def plot_with_variance(ax, x, mean, std, label, color, marker="o"):
    line, = ax.plot(x, mean, lw=2.5, marker=marker, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    return line


def plot_simple(ax, x, y, label, color, marker="o"):
    line, = ax.plot(x, y, lw=2.5, marker=marker, label=label, color=color)
    return line


def plot_all(df):
    passive = df[df["method"] == "Passive"].sort_values("labeled")
    bernoulli_mc = df[df["method"] == "Bernoulli MC Dropout"].sort_values("labeled")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    color_passive = "tab:blue"
    color_MC = "tab:red"

    # -----------------------------
    # Validation Loss
    # -----------------------------
    ax = axes[0, 0]
    h_passive = plot_with_variance(
        ax,
        passive["labeled"].values,
        passive["val_loss_mean"].values,
        passive["val_loss_std"].values,
        "Passive",
        color_passive,
        marker="o",
    )
    h_mc = plot_with_variance(
        ax,
        bernoulli_mc["labeled"].values,
        bernoulli_mc["val_loss_mean"].values,
        bernoulli_mc["val_loss_std"].values,
        "Bernoulli MC Dropout",
        color_MC,
        marker="s",
    )
    ax.set_title("Validation Loss")
    ax.set_ylabel("Loss")

    # -----------------------------
    # Validation AUC
    # -----------------------------
    ax = axes[0, 1]
    plot_with_variance(
        ax,
        passive["labeled"].values,
        passive["val_auc_mean"].values,
        passive["val_auc_std"].values,
        "Passive",
        color_passive,
        marker="o",
    )
    plot_with_variance(
        ax,
        bernoulli_mc["labeled"].values,
        bernoulli_mc["val_auc_mean"].values,
        bernoulli_mc["val_auc_std"].values,
        "Bernoulli MC  Dropout",
        color_MC,
        marker="s",
    )
    ax.set_title("Validation AUC")
    ax.set_ylabel("AUC")
    ax.text(
        0.98, 0.95,
        f"P: {passive.iloc[-1]['val_auc_mean']:.3f}\nG: {bernoulli_mc.iloc[-1]['val_auc_mean']:.3f}",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(fc="white", alpha=0.8, edgecolor="0.8")
    )

    # -----------------------------
    # Test Loss
    # -----------------------------
    ax = axes[1, 0]
    plot_simple(
        ax,
        passive["labeled"].values,
        passive["test_loss"].values,
        "Passive",
        color_passive,
        marker="o",
    )
    plot_simple(
        ax,
        bernoulli_mc["labeled"].values,
        bernoulli_mc["test_loss"].values,
        "Bernoulli MC Dropout",
        color_MC,
        marker="s",
    )
    ax.set_title("Test Loss")
    ax.set_ylabel("Loss")

    # -----------------------------
    # Test AUC
    # -----------------------------
    ax = axes[1, 1]
    plot_simple(
        ax,
        passive["labeled"].values,
        passive["test_auc"].values,
        "Passive",
        color_passive,
        marker="o",
    )
    plot_simple(
        ax,
        bernoulli_mc["labeled"].values,
        bernoulli_mc["test_auc"].values,
        "Bernoulli MC  Dropout",
        color_MC,
        marker="s",
    )
    ax.set_title("Test AUC")
    ax.set_ylabel("AUC")
    ax.text(
        0.98, 0.95,
        f"P: {passive.iloc[-1]['test_auc']:.3f}\nG: {bernoulli_mc.iloc[-1]['test_auc']:.3f}",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(fc="white", alpha=0.8, edgecolor="0.8")
    )

    # -----------------------------
    # Shared styling
    # -----------------------------
    for ax in axes.flatten():
        ax.set_xlabel("Labeled Set Size")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Proper legend: use real handles
    fig.legend(
        handles=[h_passive, h_mc],
        labels=["Passive", "Bernoulli MC Dropout"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    fig.suptitle("Passive vs Bernoulli MC Dropout", fontsize=18, y=0.995)

    # Leave room at the top for title + legend
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    with open(PASSIVE_LOG, "r", encoding="utf-8") as f:
        passive_text = f.read()

    with open(BERNOULLI_MC_LOG, "r", encoding="utf-8") as f:
        mc_text = f.read()

    passive_df = parse_log(passive_text, "Passive")
    bern_mc_df = parse_log(mc_text, "Bernoulli MC Dropout")

    df = pd.concat([passive_df, bern_mc_df], ignore_index=True)
    plot_all(df)


if __name__ == "__main__":
    main()