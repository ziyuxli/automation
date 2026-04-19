import re
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Read log file
# =========================
# log_path = "output_passive/fracturemnist3d/resnet50passive-model1/260331_182508/fracturemnist3d_passive_log.txt"
log_path = "output_gaussian/fracturemnist3d/resnet50_passive-model1/260418_212655/fracturemnist3d_entropy_log.txt"
with open(log_path, "r", encoding="utf-8") as f:
    text = f.read()

# =========================
# 2. Regex patterns
# =========================
round_pattern = re.compile(
    r"\[Round (\d+)\] labeled=(\d+)(.*?)(?=\[Round \d+\] labeled=|\Z)",
    re.S
)

val_pattern = re.compile(
    r"epoch\s+(\d+)\s+val loss:\s+([0-9.]+)\s+auc:\s+([0-9.]+)\s+acc:\s+([0-9.]+)"
)

test_pattern = re.compile(
    r"test\s+loss:\s+([0-9.]+)\s+auc:\s+([0-9.]+)\s+acc:\s+([0-9.]+)"
)

# =========================
# 3. Parse log
# =========================
rows = []

for round_match in round_pattern.finditer(text):
    round_id = int(round_match.group(1))
    labeled = int(round_match.group(2))
    block = round_match.group(3)

    vals = []
    for vm in val_pattern.finditer(block):
        epoch = int(vm.group(1))
        val_loss = float(vm.group(2))
        val_auc = float(vm.group(3))
        val_acc = float(vm.group(4))
        vals.append({
            "round": round_id,
            "labeled": labeled,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_acc": val_acc
        })

    tm = test_pattern.search(block)
    if tm is None or len(vals) == 0:
        continue

    test_loss = float(tm.group(1))
    test_auc = float(tm.group(2))
    test_acc = float(tm.group(3))

    val_df = pd.DataFrame(vals)

    rows.append({
        "round": round_id,
        "labeled": labeled,

        # validation mean
        "mean_val_auc": val_df["val_auc"].mean(),
        "mean_val_acc": val_df["val_acc"].mean(),
        "mean_val_loss": val_df["val_loss"].mean(),

        # validation median
        "median_val_auc": val_df["val_auc"].median(),
        "median_val_acc": val_df["val_acc"].median(),
        "median_val_loss": val_df["val_loss"].median(),

        # validation range
        "val_auc_min": val_df["val_auc"].min(),
        "val_auc_max": val_df["val_auc"].max(),
        "val_acc_min": val_df["val_acc"].min(),
        "val_acc_max": val_df["val_acc"].max(),
        "val_loss_min": val_df["val_loss"].min(),
        "val_loss_max": val_df["val_loss"].max(),

        # test
        "test_auc": test_auc,
        "test_acc": test_acc,
        "test_loss": test_loss,
    })

summary = pd.DataFrame(rows).sort_values("labeled").reset_index(drop=True)

if summary.empty:
    raise ValueError("No valid rounds were parsed from the log file.")

print(summary.head())

# =========================
# 4. Plot style
# =========================
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 11

# =========================
# 5. Plot AUC
# =========================
plt.figure()
plt.plot(
    summary["labeled"],
    summary["mean_val_auc"],
    marker="o",
    linewidth=2,
    label="Validation AUC (mean)"
)
plt.fill_between(
    summary["labeled"],
    summary["val_auc_min"],
    summary["val_auc_max"],
    alpha=0.15,
    label="Validation AUC range"
)
plt.plot(
    summary["labeled"],
    summary["test_auc"],
    marker="s",
    linewidth=2,
    label="Test AUC"
)
plt.xlabel("Number of labeled samples")
plt.ylabel("AUC")
plt.title("Validation vs Test AUC across Rounds")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plot_auc_mean.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 6. Plot ACC
# =========================
plt.figure()
plt.plot(
    summary["labeled"],
    summary["mean_val_acc"],
    marker="o",
    linewidth=2,
    label="Validation ACC (mean)"
)
plt.fill_between(
    summary["labeled"],
    summary["val_acc_min"],
    summary["val_acc_max"],
    alpha=0.15,
    label="Validation ACC range"
)
plt.plot(
    summary["labeled"],
    summary["test_acc"],
    marker="s",
    linewidth=2,
    label="Test ACC"
)
plt.xlabel("Number of labeled samples")
plt.ylabel("Accuracy")
plt.title("Validation vs Test Accuracy across Rounds")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plot_acc_mean.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 7. Plot Loss
# =========================
plt.figure()
plt.plot(
    summary["labeled"],
    summary["mean_val_loss"],
    marker="o",
    linewidth=2,
    label="Validation Loss (mean)"
)
plt.fill_between(
    summary["labeled"],
    summary["val_loss_min"],
    summary["val_loss_max"],
    alpha=0.15,
    label="Validation Loss range"
)
plt.plot(
    summary["labeled"],
    summary["test_loss"],
    marker="s",
    linewidth=2,
    label="Test Loss"
)
plt.xlabel("Number of labeled samples")
plt.ylabel("Loss")
plt.title("Validation vs Test Loss across Rounds")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plot_loss_mean.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 8. Optional: gap plot
# =========================
plt.figure(figsize=(8, 4))
gap_auc = summary["mean_val_auc"] - summary["test_auc"]
plt.plot(summary["labeled"], gap_auc, marker="o", linewidth=2)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Number of labeled samples")
plt.ylabel("Validation Mean AUC - Test AUC")
plt.title("Generalization Gap (AUC)")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plot_auc_gap.png", dpi=300, bbox_inches="tight")
plt.show()