import re
import pandas as pd
import matplotlib.pyplot as plt

with open("output_passive/fracturemnist3d/resnet50passive-model1/260331_182508/fracturemnist3d_passive_log.txt", "r", encoding="utf-8") as f:
    text = f.read()

with open("output_gaussian/fracturemnist3d/resnet50_passive-model1/260418_212655/fracturemnist3d_entropy_log.txt", "r", encoding="utf-8") as f:
    text = f.read()

round_pattern = re.compile(
    r"\[Round (\d+)\] labeled=(\d+)(.*?)(?=\[Round \d+\] labeled=|\Z)", re.S
)
val_pattern = re.compile(
    r"epoch\s+(\d+)\s+val loss:\s+([0-9.]+)\s+auc:\s+([0-9.]+)\s+acc:\s+([0-9.]+)"
)
test_pattern = re.compile(
    r"test\s+loss:\s+([0-9.]+)\s+auc:\s+([0-9.]+)\s+acc:\s+([0-9.]+)"
)

val_rows = []
test_rows = []
summary_rows = []

for m in round_pattern.finditer(text):
    round_id = int(m.group(1))
    labeled = int(m.group(2))
    block = m.group(3)

    round_val_aucs = []

    for vm in val_pattern.finditer(block):
        epoch = int(vm.group(1))
        val_auc = float(vm.group(3))
        val_rows.append({
            "round": round_id,
            "labeled": labeled,
            "epoch": epoch,
            "val_auc": val_auc
        })
        round_val_aucs.append(val_auc)

    tm = test_pattern.search(block)
    if tm:
        test_auc = float(tm.group(2))
        test_rows.append({
            "round": round_id,
            "labeled": labeled,
            "test_auc": test_auc
        })

    if round_val_aucs and tm:
        s = pd.Series(round_val_aucs)
        summary_rows.append({
            "round": round_id,
            "labeled": labeled,
            "val_auc_mean": s.mean(),
            "val_auc_median": s.median(),
            "val_auc_min": s.min(),
            "val_auc_max": s.max(),
            "test_auc": float(tm.group(2))
        })

val_df = pd.DataFrame(val_rows)
test_df = pd.DataFrame(test_rows).sort_values("labeled")
summary_df = pd.DataFrame(summary_rows).sort_values("labeled")

# 主图：validation中位数 + 区间 + test
plt.figure(figsize=(8, 5))
plt.plot(summary_df["labeled"], summary_df["val_auc_median"], marker="o", label="Validation AUC (median)")
plt.fill_between(
    summary_df["labeled"],
    summary_df["val_auc_min"],
    summary_df["val_auc_max"],
    alpha=0.15,
    label="Validation range"
)
plt.plot(summary_df["labeled"], summary_df["test_auc"], marker="s", label="Test AUC")
plt.xlabel("Number of labeled samples")
plt.ylabel("AUC")
plt.title("Validation and Test AUC across rounds")
plt.legend()
plt.tight_layout()
plt.savefig("plot2.png")
# 附图：所有 validation epoch 散点 + test线
plt.figure(figsize=(8, 5))
plt.scatter(val_df["labeled"], val_df["val_auc"], alpha=0.5, label="Validation AUC (all epochs)")
plt.plot(test_df["labeled"], test_df["test_auc"], marker="s", label="Test AUC")
plt.xlabel("Number of labeled samples")
plt.ylabel("AUC")
plt.title("Validation Process and Test AUC")
plt.legend()
plt.tight_layout()
plt.savefig("plot1.png")