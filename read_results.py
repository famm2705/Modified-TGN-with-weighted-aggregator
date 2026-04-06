import argparse
import pickle
import numpy as np
from pathlib import Path

BASE_PATH = "/content/drive/MyDrive/tgn_results"

parser = argparse.ArgumentParser("Read TGN experiment results")
parser.add_argument(
    "--agg",
    type=str,
    required=True,
    choices=["mean", "last", "weightedmean", "attention"],
    help="Aggregator type"
)
parser.add_argument(
    "--data",
    type=str,
    required=True,
    choices=["wikipedia", "reddit", "toy_sparse", "toy_dense"],
    help="Dataset name"
)

args = parser.parse_args()

file_path = f"{BASE_PATH}/tgn_{args.data}_{args.agg}.pkl"

if not Path(file_path).exists():
    raise FileNotFoundError(f"Results file not found: {file_path}")

with open(file_path, "rb") as f:
    data = pickle.load(f)

# Core metrics
test_ap = data.get("test_ap")
test_auc = data.get("test_auc")

# New metrics
test_acc = data.get("test_acc")
test_prec = data.get("test_prec")
test_rec = data.get("test_rec")
test_f1 = data.get("test_f1")
test_mrr = data.get("test_mrr")
test_hits1 = data.get("test_hits1")
test_hits5 = data.get("test_hits5")
test_hits10 = data.get("test_hits10")

# New node metrics
new_node_test_ap = data.get("new_node_test_ap")
new_node_test_auc = data.get("new_node_test_auc")

# Validation history
val_aps = data.get("val_aps", [])
val_aucs = data.get("val_aucs", [])

epoch_times = data.get("epoch_times", [])
total_epoch_times = data.get("total_epoch_times", [])

# Summary statistics
best_val_ap = np.max(val_aps) if len(val_aps) else None
best_val_auc = np.max(val_aucs) if len(val_aucs) else None

avg_epoch_time = np.mean(total_epoch_times) if len(total_epoch_times) else None
total_train_time = np.sum(total_epoch_times) if len(total_epoch_times) else None


def fmt(x):
    return f"{x:.4f}" if x is not None else "N/A"


print("\n===== TGN Aggregator Results =====")
print(f"Aggregator: {args.agg}")
print(f"Dataset: {args.data}")
print("----------------------------------")

print("Validation")
print(f"Best Validation AP:  {fmt(best_val_ap)}")
print(f"Best Validation AUC: {fmt(best_val_auc)}")

print("----------------------------------")
print("Test (Old Nodes)")

print(f"AP:  {fmt(test_ap)}")
print(f"AUC: {fmt(test_auc)}")
print(f"ACC: {fmt(test_acc)}")
print(f"Precision: {fmt(test_prec)}")
print(f"Recall:    {fmt(test_rec)}")
print(f"F1:        {fmt(test_f1)}")

print("----------------------------------")
print("Ranking Metrics")

print(f"MRR:    {fmt(test_mrr)}")
print(f"Hits@1: {fmt(test_hits1)}")

print("----------------------------------")
print("New Node Test")

print(f"AP:  {fmt(new_node_test_ap)}")
print(f"AUC: {fmt(new_node_test_auc)}")

print("----------------------------------")
print("Training Time")

print(f"Average Epoch Time: {avg_epoch_time:.2f} s" if avg_epoch_time else "Average Epoch Time: N/A")
print(f"Total Training Time: {total_train_time:.2f} s" if total_train_time else "Total Training Time: N/A")

print("==================================\n")