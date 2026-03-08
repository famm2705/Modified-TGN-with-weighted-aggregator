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
    choices=["wikipedia", "reddit"],
    help="Dataset name"
)

args = parser.parse_args()

file_path = f"{BASE_PATH}/tgn_{args.agg}_{args.data}.pkl"

if not Path(file_path).exists():
    raise FileNotFoundError(f"Results file not found: {file_path}")

with open(file_path, "rb") as f:
    data = pickle.load(f)

# Extract metrics
test_ap = data.get("test_ap")
test_auc = data.get("test_auc")
new_node_test_ap = data.get("new_node_test_ap")
new_node_test_auc = data.get("new_node_test_auc")

val_aps = data.get("val_aps", [])
val_aucs = data.get("val_aucs", [])

epoch_times = data.get("epoch_times", [])
total_epoch_times = data.get("total_epoch_times", [])

# Compute summary statistics
best_val_ap = np.max(val_aps) if len(val_aps) else None
best_val_auc = np.max(val_aucs) if len(val_aucs) else None

avg_epoch_time = np.mean(epoch_times) if len(epoch_times) else None
total_train_time = np.sum(total_epoch_times) if len(total_epoch_times) else None

print("\n===== TGN Aggregator Results =====")
print(f"Aggregator: {args.agg}")
print(f"Dataset: {args.data}")
print("----------------------------------")
print(f"Best Validation AP: {best_val_ap:.4f}")
print(f"Best Validation AUC: {best_val_auc:.4f}")
print(f"Test AP: {test_ap:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"New Node Test AP: {new_node_test_ap:.4f}")
print(f"New Node Test AUC: {new_node_test_auc:.4f}")
print("----------------------------------")
print(f"Average Epoch Time: {avg_epoch_time:.2f} s")
print(f"Total Training Time: {total_train_time:.2f} s")
print("==================================\n")