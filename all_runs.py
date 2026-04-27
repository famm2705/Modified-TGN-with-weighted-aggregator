import pickle
import numpy as np
from pathlib import Path
import csv

BASE_PATH = "/content/drive/MyDrive/tgn_results"

DATASETS    = ["toy_markov", "toy_iid", "toy_mixed_noise", "toy_sequential"]
PREFIXES    = ["markov",     "iid",     "mixed_noise",      "sequential"]
AGGREGATORS = ["last", "mean", "weightedmean", "attention"]


def load_runs(prefix, dataset, aggregator):
    runs = []
    p0 = Path(BASE_PATH) / f"{prefix}_{dataset}_{aggregator}.pkl"
    if p0.exists():
        with open(p0, "rb") as f:
            runs.append(pickle.load(f))
    i = 1
    while True:
        pi = Path(BASE_PATH) / f"{prefix}_{dataset}_{aggregator}_{i}.pkl"
        if not pi.exists():
            break
        with open(pi, "rb") as f:
            runs.append(pickle.load(f))
        i += 1
    return runs


def r(val):
    """Round to 4 decimal places or return empty string."""
    return round(float(val), 4) if val is not None else ""


# ── Collect all rows ─────────────────────────────────────────────────────────

rows = []

for (dataset, prefix) in zip(DATASETS, PREFIXES):
    for agg in AGGREGATORS:
        runs = load_runs(prefix, dataset, agg)
        if not runs:
            print(f"WARNING: no runs found for {dataset}/{agg}")
            continue
        for run_idx, run in enumerate(runs):
            rows.append({
                "dataset":          dataset,
                "aggregator":       agg,
                "run":              run_idx,
                "test_ap":          r(run.get("test_ap")),
                "test_auc":         r(run.get("test_auc")),
                "test_acc":         r(run.get("test_acc")),
                "test_prec":        r(run.get("test_prec")),
                "test_rec":         r(run.get("test_rec")),
                "test_f1":          r(run.get("test_f1")),
                "test_mrr":         r(run.get("test_mrr")),
                "new_node_test_ap":  r(run.get("new_node_test_ap")),
                "new_node_test_auc": r(run.get("new_node_test_auc")),
                "total_train_time":  r(np.sum(run.get("total_epoch_times", [0]))),
            })


# ── Print table ───────────────────────────────────────────────────────────────

COLS = ["dataset", "aggregator", "run",
        "test_ap", "test_auc", "test_acc",
        "test_prec", "test_rec", "test_f1", "test_mrr",
        "new_node_test_ap", "new_node_test_auc", "total_train_time"]

W = [22, 14, 4, 9, 9, 9, 9, 9, 9, 9, 16, 16, 18]

def pad(val, width):
    return str(val).ljust(width)

header = "".join(pad(c, W[i]) for i, c in enumerate(COLS))
divider = "-" * sum(W)

print("\n" + divider)
print(header)
print(divider)

prev_dataset = None
for row in rows:
    if row["dataset"] != prev_dataset:
        if prev_dataset is not None:
            print(divider)
        prev_dataset = row["dataset"]
    line = "".join(pad(row[c], W[i]) for i, c in enumerate(COLS))
    print(line)

print(divider)


# ── Save CSV ──────────────────────────────────────────────────────────────────

csv_path = Path(BASE_PATH) / "all_runs_flat.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLS)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCSV saved → {csv_path}")
print(f"Total rows: {len(rows)}")
print("Done.")