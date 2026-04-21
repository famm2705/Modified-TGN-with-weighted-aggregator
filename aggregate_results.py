import pickle
import numpy as np
from pathlib import Path
from scipy import stats
import csv

BASE_PATH = "/content/drive/MyDrive/tgn_results"

DATASETS    = ["toy_markov", "toy_iid", "toy_mixed_noise", "toy_sequential"]
PREFIXES    = ["markov",     "iid",     "mixed_noise",      "sequential"]
AGGREGATORS = ["last", "mean", "weightedmean", "attention"]

METRICS = ["test_ap", "test_auc", "test_acc", "test_prec", "test_rec", "test_f1", "test_mrr",
           "new_node_test_ap", "new_node_test_auc"]

CONFIDENCE = 0.95


def load_runs(prefix, dataset, aggregator):
    """
    Load all run pickle files for a given combination.
    Run 0  → {prefix}_{dataset}_{aggregator}.pkl
    Run i  → {prefix}_{dataset}_{aggregator}_{i}.pkl
    """
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


def summarise(values):
    """
    Return (mean, std, ci_low, ci_high) for a list of scalar values.
    Uses a two-sided t-distribution 95% CI.
    Returns (None, None, None, None) if fewer than 2 values.
    """
    arr = np.array([v for v in values if v is not None], dtype=float)
    n = len(arr)
    if n == 0:
        return None, None, None, None
    mean = float(np.mean(arr))
    if n == 1:
        return round(mean, 4), None, None, None
    std  = float(np.std(arr, ddof=1))
    se   = std / np.sqrt(n)
    t_crit = stats.t.ppf((1 + CONFIDENCE) / 2, df=n - 1)
    margin = t_crit * se
    return (
        round(mean, 4),
        round(std,  4),
        round(mean - margin, 4),
        round(mean + margin, 4),
    )


def fmt(mean, std, ci_lo, ci_hi):
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.4f} (n=1)"
    return f"{mean:.4f} ± {std:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]"


def fmt_ap(mean, std, ci_lo, ci_hi):
    """Compact format for tables."""
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


# ── Collect all results ──────────────────────────────────────────────────────

results = {}

for (dataset, prefix) in zip(DATASETS, PREFIXES):
    results[dataset] = {}
    for agg in AGGREGATORS:
        runs = load_runs(prefix, dataset, agg)
        if not runs:
            print(f"  WARNING: no runs found for {prefix}/{dataset}/{agg}")
            results[dataset][agg] = {"n_runs": 0}
            continue

        n_runs = len(runs)
        entry  = {"n_runs": n_runs}

        for metric in METRICS:
            values = [r[metric] for r in runs if metric in r and r[metric] is not None]
            entry[metric] = summarise(values)

        times = [np.sum(r.get("total_epoch_times", [])) for r in runs]
        entry["total_train_time"] = summarise([t for t in times if t > 0])

        results[dataset][agg] = entry
        print(f"  Loaded {n_runs} run(s) — {prefix}/{dataset}/{agg}")


# ── Print full results ───────────────────────────────────────────────────────

print("\n" + "="*80)
print(f"FULL RESULTS  —  mean ± std  [{CONFIDENCE*100:.0f}% CI lower, upper]")
print("="*80)

for dataset in DATASETS:
    print(f"\n── {dataset} ──")
    print(f"  {'Aggregator':<14} {'Runs':>4}  {'Test AP':<38} {'Test AUC':<38} {'New node AP'}")
    print(f"  {'-'*13} {'-'*4}  {'-'*37} {'-'*37} {'-'*37}")
    for agg in AGGREGATORS:
        e = results[dataset][agg]
        if e["n_runs"] == 0:
            print(f"  {agg:<14} {'0':>4}  N/A")
            continue
        ap  = fmt(*e.get("test_ap",          (None,)*4))
        auc = fmt(*e.get("test_auc",         (None,)*4))
        nn  = fmt(*e.get("new_node_test_ap", (None,)*4))
        print(f"  {agg:<14} {e['n_runs']:>4}  {ap:<38} {auc:<38} {nn}")


# ── Compact AP table ─────────────────────────────────────────────────────────

print("\n" + "="*80)
print("TEST AP  mean ± std  (compact)")
print("="*80)
col = 22
header = f"{'Dataset':<22}" + "".join(f"{a:<24}" for a in AGGREGATORS)
print(header)
print("-"*80)
for dataset in DATASETS:
    row = f"{dataset:<22}"
    for agg in AGGREGATORS:
        e = results[dataset][agg]
        if e["n_runs"] == 0:
            row += f"{'N/A':<24}"
        else:
            row += f"{fmt_ap(*e.get('test_ap', (None,)*4)):<24}"
    print(row)


# ── Winner per dataset ───────────────────────────────────────────────────────

print("\n" + "="*80)
print("WINNER per dataset  (highest mean test AP)")
print("="*80)
for dataset in DATASETS:
    best_agg = None
    best_ap  = -1
    for agg in AGGREGATORS:
        e = results[dataset][agg]
        if e["n_runs"] == 0:
            continue
        m = e.get("test_ap", (None,))[0]
        if m is not None and m > best_ap:
            best_ap  = m
            best_agg = agg
    if best_agg:
        m, s, lo, hi = results[dataset][best_agg].get("test_ap", (None,)*4)
        ci_str = f"  95% CI [{lo:.4f}, {hi:.4f}]" if lo is not None else ""
        print(f"  {dataset:<25} → {best_agg:<14} AP = {m:.4f} ± {s:.4f if s else 0:.4f}{ci_str}")


# ── Overlap check ────────────────────────────────────────────────────────────
# If the CI of the winner overlaps with the CI of the second-best,
# the difference is not statistically reliable.

print("\n" + "="*80)
print("CI OVERLAP CHECK  (does the winner's CI overlap with runner-up?)")
print("="*80)
for dataset in DATASETS:
    ranked = []
    for agg in AGGREGATORS:
        e = results[dataset][agg]
        if e["n_runs"] == 0:
            continue
        m, s, lo, hi = e.get("test_ap", (None,)*4)
        if m is not None:
            ranked.append((m, lo, hi, agg))
    ranked.sort(reverse=True)
    if len(ranked) < 2:
        continue
    w_m, w_lo, w_hi, w_agg   = ranked[0]
    r_m, r_lo, r_hi, r_agg   = ranked[1]
    if w_lo is None or r_hi is None:
        overlap = "cannot determine (n=1)"
    elif w_lo <= r_hi:
        overlap = f"YES — CIs overlap  ({w_agg} [{w_lo:.4f},{w_hi:.4f}] vs {r_agg} [{r_lo:.4f},{r_hi:.4f}])"
    else:
        overlap = f"NO  — clear separation  ({w_agg} [{w_lo:.4f},{w_hi:.4f}] vs {r_agg} [{r_lo:.4f},{r_hi:.4f}])"
    print(f"  {dataset:<25} {overlap}")


# ── Save CSV ─────────────────────────────────────────────────────────────────

csv_path = Path(BASE_PATH) / "aggregator_summary.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "dataset", "aggregator", "n_runs",
        "test_ap_mean",  "test_ap_std",  "test_ap_ci_low",  "test_ap_ci_high",
        "test_auc_mean", "test_auc_std", "test_auc_ci_low", "test_auc_ci_high",
        "test_acc_mean", "test_acc_std", "test_acc_ci_low", "test_acc_ci_high",
        "test_f1_mean",  "test_f1_std",  "test_f1_ci_low",  "test_f1_ci_high",
        "test_mrr_mean", "test_mrr_std", "test_mrr_ci_low", "test_mrr_ci_high",
        "new_node_ap_mean",  "new_node_ap_std",  "new_node_ap_ci_low",  "new_node_ap_ci_high",
        "new_node_auc_mean", "new_node_auc_std", "new_node_auc_ci_low", "new_node_auc_ci_high",
        "total_train_time_mean", "total_train_time_std",
    ])

    for (dataset, prefix) in zip(DATASETS, PREFIXES):
        for agg in AGGREGATORS:
            e = results[dataset][agg]
            if e["n_runs"] == 0:
                continue

            def g(key):
                return e.get(key, (None, None, None, None))

            ap  = g("test_ap")
            auc = g("test_auc")
            acc = g("test_acc")
            f1  = g("test_f1")
            mrr = g("test_mrr")
            nn  = g("new_node_test_ap")
            na  = g("new_node_test_auc")
            tt  = g("total_train_time")

            writer.writerow([
                dataset, agg, e["n_runs"],
                *ap[:4], *auc[:4], *acc[:4], *f1[:4], *mrr[:4],
                *nn[:4], *na[:4],
                tt[0], tt[1],
            ])

print(f"\nCSV saved → {csv_path}")
print("Done.")