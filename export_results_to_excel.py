import argparse
import math
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # Colab usually has scipy, but keep a fallback.
    stats = None


DEFAULT_RESULTS_DIR = "/content/drive/MyDrive/tgn_results"
DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/results"
DEFAULT_CONFIDENCE = 0.95
DEFAULT_DATASETS = ["toy_markov", "toy_iid", "toy_mixed_noise", "toy_sequential"]
DEFAULT_PREFIXES = ["markov", "iid", "mixed_noise", "sequential"]
DEFAULT_AGGREGATORS = ["last", "mean", "weightedmean", "attention"]
DEFAULT_METRICS = [
    "test_ap",
    "test_auc",
    "test_acc",
    "test_prec",
    "test_rec",
    "test_f1",
    "test_mrr",
    "new_node_test_ap",
    "new_node_test_auc",
    "best_val_ap",
    "best_val_auc",
    "avg_epoch_time",
    "total_train_time",
]

RUN_FILE_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<dataset>toy_[^_]+(?:_[^_]+)*)_(?P<aggregator>last|mean|weightedmean|attention)(?:_(?P<run>\d+))?\.pkl$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export TGN run results from Google Drive to an Excel workbook with summary charts."
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Directory containing the .pkl result files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory where the Excel file and charts will be saved.")
    parser.add_argument("--output-name", default="tgn_aggregator_report.xlsx",
                        help="Name of the Excel workbook to create.")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                        help="Confidence level for confidence intervals, e.g. 0.95.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS,
                        help="Datasets to include.")
    parser.add_argument("--prefixes", nargs="*", default=DEFAULT_PREFIXES,
                        help="Prefixes corresponding to --datasets in the same order.")
    parser.add_argument("--aggregators", nargs="*", default=DEFAULT_AGGREGATORS,
                        help="Aggregators to include.")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS,
                        help="Metrics to evaluate in the workbook and charts.")
    parser.add_argument("--skip-charts", action="store_true",
                        help="Skip PNG chart generation.")
    return parser.parse_args()


def sanitize_sheet_name(name):
    invalid = set('[]:*?/\\')
    cleaned = "".join("_" if ch in invalid else ch for ch in name)
    return cleaned[:31]


def compute_t_interval(values, confidence):
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    n = len(clean)
    if n == 0:
        return {
            "n_runs": 0,
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_margin": np.nan,
        }

    arr = np.asarray(clean, dtype=float)
    mean = float(np.mean(arr))
    if n == 1:
        return {
            "n_runs": 1,
            "mean": mean,
            "std": np.nan,
            "sem": np.nan,
            "ci_low": mean,
            "ci_high": mean,
            "ci_margin": 0.0,
        }

    std = float(np.std(arr, ddof=1))
    sem = std / math.sqrt(n)
    if stats is not None:
        t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    else:
        t_crit = 1.96
    ci_margin = t_crit * sem
    return {
        "n_runs": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": mean - ci_margin,
        "ci_high": mean + ci_margin,
        "ci_margin": ci_margin,
    }


def load_run_file(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    val_aps = data.get("val_aps", []) or []
    val_aucs = data.get("val_aucs", []) or []
    total_epoch_times = data.get("total_epoch_times", []) or []

    row = {
        "test_ap": data.get("test_ap"),
        "test_auc": data.get("test_auc"),
        "test_acc": data.get("test_acc"),
        "test_prec": data.get("test_prec"),
        "test_rec": data.get("test_rec"),
        "test_f1": data.get("test_f1"),
        "test_mrr": data.get("test_mrr"),
        "new_node_test_ap": data.get("new_node_test_ap"),
        "new_node_test_auc": data.get("new_node_test_auc"),
        "best_val_ap": float(np.max(val_aps)) if len(val_aps) else np.nan,
        "best_val_auc": float(np.max(val_aucs)) if len(val_aucs) else np.nan,
        "avg_epoch_time": float(np.mean(total_epoch_times)) if len(total_epoch_times) else np.nan,
        "total_train_time": float(np.sum(total_epoch_times)) if len(total_epoch_times) else np.nan,
        "num_epochs_recorded": len(total_epoch_times),
        "source_file": path.name,
    }
    return row


def collect_raw_runs(results_dir, dataset_prefix_map, aggregators):
    rows = []

    for dataset, prefix in dataset_prefix_map.items():
        for aggregator in aggregators:
            run_zero_path = results_dir / f"{prefix}_{dataset}_{aggregator}.pkl"
            if run_zero_path.exists():
                row = load_run_file(run_zero_path)
                row.update({
                    "dataset": dataset,
                    "prefix": prefix,
                    "aggregator": aggregator,
                    "run": 0,
                })
                rows.append(row)

            run_idx = 1
            while True:
                run_path = results_dir / f"{prefix}_{dataset}_{aggregator}_{run_idx}.pkl"
                if not run_path.exists():
                    break
                row = load_run_file(run_path)
                row.update({
                    "dataset": dataset,
                    "prefix": prefix,
                    "aggregator": aggregator,
                    "run": run_idx,
                })
                rows.append(row)
                run_idx += 1

    if rows:
        return pd.DataFrame(rows).sort_values(["dataset", "aggregator", "run"]).reset_index(drop=True)

    # Fallback: scan the folder if the configured dataset/prefix pairs do not match all files.
    scanned_rows = []
    for path in sorted(results_dir.glob("*.pkl")):
        match = RUN_FILE_RE.match(path.name)
        if not match:
            continue
        aggregator = match.group("aggregator")
        if aggregator not in aggregators:
            continue
        row = load_run_file(path)
        row.update({
            "dataset": match.group("dataset"),
            "prefix": match.group("prefix"),
            "aggregator": aggregator,
            "run": int(match.group("run") or 0),
        })
        scanned_rows.append(row)

    if not scanned_rows:
        raise FileNotFoundError(f"No result .pkl files found in {results_dir}")

    return pd.DataFrame(scanned_rows).sort_values(["dataset", "aggregator", "run"]).reset_index(drop=True)


def build_summary_tables(raw_df, metrics, confidence):
    summary_rows = []
    metric_tables = {}

    grouped = raw_df.groupby(["dataset", "aggregator"], dropna=False)
    for (dataset, aggregator), group in grouped:
        for metric in metrics:
            values = group[metric].tolist() if metric in group.columns else []
            stats_row = compute_t_interval(values, confidence)
            summary_rows.append({
                "dataset": dataset,
                "aggregator": aggregator,
                "metric": metric,
                **stats_row,
            })

    summary_df = pd.DataFrame(summary_rows)

    for metric in metrics:
        metric_df = summary_df[summary_df["metric"] == metric].copy()
        metric_df = metric_df.sort_values(["dataset", "aggregator"]).reset_index(drop=True)
        metric_tables[metric] = metric_df

    return summary_df, metric_tables


def build_dataset_tables(summary_df):
    dataset_tables = {}
    for dataset, group in summary_df.groupby("dataset", dropna=False):
        pivot = group.pivot(index="metric", columns="aggregator", values="mean")
        pivot_std = group.pivot(index="metric", columns="aggregator", values="std")
        pivot_ci = group.pivot(index="metric", columns="aggregator", values="ci_margin")

        parts = []
        for aggregator in pivot.columns:
            part = pd.DataFrame({
                f"{aggregator}_mean": pivot[aggregator],
                f"{aggregator}_std": pivot_std[aggregator],
                f"{aggregator}_ci_margin": pivot_ci[aggregator],
            })
            parts.append(part)

        dataset_tables[dataset] = pd.concat(parts, axis=1).reset_index()

    return dataset_tables


def write_excel(raw_df, summary_df, metric_tables, dataset_tables, output_path, confidence):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="raw_runs", index=False)
        summary_df.to_excel(writer, sheet_name="summary_long", index=False)

        workbook_info = pd.DataFrame([
            {"field": "confidence_level", "value": confidence},
            {"field": "datasets", "value": ", ".join(sorted(raw_df["dataset"].dropna().unique()))},
            {"field": "aggregators", "value": ", ".join(sorted(raw_df["aggregator"].dropna().unique()))},
            {"field": "num_run_rows", "value": len(raw_df)},
        ])
        workbook_info.to_excel(writer, sheet_name="report_info", index=False)

        for dataset, dataset_df in dataset_tables.items():
            dataset_df.to_excel(writer, sheet_name=sanitize_sheet_name(f"dataset_{dataset}"), index=False)

        for metric, metric_df in metric_tables.items():
            metric_df.to_excel(writer, sheet_name=sanitize_sheet_name(f"metric_{metric}"), index=False)


def save_metric_chart(metric_df, metric, output_dir, confidence):
    if metric_df.empty:
        return None

    datasets = list(metric_df["dataset"].drop_duplicates())
    aggregators = list(metric_df["aggregator"].drop_duplicates())
    if not datasets or not aggregators:
        return None

    x = np.arange(len(datasets))
    width = 0.8 / max(len(aggregators), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, aggregator in enumerate(aggregators):
        agg_df = metric_df[metric_df["aggregator"] == aggregator].set_index("dataset").reindex(datasets)
        means = agg_df["mean"].to_numpy(dtype=float)
        errors = agg_df["ci_margin"].fillna(0.0).to_numpy(dtype=float)
        offsets = x + (idx - (len(aggregators) - 1) / 2.0) * width
        ax.bar(offsets, means, width=width, label=aggregator, yerr=errors, capsize=4)

    ax.set_title(f"{metric}: mean score by dataset and aggregator")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.legend(title="Aggregator")
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        0.99,
        0.01,
        f"Error bars: {int(confidence * 100)}% CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()

    chart_path = output_dir / f"{metric}_bar_chart.png"
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def main():
    args = parse_args()

    if len(args.datasets) != len(args.prefixes):
        raise ValueError("--datasets and --prefixes must have the same number of values.")

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_prefix_map = dict(zip(args.datasets, args.prefixes))
    raw_df = collect_raw_runs(results_dir, dataset_prefix_map, args.aggregators)
    summary_df, metric_tables = build_summary_tables(raw_df, args.metrics, args.confidence)
    dataset_tables = build_dataset_tables(summary_df)

    output_path = output_dir / args.output_name
    write_excel(raw_df, summary_df, metric_tables, dataset_tables, output_path, args.confidence)

    chart_paths = []
    if not args.skip_charts:
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        for metric, metric_df in metric_tables.items():
            chart_path = save_metric_chart(metric_df, metric, charts_dir, args.confidence)
            if chart_path is not None:
                chart_paths.append(chart_path)

    print(f"Excel workbook saved to: {output_path}")
    print(f"Total run rows exported: {len(raw_df)}")
    print(f"Datasets found: {', '.join(sorted(raw_df['dataset'].unique()))}")
    print(f"Aggregators found: {', '.join(sorted(raw_df['aggregator'].unique()))}")
    if chart_paths:
        print("Charts saved to:")
        for chart_path in chart_paths:
            print(f"  - {chart_path}")


if __name__ == "__main__":
    main()
