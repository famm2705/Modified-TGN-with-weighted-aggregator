import argparse
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.paths import get_reports_dir, get_results_dir

try:
    from scipy import stats
except ImportError:
    stats = None

DEFAULT_RESULTS_DIR = str(get_results_dir())
DEFAULT_OUTPUT_DIR = str(get_reports_dir())
DEFAULT_OUTPUT_NAME = "tgn_aggregator_report.xlsx"
DEFAULT_CONFIDENCE = 0.95
DEFAULT_EXPECTED_RUNS = 10

DEFAULT_DATASETS = [
    "toy_v3_last_event",
    "toy_v3_persistent_mean",
    "toy_v3_rare_spike",
    "toy_v3_ordered_pattern",
    "toy_last_event",
    "toy_persistent_mean",
    "toy_rare_spike",
    "toy_ordered_pattern",
]
DEFAULT_AGGREGATORS = ["last", "mean", "weightedmean", "attention"]

PRIMARY_METRICS = [
    "test_ap",
    "test_auc",
    "new_node_test_ap",
    "new_node_test_auc",
]
SECONDARY_METRICS = [
    "test_acc",
    "test_prec",
    "test_rec",
    "test_f1",
    "test_mrr",
    "best_val_ap",
    "best_val_auc",
    "final_val_ap",
    "final_val_auc",
    "final_train_loss",
    "min_train_loss",
    "avg_epoch_time",
    "total_train_time",
]
DEFAULT_METRICS = PRIMARY_METRICS + SECONDARY_METRICS


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build clean CSV and Excel summaries from TGN result pickle files. "
            "Defaults target the isolated toy aggregator datasets."
        )
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Directory containing .pkl files produced by train_self_supervised.py.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for CSV, Excel, and optional PNG chart outputs.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME,
                        help="Excel workbook file name.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS,
                        help="Dataset names to include. Used to parse filenames reliably.")
    parser.add_argument("--aggregators", nargs="*", default=DEFAULT_AGGREGATORS,
                        help="Aggregator names to include.")
    parser.add_argument("--prefixes", nargs="*", default=None,
                        help="Optional prefix filter. Leave unset to accept any training prefix.")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS,
                        help="Metrics to summarize.")
    parser.add_argument("--chart-metrics", nargs="*", default=PRIMARY_METRICS,
                        help="Metrics to make chart-ready Excel sheets and PNG charts for.")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                        help="Confidence interval level, for example 0.95.")
    parser.add_argument("--expected-runs", type=int, default=DEFAULT_EXPECTED_RUNS,
                        help="Expected number of runs per dataset/aggregator pair.")
    parser.add_argument("--include-incomplete", action="store_true",
                        help="Include partially written pickle files that do not have test metrics yet.")
    parser.add_argument("--skip-png-charts", action="store_true",
                        help="Skip PNG bar charts. Excel and CSV tables are still written.")
    return parser.parse_args()


def sanitize_sheet_name(name):
    invalid = set("[]:*?/\\")
    cleaned = "".join("_" if ch in invalid else ch for ch in name)
    return cleaned[:31]


def scalar_or_nan(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_max(values):
    clean = [scalar_or_nan(v) for v in values]
    clean = [v for v in clean if not np.isnan(v)]
    return float(np.max(clean)) if clean else np.nan


def safe_min(values):
    clean = [scalar_or_nan(v) for v in values]
    clean = [v for v in clean if not np.isnan(v)]
    return float(np.min(clean)) if clean else np.nan


def safe_last(values):
    clean = [scalar_or_nan(v) for v in values]
    clean = [v for v in clean if not np.isnan(v)]
    return float(clean[-1]) if clean else np.nan


def parse_result_filename(path, datasets, aggregators):
    aggregator_pattern = "|".join(re.escape(a) for a in sorted(aggregators, key=len, reverse=True))
    match = re.match(
        rf"^(?P<body>.+)_(?P<aggregator>{aggregator_pattern})(?:_(?P<run>\d+))?$",
        path.stem,
    )
    if not match:
        return None

    body = match.group("body")
    dataset_candidates = [
        dataset for dataset in datasets
        if body == dataset or body.endswith(f"_{dataset}")
    ]
    if not dataset_candidates:
        return None

    dataset = max(dataset_candidates, key=len)
    prefix = body[:-len(dataset)]
    if prefix.endswith("_"):
        prefix = prefix[:-1]
    if prefix == "_":
        prefix = ""

    return {
        "dataset": dataset,
        "prefix": prefix,
        "aggregator": match.group("aggregator"),
        "run": int(match.group("run") or 0),
    }


def load_run_file(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    val_aps = data.get("val_aps", []) or []
    val_aucs = data.get("val_aucs", []) or []
    train_losses = data.get("train_losses", []) or []
    total_epoch_times = data.get("total_epoch_times", []) or []

    row = {
        "test_ap": scalar_or_nan(data.get("test_ap")),
        "test_auc": scalar_or_nan(data.get("test_auc")),
        "test_acc": scalar_or_nan(data.get("test_acc")),
        "test_prec": scalar_or_nan(data.get("test_prec")),
        "test_rec": scalar_or_nan(data.get("test_rec")),
        "test_f1": scalar_or_nan(data.get("test_f1")),
        "test_mrr": scalar_or_nan(data.get("test_mrr")),
        "new_node_test_ap": scalar_or_nan(data.get("new_node_test_ap")),
        "new_node_test_auc": scalar_or_nan(data.get("new_node_test_auc")),
        "best_val_ap": safe_max(val_aps),
        "best_val_auc": safe_max(val_aucs),
        "final_val_ap": safe_last(val_aps),
        "final_val_auc": safe_last(val_aucs),
        "final_train_loss": safe_last(train_losses),
        "min_train_loss": safe_min(train_losses),
        "avg_epoch_time": float(np.mean(total_epoch_times)) if len(total_epoch_times) else np.nan,
        "total_train_time": float(np.sum(total_epoch_times)) if len(total_epoch_times) else np.nan,
        "num_epochs_recorded": len(total_epoch_times),
        "source_file": path.name,
    }
    row["is_complete"] = not np.isnan(row["test_ap"]) and not np.isnan(row["test_auc"])
    return row


def collect_raw_runs(results_dir, datasets, aggregators, prefixes=None, include_incomplete=False):
    rows = []
    prefixes = set(prefixes or [])

    for path in sorted(results_dir.glob("*.pkl")):
        parsed = parse_result_filename(path, datasets=datasets, aggregators=aggregators)
        if parsed is None:
            continue
        if prefixes and parsed["prefix"] not in prefixes:
            continue

        row = load_run_file(path)
        if not include_incomplete and not row["is_complete"]:
            continue
        row.update(parsed)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No matching result .pkl files found in {results_dir}. "
            "Check --datasets, --aggregators, and --prefixes."
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "aggregator", "run", "source_file"])
        .reset_index(drop=True)
    )


def compute_interval(values, confidence):
    clean = [scalar_or_nan(v) for v in values]
    clean = [v for v in clean if not np.isnan(v)]
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
        critical = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    else:
        critical = 1.96
    ci_margin = critical * sem
    return {
        "n_runs": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": mean - ci_margin,
        "ci_high": mean + ci_margin,
        "ci_margin": ci_margin,
    }


def build_summary(raw_df, metrics, confidence):
    summary_rows = []
    for (dataset, aggregator), group in raw_df.groupby(["dataset", "aggregator"], dropna=False):
        prefixes = ", ".join(sorted(str(p) for p in group["prefix"].dropna().unique()))
        source_files = len(group["source_file"].unique())
        for metric in metrics:
            values = group[metric].tolist() if metric in group.columns else []
            summary_rows.append({
                "dataset": dataset,
                "aggregator": aggregator,
                "metric": metric,
                "prefixes": prefixes,
                "source_files": source_files,
                **compute_interval(values, confidence),
            })

    summary_long = pd.DataFrame(summary_rows)

    flat = {}
    for (dataset, aggregator), group in summary_long.groupby(["dataset", "aggregator"], dropna=False):
        row = {"dataset": dataset, "aggregator": aggregator}
        for _, metric_row in group.iterrows():
            metric = metric_row["metric"]
            for stat in ["n_runs", "mean", "std", "sem", "ci_low", "ci_high", "ci_margin"]:
                row[f"{metric}_{stat}"] = metric_row[stat]
        flat[(dataset, aggregator)] = row

    summary_wide = pd.DataFrame(flat.values()).sort_values(["dataset", "aggregator"]).reset_index(drop=True)
    return summary_long, summary_wide


def build_completion_table(summary_long, expected_runs):
    test_ap = summary_long[summary_long["metric"] == "test_ap"].copy()
    if test_ap.empty:
        return pd.DataFrame()
    completion = test_ap[["dataset", "aggregator", "n_runs", "source_files"]].copy()
    completion["expected_runs"] = expected_runs
    completion["missing_runs"] = expected_runs - completion["n_runs"]
    completion["complete"] = completion["missing_runs"] <= 0
    return completion.sort_values(["dataset", "aggregator"]).reset_index(drop=True)


def build_plot_tables(summary_long, chart_metrics):
    tables = {}
    for metric in chart_metrics:
        metric_df = summary_long[summary_long["metric"] == metric].copy()
        if metric_df.empty:
            continue
        tables[metric] = {
            "mean": metric_df.pivot(index="dataset", columns="aggregator", values="mean"),
            "std": metric_df.pivot(index="dataset", columns="aggregator", values="std"),
            "ci_margin": metric_df.pivot(index="dataset", columns="aggregator", values="ci_margin"),
            "n_runs": metric_df.pivot(index="dataset", columns="aggregator", values="n_runs"),
        }
    return tables


def write_plot_sheet(writer, metric, tables):
    sheet_name = sanitize_sheet_name(f"plot_{metric}")
    row = 0
    for label, table in tables.items():
        title_df = pd.DataFrame({f"{metric}_{label}": []})
        title_df.to_excel(writer, sheet_name=sheet_name, startrow=row, index=False)
        row += 1
        table.reset_index().to_excel(writer, sheet_name=sheet_name, startrow=row, index=False)
        row += len(table) + 4


def write_outputs(raw_df, summary_long, summary_wide, completion, plot_tables, output_dir,
                  output_name, confidence, expected_runs):
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = output_dir / "raw_runs.csv"
    summary_long_csv = output_dir / "summary_long.csv"
    summary_wide_csv = output_dir / "summary_wide.csv"
    completion_csv = output_dir / "run_completion.csv"

    raw_df.to_csv(raw_csv, index=False)
    summary_long.to_csv(summary_long_csv, index=False)
    summary_wide.to_csv(summary_wide_csv, index=False)
    completion.to_csv(completion_csv, index=False)

    workbook_path = output_dir / output_name
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        info = pd.DataFrame([
            {"field": "confidence_level", "value": confidence},
            {"field": "expected_runs", "value": expected_runs},
            {"field": "num_raw_rows", "value": len(raw_df)},
            {"field": "datasets", "value": ", ".join(sorted(raw_df["dataset"].unique()))},
            {"field": "aggregators", "value": ", ".join(sorted(raw_df["aggregator"].unique()))},
            {"field": "primary_metrics", "value": ", ".join(PRIMARY_METRICS)},
        ])
        info.to_excel(writer, sheet_name="report_info", index=False)
        completion.to_excel(writer, sheet_name="run_completion", index=False)
        raw_df.to_excel(writer, sheet_name="raw_runs", index=False)
        summary_long.to_excel(writer, sheet_name="summary_long", index=False)
        summary_wide.to_excel(writer, sheet_name="summary_wide", index=False)
        for metric, tables in plot_tables.items():
            write_plot_sheet(writer, metric, tables)

    return {
        "workbook": workbook_path,
        "raw_csv": raw_csv,
        "summary_long_csv": summary_long_csv,
        "summary_wide_csv": summary_wide_csv,
        "completion_csv": completion_csv,
    }


def save_png_charts(plot_tables, output_dir, confidence):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = []

    for metric, tables in plot_tables.items():
        means = tables["mean"]
        errors = tables["ci_margin"].reindex(index=means.index, columns=means.columns).fillna(0.0)
        if means.empty:
            continue

        datasets = list(means.index)
        aggregators = list(means.columns)
        x = np.arange(len(datasets))
        width = 0.8 / max(len(aggregators), 1)

        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, aggregator in enumerate(aggregators):
            offsets = x + (idx - (len(aggregators) - 1) / 2.0) * width
            ax.bar(
                offsets,
                means[aggregator].to_numpy(dtype=float),
                width=width,
                label=aggregator,
                yerr=errors[aggregator].to_numpy(dtype=float),
                capsize=4,
            )

        ax.set_title(f"{metric}: mean by dataset and aggregator")
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

        chart_path = charts_dir / f"{metric}_bar_chart.png"
        fig.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append(chart_path)

    return chart_paths


def print_compact_summary(summary_long, completion, paths, chart_paths):
    print("\nRun completion")
    if completion.empty:
        print("  No completion table available.")
    else:
        print(completion.to_string(index=False))

    print("\nPrimary metric summary")
    primary = summary_long[summary_long["metric"].isin(PRIMARY_METRICS)].copy()
    primary = primary[["dataset", "aggregator", "metric", "n_runs", "mean", "std", "ci_margin"]]
    with pd.option_context("display.max_rows", 200, "display.width", 140):
        print(primary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nSaved outputs")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    if chart_paths:
        print("  png_charts:")
        for chart_path in chart_paths:
            print(f"    {chart_path}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    raw_df = collect_raw_runs(
        results_dir=results_dir,
        datasets=args.datasets,
        aggregators=args.aggregators,
        prefixes=args.prefixes,
        include_incomplete=args.include_incomplete,
    )
    summary_long, summary_wide = build_summary(raw_df, args.metrics, args.confidence)
    completion = build_completion_table(summary_long, args.expected_runs)
    plot_tables = build_plot_tables(summary_long, args.chart_metrics)
    paths = write_outputs(
        raw_df=raw_df,
        summary_long=summary_long,
        summary_wide=summary_wide,
        completion=completion,
        plot_tables=plot_tables,
        output_dir=output_dir,
        output_name=args.output_name,
        confidence=args.confidence,
        expected_runs=args.expected_runs,
    )
    chart_paths = [] if args.skip_png_charts else save_png_charts(plot_tables, output_dir, args.confidence)
    print_compact_summary(summary_long, completion, paths, chart_paths)


if __name__ == "__main__":
    main()
