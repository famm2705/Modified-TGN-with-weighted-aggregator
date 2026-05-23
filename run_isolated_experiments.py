import argparse
import json
import math
import pickle
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from utils.paths import get_data_dir, get_project_root


DATASETS = [
  "toy_last_event",
  "toy_persistent_mean",
  "toy_rare_spike",
  "toy_ordered_pattern",
]

AGGREGATORS = ["last", "mean", "weightedmean", "attention"]

INTENDED_PAIRS = [
  ("toy_last_event", "last"),
  ("toy_persistent_mean", "mean"),
  ("toy_rare_spike", "weightedmean"),
  ("toy_ordered_pattern", "attention"),
]


def default_output_base():
  drive_root = Path("/content/drive/MyDrive")
  if drive_root.exists():
    return drive_root / "tgn_experiment_runs"
  return get_project_root() / "outputs" / "experiment_runs"


def default_runtime_checkpoint_base():
  return Path("/content/tgn_experiment_checkpoints")


def is_colab_runtime():
  return Path("/content").exists()


def parse_args():
  parser = argparse.ArgumentParser(
    description=(
      "Run the current isolated TGN aggregator experiments. The default is the full "
      "4 dataset x 4 aggregator x 10 run matrix for self-supervised encoders, then "
      "the matching supervised edge-label runs."
    )
  )
  parser.add_argument("--experiment-name", default=None,
                      help="Folder name for this experiment. Defaults to a timestamped name.")
  parser.add_argument("--output-base", default=None,
                      help="Parent directory for experiment folders.")
  parser.add_argument("--checkpoint-base", default=None,
                      help=(
                        "Parent directory for per-experiment early-stopping checkpoints. "
                        "Defaults to /content/tgn_experiment_checkpoints on Colab and to "
                        "the experiment models folder elsewhere."
                      ))
  parser.add_argument("--checkpoints-on-runtime", action="store_true",
                      help=(
                        "Store temporary per-epoch checkpoints under "
                        "/content/tgn_experiment_checkpoints/<experiment-name>/checkpoints. "
                        "Final models, results, reports, and logs still use the experiment folder."
                      ))
  parser.add_argument("--checkpoints-with-models", action="store_true",
                      help=(
                        "Store per-epoch checkpoints in the experiment models folder. "
                        "This is not recommended on Colab Drive."
                      ))
  parser.add_argument("--data-dir", default=None,
                      help="Directory containing or receiving preprocessed dataset files.")
  parser.add_argument("--datasets", nargs="*", default=DATASETS,
                      help="Datasets to run. Defaults to the current isolated toy datasets.")
  parser.add_argument("--aggregators", nargs="*", default=AGGREGATORS,
                      help="Aggregators to run.")
  parser.add_argument("--pairs-only", action="store_true",
                      help="Run only the intended one-to-one dataset/aggregator pairs.")
  parser.add_argument("--n-runs", type=int, default=10,
                      help="Independent runs per dataset/aggregator pair.")
  parser.add_argument("--self-epochs", type=int, default=50,
                      help="Maximum self-supervised epochs per run.")
  parser.add_argument("--supervised-epochs", type=int, default=50,
                      help="Maximum supervised epochs per run.")
  parser.add_argument("--patience", type=int, default=5,
                      help="Early stopping patience.")
  parser.add_argument("--batch-size", type=int, default=128,
                      help="Batch size. The isolated datasets are designed around 128.")
  parser.add_argument("--n-degree", type=int, default=10,
                      help="Number of temporal neighbors sampled by TGN.")
  parser.add_argument("--gpu", type=int, default=0,
                      help="CUDA device index used by the training scripts.")
  parser.add_argument("--python", default=sys.executable,
                      help="Python executable used to launch subprocesses.")
  parser.add_argument("--num-trials", type=int, default=32,
                      help="Number of two-batch trials when generating isolated datasets.")
  parser.add_argument("--skip-data-generation", action="store_true",
                      help="Do not generate missing isolated datasets.")
  parser.add_argument("--regenerate-data", action="store_true",
                      help="Regenerate isolated datasets before training.")
  parser.add_argument("--skip-self-supervised", action="store_true",
                      help="Skip self-supervised encoder training.")
  parser.add_argument("--skip-supervised", action="store_true",
                      help="Skip supervised edge-label training.")
  parser.add_argument("--skip-reports", action="store_true",
                      help="Skip Excel/CSV report generation.")
  parser.add_argument("--no-supervised-validation", action="store_true",
                      help="Do not reserve validation data for supervised early stopping.")
  parser.add_argument("--no-resume", dest="resume", action="store_false",
                      help="Rerun completed files instead of skipping them.")
  parser.add_argument("--dry-run", action="store_true",
                      help="Print commands without executing them.")
  parser.add_argument("--continue-on-error", action="store_true",
                      help="Continue to the next run if one subprocess fails.")
  parser.set_defaults(resume=True)
  return parser.parse_args()


def experiment_pairs(args):
  if args.pairs_only:
    selected_datasets = set(args.datasets)
    selected_aggregators = set(args.aggregators)
    return [
      (dataset, aggregator)
      for dataset, aggregator in INTENDED_PAIRS
      if dataset in selected_datasets and aggregator in selected_aggregators
    ]

  return [
    (dataset, aggregator)
    for dataset in args.datasets
    for aggregator in args.aggregators
  ]


def preprocessed_files(data_dir, dataset):
  return [
    data_dir / f"ml_{dataset}.csv",
    data_dir / f"ml_{dataset}.npy",
    data_dir / f"ml_{dataset}_node.npy",
  ]


def missing_datasets(data_dir, datasets):
  missing = []
  for dataset in datasets:
    if any(not path.exists() for path in preprocessed_files(data_dir, dataset)):
      missing.append(dataset)
  return missing


def run_command(cmd, cwd, command_log, dry_run=False, continue_on_error=False):
  rendered = shlex.join(str(part) for part in cmd)
  print(f"\n$ {rendered}")
  with command_log.open("a", encoding="utf-8") as handle:
    handle.write(rendered + "\n")

  if dry_run:
    return True

  completed = subprocess.run([str(part) for part in cmd], cwd=str(cwd))
  if completed.returncode == 0:
    return True

  message = f"Command failed with exit code {completed.returncode}: {rendered}"
  if continue_on_error:
    print(message)
    return False
  raise RuntimeError(message)


def result_complete(path, required_keys):
  if not path.exists():
    return False
  try:
    with path.open("rb") as handle:
      result = pickle.load(handle)
  except Exception:
    return False
  if not isinstance(result, dict):
    return False

  for key in required_keys:
    value = result.get(key)
    if value is None:
      return False
    try:
      if math.isnan(float(value)):
        return False
    except (TypeError, ValueError):
      pass
  return True


def run_prefixes(n_runs):
  return [f"run{run_idx:02d}" for run_idx in range(n_runs)]


def run_specs(args, pairs):
  for dataset, aggregator in pairs:
    for prefix in run_prefixes(args.n_runs):
      yield dataset, aggregator, prefix


def write_manifest(path, args, exp_root, data_dir, models_dir, checkpoints_dir,
                   results_dir, reports_dir, logs_dir, pairs):
  manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "experiment_root": str(exp_root),
    "data_dir": str(data_dir),
    "models_dir": str(models_dir),
    "checkpoints_dir": str(checkpoints_dir),
    "results_dir": str(results_dir),
    "reports_dir": str(reports_dir),
    "logs_dir": str(logs_dir),
    "datasets": args.datasets,
    "aggregators": args.aggregators,
    "pairs": [{"dataset": dataset, "aggregator": aggregator} for dataset, aggregator in pairs],
    "n_runs_per_pair": args.n_runs,
    "self_epochs": args.self_epochs,
    "supervised_epochs": args.supervised_epochs,
    "patience": args.patience,
    "batch_size": args.batch_size,
    "n_degree": args.n_degree,
    "use_memory": True,
    "learnable": True,
    "add_cls_token": True,
    "phase_order": ["self_supervised", "supervised", "reports"],
  }
  path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def maybe_generate_data(args, data_dir, project_root, command_log):
  missing = missing_datasets(data_dir, args.datasets)
  should_generate = args.regenerate_data or bool(missing)

  if not should_generate:
    return

  if args.skip_data_generation:
    missing_text = ", ".join(missing) if missing else "requested regeneration"
    raise FileNotFoundError(
      f"Dataset generation is disabled, but data is missing or stale: {missing_text}"
    )

  data_dir.mkdir(parents=True, exist_ok=True)
  generated = run_command(
    [
      args.python,
      project_root / "utils" / "toydatasets_generator.py",
      "--output-dir",
      data_dir,
      "--num-trials",
      str(args.num_trials),
    ],
    cwd=project_root,
    command_log=command_log,
    dry_run=args.dry_run,
    continue_on_error=args.continue_on_error,
  )
  if not generated or args.dry_run:
    return

  still_missing = missing_datasets(data_dir, args.datasets)
  if still_missing:
    missing_text = ", ".join(still_missing)
    message = f"Dataset generation finished, but files are still missing: {missing_text}"
    if args.continue_on_error:
      print(message)
      return
    raise FileNotFoundError(message)


def self_supervised_command(args, project_root, data_dir, models_dir, checkpoints_dir,
                            results_dir, logs_dir, dataset, aggregator, prefix):
  return [
    args.python,
    project_root / "train_self_supervised.py",
    "--data",
    dataset,
    "--aggregator",
    aggregator,
    "--prefix",
    prefix,
    "--n_runs",
    "1",
    "--n_epoch",
    str(args.self_epochs),
    "--patience",
    str(args.patience),
    "--bs",
    str(args.batch_size),
    "--n_degree",
    str(args.n_degree),
    "--gpu",
    str(args.gpu),
    "--use_memory",
    "--learnable",
    "--add_cls_token",
    "--data-dir",
    data_dir,
    "--model-dir",
    models_dir,
    "--checkpoint-dir",
    checkpoints_dir,
    "--results-dir",
    results_dir,
    "--log-dir",
    logs_dir,
  ]


def supervised_command(args, project_root, data_dir, models_dir, checkpoints_dir,
                       results_dir, logs_dir, dataset, aggregator, prefix):
  cmd = [
    args.python,
    project_root / "train_supervised.py",
    "--data",
    dataset,
    "--aggregator",
    aggregator,
    "--prefix",
    prefix,
    "--n_runs",
    "1",
    "--n_epoch",
    str(args.supervised_epochs),
    "--patience",
    str(args.patience),
    "--bs",
    str(args.batch_size),
    "--n_degree",
    str(args.n_degree),
    "--gpu",
    str(args.gpu),
    "--use_memory",
    "--learnable",
    "--add_cls_token",
    "--data-dir",
    data_dir,
    "--model-dir",
    models_dir,
    "--checkpoint-dir",
    checkpoints_dir,
    "--results-dir",
    results_dir,
    "--log-dir",
    logs_dir,
  ]
  if not args.no_supervised_validation:
    cmd.append("--use_validation")
  return cmd


def self_supervised_paths(models_dir, results_dir, dataset, aggregator, prefix):
  return (
    models_dir / f"{prefix}_{dataset}_{aggregator}.pth",
    results_dir / f"{prefix}_{dataset}_{aggregator}.pkl",
  )


def supervised_paths(models_dir, results_dir, dataset, aggregator, prefix):
  return (
    models_dir / f"supervised_{prefix}_{dataset}_{aggregator}_edge_label_decoder.pth",
    results_dir / f"supervised_{prefix}_{dataset}_{aggregator}_edge_label_classification.pkl",
  )


def run_self_supervised_phase(args, project_root, data_dir, models_dir, checkpoints_dir,
                              results_dir, logs_dir, command_log, pairs):
  if args.skip_self_supervised:
    return

  print("\nStarting self-supervised phase")
  for dataset, aggregator, prefix in run_specs(args, pairs):
    self_model, self_result = self_supervised_paths(
      models_dir, results_dir, dataset, aggregator, prefix,
    )
    if args.resume and self_model.exists() and result_complete(self_result, ["test_ap", "test_auc"]):
      print(f"Skipping completed self-supervised run: {dataset}/{aggregator}/{prefix}")
      continue

    run_command(
      self_supervised_command(
        args, project_root, data_dir, models_dir, checkpoints_dir, results_dir, logs_dir,
        dataset, aggregator, prefix,
      ),
      cwd=project_root,
      command_log=command_log,
      dry_run=args.dry_run,
      continue_on_error=args.continue_on_error,
    )


def run_supervised_phase(args, project_root, data_dir, models_dir, checkpoints_dir,
                         results_dir, logs_dir, command_log, pairs):
  if args.skip_supervised:
    return

  print("\nStarting supervised phase")
  for dataset, aggregator, prefix in run_specs(args, pairs):
    encoder_model, _ = self_supervised_paths(
      models_dir, results_dir, dataset, aggregator, prefix,
    )
    decoder_model, supervised_result = supervised_paths(
      models_dir, results_dir, dataset, aggregator, prefix,
    )

    if (
      args.resume
      and decoder_model.exists()
      and result_complete(supervised_result, ["test_ap", "test_auc"])
    ):
      print(f"Skipping completed supervised run: {dataset}/{aggregator}/{prefix}")
      continue

    if not args.dry_run and not encoder_model.exists():
      message = f"Missing encoder model for supervised run: {encoder_model}"
      if args.continue_on_error:
        print(message)
        continue
      raise FileNotFoundError(message)

    run_command(
      supervised_command(
        args, project_root, data_dir, models_dir, checkpoints_dir, results_dir, logs_dir,
        dataset, aggregator, prefix,
      ),
      cwd=project_root,
      command_log=command_log,
      dry_run=args.dry_run,
      continue_on_error=args.continue_on_error,
    )


def run_training_matrix(args, project_root, data_dir, models_dir, checkpoints_dir,
                        results_dir, logs_dir, command_log, pairs):
  run_self_supervised_phase(
    args, project_root, data_dir, models_dir, checkpoints_dir, results_dir,
    logs_dir, command_log, pairs,
  )
  run_supervised_phase(
    args, project_root, data_dir, models_dir, checkpoints_dir, results_dir,
    logs_dir, command_log, pairs,
  )


def generate_reports(args, project_root, results_dir, reports_dir, command_log):
  if args.skip_reports:
    return

  reports_dir.mkdir(parents=True, exist_ok=True)
  common = [
    "--results-dir",
    results_dir,
    "--output-dir",
    reports_dir,
    "--datasets",
    *args.datasets,
    "--aggregators",
    *args.aggregators,
    "--expected-runs",
    str(args.n_runs),
    "--prefixes",
    *run_prefixes(args.n_runs),
  ]

  if not args.skip_self_supervised:
    run_command(
      [
        args.python,
        project_root / "export_results_to_excel.py",
        "--output-name",
        "self_supervised_report.xlsx",
        *common,
      ],
      cwd=project_root,
      command_log=command_log,
      dry_run=args.dry_run,
      continue_on_error=args.continue_on_error,
    )

  if not args.skip_supervised:
    run_command(
      [
        args.python,
        project_root / "export_supervised_results_to_excel.py",
        "--output-name",
        "supervised_report.xlsx",
        *common,
      ],
      cwd=project_root,
      command_log=command_log,
      dry_run=args.dry_run,
      continue_on_error=args.continue_on_error,
    )


def main():
  args = parse_args()
  if args.n_runs < 1:
    raise ValueError("--n-runs must be at least 1.")
  if args.batch_size < 1:
    raise ValueError("--batch-size must be at least 1.")
  checkpoint_mode_count = sum([
    bool(args.checkpoints_on_runtime),
    bool(args.checkpoints_with_models),
    bool(args.checkpoint_base),
  ])
  if checkpoint_mode_count > 1:
    raise ValueError(
      "Use only one of --checkpoints-on-runtime, --checkpoints-with-models, or --checkpoint-base."
    )

  project_root = get_project_root()
  data_dir = Path(args.data_dir).expanduser() if args.data_dir else get_data_dir()
  experiment_name = args.experiment_name or datetime.now().strftime("isolated_%Y%m%d_%H%M%S")
  output_base = Path(args.output_base).expanduser() if args.output_base else default_output_base()
  exp_root = output_base / experiment_name
  models_dir = exp_root / "models"
  if args.checkpoints_on_runtime or (is_colab_runtime() and not args.checkpoints_with_models and not args.checkpoint_base):
    checkpoint_base = default_runtime_checkpoint_base()
    checkpoints_dir = checkpoint_base / experiment_name / "checkpoints"
  elif args.checkpoint_base:
    checkpoint_base = Path(args.checkpoint_base).expanduser()
    checkpoints_dir = checkpoint_base / experiment_name / "checkpoints"
  else:
    checkpoints_dir = models_dir
  results_dir = exp_root / "results"
  reports_dir = exp_root / "reports"
  logs_dir = exp_root / "logs"
  command_log = exp_root / "commands.txt"
  pairs = experiment_pairs(args)

  if not pairs:
    raise ValueError("No dataset/aggregator pairs selected.")

  for path in (exp_root, models_dir, checkpoints_dir, results_dir, reports_dir, logs_dir):
    path.mkdir(parents=True, exist_ok=True)

  write_manifest(
    exp_root / "manifest.json",
    args,
    exp_root,
    data_dir,
    models_dir,
    checkpoints_dir,
    results_dir,
    reports_dir,
    logs_dir,
    pairs,
  )

  print(f"Experiment folder: {exp_root}")
  print(f"Per-epoch checkpoints: {checkpoints_dir}")
  print(f"Pairs: {len(pairs)}")
  print(f"Runs per pair: {args.n_runs}")
  print(f"Self-supervised runs planned: {0 if args.skip_self_supervised else len(pairs) * args.n_runs}")
  print(f"Supervised runs planned: {0 if args.skip_supervised else len(pairs) * args.n_runs}")

  maybe_generate_data(args, data_dir, project_root, command_log)
  run_training_matrix(
    args,
    project_root,
    data_dir,
    models_dir,
    checkpoints_dir,
    results_dir,
    logs_dir,
    command_log,
    pairs,
  )
  generate_reports(args, project_root, results_dir, reports_dir, command_log)
  print(f"\nDone. All outputs for this experiment are under: {exp_root}")


if __name__ == "__main__":
  main()
