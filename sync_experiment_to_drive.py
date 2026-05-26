import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_BASE = PROJECT_ROOT / "outputs" / "experiment_runs"
DEFAULT_DRIVE_BASE = Path("/content/drive/MyDrive/tgn_experiment_runs")


def parse_args():
  parser = argparse.ArgumentParser(
    description=(
      "Copy a completed experiment folder from Colab runtime storage to Google Drive. "
      "By default this copies final models, results, reports, logs, manifest.json, and "
      "commands.txt, while skipping bulky per-epoch checkpoint files."
    )
  )
  parser.add_argument("--experiment-root", default=None,
                      help="Exact experiment folder to copy.")
  parser.add_argument("--experiment-name", default=None,
                      help="Experiment folder name, for example isolated_20260523_204320.")
  parser.add_argument("--source-base", default=None,
                      help="Parent directory containing experiment folders.")
  parser.add_argument("--drive-base", default=str(DEFAULT_DRIVE_BASE),
                      help="Drive parent directory receiving experiment folders.")
  parser.add_argument("--include-epoch-checkpoints", action="store_true",
                      help="Also copy files ending in _<epoch>.pth. Not recommended.")
  parser.add_argument("--force", action="store_true",
                      help="Overwrite destination files even when size and mtime look current.")
  parser.add_argument("--dry-run", action="store_true",
                      help="Print what would be copied without copying.")
  return parser.parse_args()


def is_relative_to(path, parent):
  try:
    path.resolve().relative_to(parent.resolve())
    return True
  except ValueError:
    return False


def experiment_candidates(source_base, drive_base):
  bases = []
  if source_base:
    bases.append(Path(source_base).expanduser())
  else:
    bases.extend([DEFAULT_LOCAL_BASE, drive_base])

  candidates = []
  for base in bases:
    if not base.exists():
      continue
    candidates.extend(path for path in base.iterdir() if path.is_dir())
  return candidates


def select_experiment(args, drive_base):
  if args.experiment_root:
    root = Path(args.experiment_root).expanduser()
    if not root.exists():
      raise FileNotFoundError(f"Experiment folder does not exist: {root}")
    return root

  candidates = experiment_candidates(args.source_base, drive_base)
  if args.experiment_name:
    matches = [path for path in candidates if path.name == args.experiment_name]
    if not matches:
      raise FileNotFoundError(
        f"No experiment named {args.experiment_name!r} found. "
        "Pass --experiment-root with the exact folder if needed."
      )
    return max(matches, key=lambda path: path.stat().st_mtime)

  if not candidates:
    raise FileNotFoundError(
      f"No experiment folders found under {DEFAULT_LOCAL_BASE} or {drive_base}. "
      "Pass --experiment-root with the exact folder."
    )
  return max(candidates, key=lambda path: path.stat().st_mtime)


def is_epoch_checkpoint(path):
  return path.suffix == ".pth" and path.stem.rsplit("_", 1)[-1].isdigit()


def should_copy(src, dst, force):
  if force or not dst.exists():
    return True
  if src.stat().st_size != dst.stat().st_size:
    return True
  return src.stat().st_mtime > dst.stat().st_mtime + 1


def copy_experiment(source, destination, include_epoch_checkpoints, force, dry_run):
  copied = 0
  skipped = 0
  bytes_copied = 0

  for src in sorted(source.rglob("*")):
    rel = src.relative_to(source)
    dst = destination / rel

    if src.is_dir():
      if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
      continue

    if is_epoch_checkpoint(src) and not include_epoch_checkpoints:
      skipped += 1
      continue

    if not should_copy(src, dst, force):
      skipped += 1
      continue

    copied += 1
    bytes_copied += src.stat().st_size
    print(f"{'Would copy' if dry_run else 'Copying'} {rel}")
    if not dry_run:
      dst.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(src, dst)

  return copied, skipped, bytes_copied


def summarize_folder(path):
  summary = {}
  for name in ["models", "results", "reports", "logs"]:
    subdir = path / name
    if not subdir.exists():
      summary[name] = {"files": 0, "bytes": 0}
      continue
    files = [item for item in subdir.rglob("*") if item.is_file()]
    summary[name] = {
      "files": len(files),
      "bytes": sum(item.stat().st_size for item in files),
    }
  return summary


def write_sync_manifest(destination, source, copied, skipped, bytes_copied):
  manifest = {
    "synced_at": datetime.now().isoformat(timespec="seconds"),
    "source": str(source),
    "destination": str(destination),
    "copied_files": copied,
    "skipped_files": skipped,
    "bytes_copied": bytes_copied,
  }
  (destination / "drive_sync_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
  )


def print_summary(path):
  print("\nSaved location")
  print(path)
  print("\nFolder summary")
  for name, values in summarize_folder(path).items():
    size_mb = values["bytes"] / (1024 * 1024)
    print(f"  {name}: {values['files']} files, {size_mb:.2f} MB")


def main():
  args = parse_args()
  drive_base = Path(args.drive_base).expanduser()
  source = select_experiment(args, drive_base)

  if is_relative_to(source, drive_base):
    print("Experiment is already on Drive; no copy needed.")
    print_summary(source)
    return

  if not drive_base.exists() and drive_base.parent.exists():
    if not args.dry_run:
      drive_base.mkdir(parents=True, exist_ok=True)
  elif not drive_base.exists():
    raise FileNotFoundError(
      f"Drive destination does not exist: {drive_base}\n"
      "In Colab, run:\n"
      "  from google.colab import drive\n"
      "  drive.mount('/content/drive')"
    )

  destination = drive_base / source.name
  print(f"Source:      {source}")
  print(f"Destination: {destination}")
  print("Skipping per-epoch checkpoint files ending in _<epoch>.pth")

  if not args.dry_run:
    destination.mkdir(parents=True, exist_ok=True)

  copied, skipped, bytes_copied = copy_experiment(
    source=source,
    destination=destination,
    include_epoch_checkpoints=args.include_epoch_checkpoints,
    force=args.force,
    dry_run=args.dry_run,
  )

  if not args.dry_run:
    write_sync_manifest(destination, source, copied, skipped, bytes_copied)

  print(f"\nCopied files: {copied}")
  print(f"Skipped files: {skipped}")
  print(f"Copied size: {bytes_copied / (1024 * 1024):.2f} MB")
  print_summary(destination)


if __name__ == "__main__":
  main()
