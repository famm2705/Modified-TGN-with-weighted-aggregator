import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive")


def _non_empty(value):
  if value is None:
    return None
  value = str(value).strip()
  return value or None


def _as_path(value):
  return Path(value).expanduser()


def get_project_root():
  return PROJECT_ROOT


def get_output_root(output_root=None):
  explicit = _non_empty(output_root)
  if explicit:
    return _as_path(explicit)

  env_value = _non_empty(os.environ.get("TGN_OUTPUT_ROOT"))
  if env_value:
    return _as_path(env_value)

  if COLAB_DRIVE_ROOT.exists():
    return COLAB_DRIVE_ROOT

  return PROJECT_ROOT / "outputs"


def get_data_dir(data_dir=None):
  explicit = _non_empty(data_dir)
  if explicit:
    return _as_path(explicit)

  env_value = _non_empty(os.environ.get("TGN_DATA_DIR"))
  if env_value:
    return _as_path(env_value)

  return PROJECT_ROOT / "data"


def get_models_dir(model_dir=None, output_root=None):
  explicit = _non_empty(model_dir)
  if explicit:
    return _as_path(explicit)

  env_value = _non_empty(os.environ.get("TGN_MODEL_DIR"))
  if env_value:
    return _as_path(env_value)

  return get_output_root(output_root) / "tgn_models"


def get_checkpoints_dir(checkpoint_dir=None, model_dir=None, output_root=None):
  explicit = _non_empty(checkpoint_dir)
  if explicit:
    return _as_path(explicit)

  env_value = _non_empty(os.environ.get("TGN_CHECKPOINT_DIR"))
  if env_value:
    return _as_path(env_value)

  return get_models_dir(model_dir, output_root)


def get_results_dir(results_dir=None, output_root=None):
  explicit = _non_empty(results_dir)
  if explicit:
    return _as_path(explicit)

  env_value = _non_empty(os.environ.get("TGN_RESULTS_DIR"))
  if env_value:
    return _as_path(env_value)

  return get_output_root(output_root) / "tgn_results"


def get_reports_dir(reports_dir=None, output_root=None):
  explicit = _non_empty(reports_dir)
  if explicit:
    return _as_path(explicit)

  env_value = _non_empty(os.environ.get("TGN_REPORTS_DIR"))
  if env_value:
    return _as_path(env_value)

  return get_output_root(output_root) / "tgn_reports"


def get_logs_dir(log_dir=None, output_root=None):
  explicit = _non_empty(log_dir)
  if explicit:
    return _as_path(explicit)

  env_value = _non_empty(os.environ.get("TGN_LOG_DIR"))
  if env_value:
    return _as_path(env_value)

  return get_output_root(output_root) / "tgn_logs"


def ensure_dir(path):
  path = _as_path(path)
  path.mkdir(parents=True, exist_ok=True)
  return path
