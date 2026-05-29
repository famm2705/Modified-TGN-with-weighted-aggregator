# TGN: Temporal Graph Networks [[arXiv](https://arxiv.org/abs/2006.10637), [YouTube](https://www.youtube.com/watch?v=W1GvX2ZcUmY), [Blog Post](https://towardsdatascience.com/temporal-graph-networks-ab8f327f2efe)] 

Dynamic Graph             |  TGN	
:-------------------------:|:-------------------------:	
![](figures/dynamic_graph.png)  |  ![](figures/tgn.png)	

#### Paper link: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637)


## What's Difference between tgn and tgn-aa
Multiple choices can be considered for implementing the Message Aggregator module. While the original paper only considered two efficient __non-learnable__ solutions: most recent message (keep only most recent message for a given node) and mean message (average all messages for a given node), our tgn-aa design a __learnable attention-based aggregation__ to aggregate the messages from multiple events for nodes in the same batch

![TGN-AA](https://github.com/kAI-swa/tgn-aa/assets/146005327/cc9d3cc7-3824-4196-9c73-2b5e7097c362)


## What's changed
```lua
|-modules
|- |-message_aggregator.py add AttentionMessageAggregator
|-train_self_supervised.py add attention for argument --aggregator
|-train_supervised.py add attention for argument --aggregator
```

### Install locally
The project is standard Python and is not tied to Windows. Use the activation command for your shell:

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux, macOS, WSL, and Colab:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CUDA-specific PyTorch wheels, install the matching `torch` build first, then run `pip install -r requirements.txt`.

### Preprocess the data
We use the dense `npy` format to save the features in binary format. If edge features or nodes 
features are absent, they will be replaced by a vector of zeros. 
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
```

### Local and Colab paths
The scripts no longer require Google Drive paths. By default:
- Data is read from and written to `<repo>/data`.
- Local training outputs go to `<repo>/outputs/tgn_models`, `<repo>/outputs/tgn_results`, and `<repo>/outputs/tgn_reports`.
- In Colab, if Google Drive is mounted at `/content/drive/MyDrive`, training and report outputs use `/content/drive/MyDrive/tgn_models`, `/content/drive/MyDrive/tgn_results`, and `/content/drive/MyDrive/tgn_reports`.

Override paths with CLI flags when needed:
```{bash}
python train_self_supervised.py --data wikipedia --use_memory \
  --data-dir ./data --output-root ./outputs

python train_supervised.py --data wikipedia --use_memory \
  --data-dir ./data --output-root ./outputs

python export_results_to_excel.py \
  --results-dir ./outputs/tgn_results --output-dir ./outputs/tgn_reports
```

Use `--model-dir`, `--results-dir`, or `--log-dir` when you need to override one output directory specifically. The same defaults can be overridden with environment variables: `TGN_DATA_DIR`, `TGN_OUTPUT_ROOT`, `TGN_MODEL_DIR`, `TGN_RESULTS_DIR`, `TGN_REPORTS_DIR`, and `TGN_LOG_DIR`.

### Isolated toy aggregator experiments
Use the Python runner instead of the old shell command lists. By default it uses the v3 label-prediction toy suite, creates a new timestamped experiment folder, generates missing toy data, runs the full `4 datasets x 4 aggregators x 10 runs` self-supervised matrix, then runs the matching supervised edge-label jobs from the corresponding encoder checkpoints.

```{bash}
python run_isolated_experiments.py
```

The default v3 datasets are:
- `toy_v3_last_event`
- `toy_v3_persistent_mean`
- `toy_v3_rare_spike`
- `toy_v3_ordered_pattern`

V3 supervised training uses `query_mask=1` rows by default. History and filler rows still update TGN memory, but the supervised decoder loss and metrics are computed only on neutral-feature query edges, so the task is label prediction from prior temporal history rather than current-edge feature leakage.

Each experiment writes isolated `models`, `results`, `reports`, and `logs` subfolders. On Colab, when Google Drive is mounted, the default parent is `/content/drive/MyDrive/tgn_experiment_runs`; otherwise it is `outputs/experiment_runs`.

On Colab, temporary per-epoch early-stopping checkpoints default to `/content/tgn_experiment_checkpoints/<experiment-name>/checkpoints` so they do not fill Drive. Completed jobs delete their per-epoch checkpoints after saving the final model/decoder. You can still pass the flag explicitly:
```{bash}
python -u run_isolated_experiments.py \
  --regenerate-data \
  --checkpoints-on-runtime
```

With this layout, final encoder/decoder files are saved in the Drive experiment `models` folder, while temporary checkpoints use the Colab runtime disk. Use `--checkpoints-with-models` only if you intentionally want checkpoint files in the experiment `models` folder.

Useful smaller runs:
```{bash}
# Only intended dataset/aggregator pairs
python run_isolated_experiments.py --pairs-only

# Run the previous v2 toy suite
python run_isolated_experiments.py --dataset-suite v2

# Regenerate isolated toy datasets before training
python run_isolated_experiments.py --regenerate-data

# Inspect commands without launching training
python run_isolated_experiments.py --pairs-only --n-runs 1 --dry-run
```

From MATLAB:
```matlab
system("python run_isolated_experiments.py --pairs-only")
```

### Training on Wikipedia Dataset
```{bash}
### tgn-aa
# TGN-attn with attention aggregator: Self-Supervised learning on the wikipedia dataset
## unlearnable
python train_self_supervised.py --aggregator attention --use_memory --prefix tgn-attn --n_runs 10
## learnable
python train_self_supervised.py --aggregator attention --learnable --use_memory --prefix tgn-attn --n_runs 10 --add_cls_token


### Baselines
# Jodie
python train_self_supervised.py --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --n_runs 10

# Jodie
python train_supervised.py --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --n_runs 10
```



