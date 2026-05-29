import math
import logging
import time
import sys
import random
import argparse
import pickle
import csv
from pathlib import Path

import torch
import numpy as np

from tgn import TGN
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from utils.data_processing import compute_time_statistics, get_data_node_classification
from evaluation.evaluation import eval_edge_label_prediction
from utils.paths import get_checkpoints_dir, get_data_dir, get_logs_dir, get_models_dir, get_project_root, get_results_dir

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN supervised edge-label training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--require-gpu', action='store_true',
                    help='Fail instead of falling back to CPU when CUDA is unavailable.')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--aggregator', type=str, default="last", choices=["last", "mean", "weightedmean", "attention"], help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--use_validation', action='store_true',
                    help='Whether to use a validation set')
parser.add_argument('--query-only-labels', action='store_true',
                    help='Train and evaluate edge-label decoder only on rows with query_mask=1.')
parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--learnable', action="store_true",
                    help="Whether Message Aggregator is learnable module")
parser.add_argument('--add_cls_token', action="store_true",
                    help="Apend cls token like BERT to represent the final message")
parser.add_argument('--data-dir', default=None,
                    help='Directory containing preprocessed ml_<dataset> files.')
parser.add_argument('--output-root', default=None,
                    help='Root directory for default model/result outputs.')
parser.add_argument('--model-dir', default=None,
                    help='Directory containing encoder checkpoints and receiving decoder checkpoints.')
parser.add_argument('--checkpoint-dir', default=None,
                    help='Directory for per-epoch decoder early-stopping checkpoints. Defaults to --model-dir.')
parser.add_argument('--results-dir', default=None,
                    help='Directory for supervised result pickle files and metrics CSVs.')
parser.add_argument('--log-dir', default=None,
                    help='Directory for training log files.')
parser.add_argument('--keep-checkpoints', action='store_true',
                    help='Keep per-epoch decoder early-stopping checkpoints after the final decoder is saved.')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

DATA_DIR = get_data_dir(args.data_dir)
MODEL_BASE_PATH = get_models_dir(args.model_dir, args.output_root)
CHECKPOINT_BASE_PATH = get_checkpoints_dir(args.checkpoint_dir, args.model_dir, args.output_root)
RESULTS_PATH = get_results_dir(args.results_dir, args.output_root)
MODEL_BASE_PATH.mkdir(parents=True, exist_ok=True)
CHECKPOINT_BASE_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

ENCODER_MODEL_PATH = MODEL_BASE_PATH / f'{args.prefix}_{args.data}_{args.aggregator}.pth'
DECODER_MODEL_SAVE_PATH = (
  MODEL_BASE_PATH / f'supervised_{args.prefix}_{args.data}_{args.aggregator}_edge_label_decoder.pth'
)
get_checkpoint_path = lambda epoch: (
  CHECKPOINT_BASE_PATH / f'supervised_{args.prefix}_{args.data}_{args.aggregator}_edge_label_decoder_{epoch}.pth'
)


def cleanup_epoch_checkpoints():
  if args.keep_checkpoints:
    return
  pattern = f'supervised_{args.prefix}_{args.data}_{args.aggregator}_edge_label_decoder_[0-9]*.pth'
  for path in CHECKPOINT_BASE_PATH.glob(pattern):
    path.unlink(missing_ok=True)

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
LOG_DIR = get_logs_dir(args.log_dir, args.output_root)
LOG_DIR.mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(LOG_DIR / '{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

full_data, node_features, edge_features, train_data, val_data, test_data = \
  get_data_node_classification(DATA, use_validation=args.use_validation, data_dir=DATA_DIR)

max_idx = max(full_data.unique_nodes)

train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)
full_ngh_finder = get_neighbor_finder(full_data, uniform=UNIFORM, max_node_idx=max_idx)
EDGE_DECODER_INPUT_DIM = 2 * node_features.shape[1] + edge_features.shape[1]

# Set device
if args.require_gpu and not torch.cuda.is_available():
  raise RuntimeError(
    "CUDA is not available in this Python process. In Colab, select Runtime > "
    "Change runtime type > GPU, then verify `torch.cuda.is_available()` is true."
  )
if torch.cuda.is_available():
  device_count = torch.cuda.device_count()
  if GPU < 0 or GPU >= device_count:
    raise ValueError(f'Invalid --gpu {GPU}; CUDA device count is {device_count}.')
  torch.cuda.set_device(GPU)
  device = torch.device(f'cuda:{GPU}')
  logger.info(f'Using CUDA device {GPU}: {torch.cuda.get_device_name(GPU)}')
else:
  device = torch.device('cpu')
  logger.warning('CUDA is not available; using CPU.')

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = (
    RESULTS_PATH / f"supervised_{args.prefix}_{args.data}_{args.aggregator}_edge_label_classification_{i}.pkl"
    if i > 0
    else RESULTS_PATH / f"supervised_{args.prefix}_{args.data}_{args.aggregator}_edge_label_classification.pkl"
  )

  csv_path = RESULTS_PATH / f"supervised_{args.prefix}_{args.data}_{args.aggregator}_edge_label_classification_metrics.csv"
  if not Path(csv_path).exists():
    with open(csv_path, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow([
        "run",
        "epoch",
        "train_loss",
        "val_auc",
        "val_ap",
        "val_acc",
        "val_precision",
        "val_recall",
        "val_f1",
      ])

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message, learnable=args.learnable,
            add_cls_token=args.add_cls_token)

  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)
  
  logger.debug('Num of training instances: {}'.format(num_instance))
  logger.debug('Num of batches per epoch: {}'.format(num_batch))

  logger.info('Loading saved TGN model')
  tgn.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device))
  tgn.eval()
  logger.info(f'TGN model loaded from {ENCODER_MODEL_PATH}')
  logger.info('Start training edge label classification task')

  decoder = MLP(EDGE_DECODER_INPUT_DIM, drop=DROP_OUT)
  decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
  decoder = decoder.to(device)
  decoder_loss_criterion = torch.nn.BCEWithLogitsLoss()

  val_aps = []
  val_aucs = []
  val_accs = []
  val_precs = []
  val_recs = []
  val_f1s = []
  train_losses = []
  epoch_times = []
  total_epoch_times = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(args.n_epoch):
    start_epoch = time.time()
    
    # Initialize memory of the model at each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    tgn.set_neighbor_finder(train_ngh_finder)
    tgn = tgn.eval()
    decoder = decoder.train()
    loss = 0
    supervised_batches = 0
    
    for k in range(num_batch):
      s_idx = k * BATCH_SIZE
      e_idx = min(num_instance, s_idx + BATCH_SIZE)

      sources_batch = train_data.sources[s_idx: e_idx]
      destinations_batch = train_data.destinations[s_idx: e_idx]
      timestamps_batch = train_data.timestamps[s_idx: e_idx]
      edge_idxs_batch = train_data.edge_idxs[s_idx: e_idx]
      labels_batch = train_data.labels[s_idx: e_idx]
      query_mask_batch = (
        train_data.query_mask[s_idx: e_idx]
        if args.query_only_labels
        else np.ones(e_idx - s_idx, dtype=bool)
      )

      decoder_optimizer.zero_grad()
      with torch.no_grad():
        source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                     destinations_batch,
                                                                                     destinations_batch,
                                                                                     timestamps_batch,
                                                                                     edge_idxs_batch,
                                                                                     NUM_NEIGHBORS)

      labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
      edge_features_batch = tgn.edge_raw_features[edge_idxs_batch]
      decoder_input = torch.cat([source_embedding, destination_embedding, edge_features_batch],
                                dim=1)
      if args.query_only_labels:
        query_mask_torch = torch.from_numpy(query_mask_batch).bool().to(device)
        if not torch.any(query_mask_torch):
          continue
        decoder_input = decoder_input[query_mask_torch]
        labels_batch_torch = labels_batch_torch[query_mask_torch]

      logits = decoder(decoder_input)
      decoder_loss = decoder_loss_criterion(logits, labels_batch_torch)
      decoder_loss.backward()
      decoder_optimizer.step()
      loss += decoder_loss.item()
      supervised_batches += 1
    if supervised_batches == 0:
      raise RuntimeError(
        "No supervised labels were available for this epoch. "
        "If this is not a query-mask dataset, remove --query-only-labels."
      )
    mean_train_loss = loss / supervised_batches
    train_losses.append(mean_train_loss)
    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    tgn.set_neighbor_finder(full_ngh_finder)
    val_metrics = eval_edge_label_prediction(tgn, decoder, val_data, BATCH_SIZE,
                                             n_neighbors=NUM_NEIGHBORS,
                                             query_only=args.query_only_labels)
    val_auc = val_metrics["auc"]
    val_ap = val_metrics["ap"]
    val_aucs.append(val_auc)
    val_aps.append(val_ap)
    val_accs.append(val_metrics["acc"])
    val_precs.append(val_metrics["precision"])
    val_recs.append(val_metrics["recall"])
    val_f1s.append(val_metrics["f1"])
    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    pickle.dump({
      "val_aps": val_aps,
      "val_aucs": val_aucs,
      "val_accs": val_accs,
      "val_precs": val_precs,
      "val_recs": val_recs,
      "val_f1s": val_f1s,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times,
      "new_nodes_val_aps": [],
      "source_encoder_model": str(ENCODER_MODEL_PATH),
      "decoder_model": str(DECODER_MODEL_SAVE_PATH),
      "prediction_task": "edge_label_classification",
      "label_filter": "query_only" if args.query_only_labels else "all_edges",
      "val_n_eval": val_metrics.get("n_eval", 0),
      "edge_decoder_input_dim": EDGE_DECODER_INPUT_DIM,
    }, open(results_path, "wb"))

    logger.info(
      f'Epoch {epoch}: train loss: {mean_train_loss}, val auc: {val_auc}, '
      f'val ap: {val_ap}, val f1: {val_metrics["f1"]}, time: {total_epoch_time}')

    with open(csv_path, "a", newline="") as f:
      writer = csv.writer(f)
      writer.writerow([
        i,
        epoch,
        mean_train_loss,
        val_auc,
        val_ap,
        val_metrics["acc"],
        val_metrics["precision"],
        val_metrics["recall"],
        val_metrics["f1"],
      ])

    if args.use_validation:
      early_stop_metric = val_auc
      if np.isnan(early_stop_metric):
        early_stop_metric = val_ap
      if np.isnan(early_stop_metric):
        early_stop_metric = 0.0

      if early_stopper.early_stop_check(early_stop_metric):
        logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
        break
      else:
        torch.save(decoder.state_dict(), get_checkpoint_path(epoch))

  if args.use_validation:
    logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
    best_model_path = get_checkpoint_path(early_stopper.best_epoch)
    decoder.load_state_dict(torch.load(best_model_path, map_location=device))
    logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
    decoder.eval()

    tgn.set_neighbor_finder(full_ngh_finder)
    test_metrics = eval_edge_label_prediction(tgn, decoder, test_data, BATCH_SIZE,
                                              n_neighbors=NUM_NEIGHBORS,
                                              query_only=args.query_only_labels)
  else:
    # If we are not using a validation set, the test performance is just the performance computed
    # in the last epoch
    test_metrics = {
      "auc": val_aucs[-1],
      "ap": val_aps[-1],
      "acc": val_accs[-1],
      "precision": val_precs[-1],
      "recall": val_recs[-1],
      "f1": val_f1s[-1],
      "n_eval": val_metrics.get("n_eval", 0),
    }
    
  pickle.dump({
    "val_aps": val_aps,
    "val_aucs": val_aucs,
    "val_accs": val_accs,
    "val_precs": val_precs,
    "val_recs": val_recs,
    "val_f1s": val_f1s,
    "test_auc": test_metrics["auc"],
    "test_ap": test_metrics["ap"],
    "test_acc": test_metrics["acc"],
    "test_prec": test_metrics["precision"],
    "test_rec": test_metrics["recall"],
    "test_f1": test_metrics["f1"],
    "train_losses": train_losses,
    "epoch_times": epoch_times,
    "total_epoch_times": total_epoch_times,
    "new_nodes_val_aps": [],
    "new_node_test_ap": 0,
    "source_encoder_model": str(ENCODER_MODEL_PATH),
    "decoder_model": str(DECODER_MODEL_SAVE_PATH),
    "prediction_task": "edge_label_classification",
    "label_filter": "query_only" if args.query_only_labels else "all_edges",
    "test_n_eval": test_metrics.get("n_eval", 0),
    "edge_decoder_input_dim": EDGE_DECODER_INPUT_DIM,
  }, open(results_path, "wb"))

  torch.save(decoder.state_dict(), DECODER_MODEL_SAVE_PATH)
  cleanup_epoch_checkpoints()
  logger.info(f'Supervised decoder saved to {DECODER_MODEL_SAVE_PATH}')
  logger.info(
    f'test auc: {test_metrics["auc"]}, test ap: {test_metrics["ap"]}, '
    f'test f1: {test_metrics["f1"]}')
