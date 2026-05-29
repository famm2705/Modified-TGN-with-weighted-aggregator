import math

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc, val_acc, val_precision, val_recall, val_f1, val_mrr = [], [], [], [], [], [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])
      pred_label = (pred_score >= 0.5).astype(int)

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))
      val_acc.append(accuracy_score(true_label, pred_label))
      val_precision.append(precision_score(true_label, pred_label, zero_division=0))
      val_recall.append(recall_score(true_label, pred_label, zero_division=0))
      val_f1.append(f1_score(true_label, pred_label, zero_division=0))
      ranks = 1.0 + (neg_prob > pos_prob).float().cpu().numpy()
      val_mrr.append(np.mean(1.0 / ranks))

  return (
    np.mean(val_ap),
    np.mean(val_auc),
    np.mean(val_acc),
    np.mean(val_precision),
    np.mean(val_recall),
    np.mean(val_f1),
    np.mean(val_mrr),
  )


def binary_classification_metrics(labels, pred_prob):
  labels = np.asarray(labels).astype(int)
  pred_prob = np.asarray(pred_prob)
  if len(labels) == 0:
    return {
      "ap": np.nan,
      "auc": np.nan,
      "acc": np.nan,
      "precision": np.nan,
      "recall": np.nan,
      "f1": np.nan,
      "n_eval": 0,
    }
  pred_label = (pred_prob >= 0.5).astype(int)
  has_both_classes = len(np.unique(labels)) > 1

  return {
    "ap": average_precision_score(labels, pred_prob) if has_both_classes else np.nan,
    "auc": roc_auc_score(labels, pred_prob) if has_both_classes else np.nan,
    "acc": accuracy_score(labels, pred_label),
    "precision": precision_score(labels, pred_label, zero_division=0),
    "recall": recall_score(labels, pred_label, zero_division=0),
    "f1": f1_score(labels, pred_label, zero_division=0),
    "n_eval": len(labels),
  }


def eval_edge_label_prediction(tgn, decoder, data, batch_size, n_neighbors, query_only=False):
  pred_probs = []
  labels = []
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx: e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]
      labels_batch = data.labels[s_idx: e_idx]
      query_mask_batch = data.query_mask[s_idx: e_idx] if query_only else np.ones(e_idx - s_idx, dtype=bool)

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      edge_features_batch = tgn.edge_raw_features[edge_idxs_batch]
      decoder_input = torch.cat([source_embedding, destination_embedding, edge_features_batch],
                                dim=1)
      pred_prob_batch = decoder(decoder_input).sigmoid()
      if np.any(query_mask_batch):
        pred_probs.append(pred_prob_batch.view(-1).cpu().numpy()[query_mask_batch])
        labels.append(labels_batch[query_mask_batch])

  if not pred_probs:
    return binary_classification_metrics([], [])
  return binary_classification_metrics(np.concatenate(labels), np.concatenate(pred_probs))


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx] if edge_idxs is None else edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.view(-1).cpu().numpy()

  return binary_classification_metrics(data.labels, pred_prob)["auc"]
