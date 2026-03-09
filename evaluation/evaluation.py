import math
import numpy as np
import torch

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    """
    Evaluate link prediction.

    Returns:
    AP
    AUC
    Accuracy
    Precision
    Recall
    F1
    MRR
    Hits@1
    Hits@5
    Hits@10
    """

    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap = []
    val_auc = []

    val_acc = []
    val_prec = []
    val_rec = []
    val_f1 = []

    val_mrr = []
    hits1 = []
    hits5 = []
    hits10 = []

    with torch.no_grad():
        model = model.eval()

        num_instance = len(data.sources)
        num_batch = math.ceil(num_instance / batch_size)

        for k in range(num_batch):

            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

            size = len(sources_batch)

            _, negative_samples = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.compute_edge_probabilities(
                sources_batch,
                destinations_batch,
                negative_samples,
                timestamps_batch,
                edge_idxs_batch,
                n_neighbors
            )

            pos_prob = pos_prob.squeeze().cpu().numpy()
            neg_prob = neg_prob.squeeze().cpu().numpy()

            pred_score = np.concatenate([pos_prob, neg_prob])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            pred_label = (pred_score > 0.5).astype(int)

            # Standard classification metrics
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
            val_acc.append(accuracy_score(true_label, pred_label))
            val_prec.append(precision_score(true_label, pred_label))
            val_rec.append(recall_score(true_label, pred_label))
            val_f1.append(f1_score(true_label, pred_label))

            # Ranking metrics
            batch_mrr = []
            batch_hits1 = []
            batch_hits5 = []
            batch_hits10 = []

            for i in range(size):
                pos = pos_prob[i]
                neg = neg_prob[i]

                scores = np.array([pos, neg])
                rank = (scores > pos).sum() + 1

                batch_mrr.append(1.0 / rank)
                batch_hits1.append(1 if rank <= 1 else 0)
                batch_hits5.append(1 if rank <= 5 else 0)
                batch_hits10.append(1 if rank <= 10 else 0)

            val_mrr.append(np.mean(batch_mrr))
            hits1.append(np.mean(batch_hits1))
            hits5.append(np.mean(batch_hits5))
            hits10.append(np.mean(batch_hits10))

    return (
        np.mean(val_ap),
        np.mean(val_auc),
        np.mean(val_acc),
        np.mean(val_prec),
        np.mean(val_rec),
        np.mean(val_f1),
        np.mean(val_mrr),
        np.mean(hits1),
        np.mean(hits5),
        np.mean(hits10),
    )