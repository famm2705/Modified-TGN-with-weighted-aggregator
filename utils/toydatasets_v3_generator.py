import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from utils.paths import get_data_dir
except ModuleNotFoundError:
    from paths import get_data_dir


# V3 datasets are designed for edge label prediction.
#
# With bs=128, each trial is exactly two batches:
#   batch A: 128 history edges for one source node, query_mask=0
#   batch B: 64 query edges for that source, then 64 filler edges
#
# Query edge features are neutral for both classes. The label is recoverable only
# from the source node history compressed into TGN memory by the message aggregator.

NUM_NODES = 256
NUM_SERVERS = 16
FEATURE_DIM = 6
NODE_FEATURE_DIM = 172

HISTORY_SIZE = 128
QUERY_SIZE = 64
FILLER_SIZE = 64
DEFAULT_TRIALS = 80

SERVERS = np.arange(NUM_SERVERS)
QUERY_SERVER = int(SERVERS[0])
TRIAL_DEVICES = np.arange(NUM_SERVERS, NUM_SERVERS + 128)
FILLER_DEVICES = np.arange(NUM_SERVERS + 128, NUM_NODES)

NEUTRAL = np.zeros(FEATURE_DIM)

LAST_POS = np.array([2.2, -2.1, 2.0, -1.9, 1.8, -1.7])
LAST_NEG = -LAST_POS

MEAN_POS = np.array([0.95, 0.72, 0.56, -0.72, 0.48, -0.44])
MEAN_NEG = -MEAN_POS
MEAN_DISTRACTOR = np.array([0.0, 0.0, 2.2, -2.2, 0.0, 0.0])

SPIKE_POS = np.array([5.2, -5.0, 4.6, -4.4, 4.2, -4.0])
SPIKE_NEG = -SPIKE_POS

ORDER_A = np.array([3.4, 0.0, 0.0, -1.7, 0.8, 0.0])
ORDER_B = np.array([0.0, 3.7, 0.0, 0.0, -1.7, 0.8])
ORDER_C = np.array([0.0, 0.0, 3.9, 0.8, 0.0, -2.4])

DATASET_CONFIGS = [
    ("toy_v3_last_event", "last_event", 10101),
    ("toy_v3_persistent_mean", "persistent_mean", 20202),
    ("toy_v3_rare_spike", "rare_spike", 30303),
    ("toy_v3_ordered_pattern", "ordered_pattern", 40404),
]


def jitter(rng, feature, scale):
    return np.asarray(feature, dtype=float) + rng.normal(0.0, scale, FEATURE_DIM)


def split_boundaries(num_trials):
    train_end = max(1, int(round(num_trials * 0.70)))
    val_end = max(train_end + 1, int(round(num_trials * 0.85)))
    val_end = min(val_end, num_trials - 1) if num_trials > 2 else num_trials
    return train_end, val_end


def trial_split(trial_idx, num_trials):
    train_end, val_end = split_boundaries(num_trials)
    if trial_idx < train_end:
        return "train", trial_idx
    if trial_idx < val_end:
        return "val", trial_idx - train_end
    return "test", trial_idx - val_end


def trial_source_and_label(trial_idx):
    # Default v3 runs use at most 80 trials, so each trial gets a fresh source.
    # This prevents source state from carrying a previous trial's label into the
    # current trial and becoming a shortcut around the message aggregator.
    source = int(TRIAL_DEVICES[trial_idx % len(TRIAL_DEVICES)])
    label = int(trial_idx % 2)
    return source, label


def append_event(rows, source, destination, timestamp, feature, label,
                 query_mask, split, trial, role):
    rows.append({
        "u": int(source),
        "i": int(destination),
        "ts": float(timestamp),
        "label": int(label),
        "query_mask": int(query_mask),
        "split": split,
        "trial": int(trial),
        "role": role,
        **{f"f{idx}": float(feature[idx]) for idx in range(FEATURE_DIM)},
    })


def neutral_query_feature(rng):
    return jitter(rng, NEUTRAL, 0.015)


def filler_feature(rng):
    return jitter(rng, NEUTRAL, 0.10)


def filler_edge(rng, trial_source):
    available = FILLER_DEVICES[FILLER_DEVICES != trial_source]
    source = int(rng.choice(available))
    destination = int(rng.choice(SERVERS if rng.random() < 0.85 else available))
    return source, destination, filler_feature(rng)


def last_event_history(rng, label):
    correct_signal = LAST_POS if label else LAST_NEG
    cancel_signal = LAST_NEG if label else LAST_POS
    message_types = ["pos"] * 24 + ["neg"] * 24 + ["cancel"] + ["neutral"] * (HISTORY_SIZE - 50)
    rng.shuffle(message_types)

    history = []
    for message_type in message_types:
        if message_type == "pos":
            history.append(jitter(rng, LAST_POS * 1.15, 0.08))
        elif message_type == "neg":
            history.append(jitter(rng, LAST_NEG * 1.15, 0.08))
        elif message_type == "cancel":
            history.append(jitter(rng, cancel_signal * 2.6, 0.03))
        else:
            history.append(jitter(rng, NEUTRAL, 0.18))

    history.append(jitter(rng, correct_signal * 2.6, 0.03))
    return history


def persistent_mean_history(rng, label):
    majority = MEAN_POS if label else MEAN_NEG
    minority = MEAN_NEG if label else MEAN_POS
    message_types = (
        ["majority"] * 96
        + ["minority"] * 24
        + ["distractor"] * 6
        + ["neutral"] * 2
    )
    rng.shuffle(message_types)
    message_types[-1] = "neutral"

    history = []
    for message_type in message_types:
        if message_type == "majority":
            history.append(jitter(rng, majority, 0.08))
        elif message_type == "minority":
            history.append(jitter(rng, minority, 0.08))
        elif message_type == "distractor":
            sign = 1.0 if rng.random() < 0.5 else -1.0
            history.append(jitter(rng, sign * MEAN_DISTRACTOR, 0.06))
        else:
            history.append(jitter(rng, NEUTRAL, 0.16))
    return history


def rare_spike_history(rng, label):
    history = [jitter(rng, NEUTRAL, 0.35) for _ in range(HISTORY_SIZE)]
    spike = SPIKE_POS if label else SPIKE_NEG
    anti_spike = SPIKE_NEG if label else SPIKE_POS

    spike_positions = rng.choice(np.arange(10, HISTORY_SIZE - 10), size=2, replace=False)
    for pos in spike_positions:
        history[pos] = jitter(rng, spike, 0.04)

    # Small opposite-sign distractors keep the mean close to neutral while preserving
    # a few dominant salient events for the weighted aggregator to discover.
    distractor_positions = rng.choice(
        np.setdiff1d(np.arange(8, HISTORY_SIZE - 8), spike_positions),
        size=6,
        replace=False,
    )
    for pos in distractor_positions:
        history[pos] = jitter(rng, anti_spike * 0.75, 0.08)

    history[-1] = jitter(rng, NEUTRAL, 0.12)
    return history


def ordered_pattern_history(rng, label):
    history = [jitter(rng, NEUTRAL, 0.08) for _ in range(HISTORY_SIZE)]

    # Same multiset and same final token in both classes. The discriminative
    # information is whether A precedes B or B precedes A before the final C.
    pattern = [ORDER_A, ORDER_B, ORDER_C] if label else [ORDER_B, ORDER_A, ORDER_C]
    positions = [HISTORY_SIZE - 5, HISTORY_SIZE - 3, HISTORY_SIZE - 1]
    for pos, token in zip(positions, pattern):
        history[pos] = jitter(rng, token, 0.02)

    decoy_positions = rng.choice(np.arange(8, HISTORY_SIZE - 24), size=6, replace=False)
    decoys = [ORDER_A, ORDER_B, ORDER_C] * 2
    rng.shuffle(decoys)
    for pos, token in zip(decoy_positions, decoys):
        history[pos] = jitter(rng, token * 0.55, 0.05)

    return history


def build_history(rng, mode, label):
    if mode == "last_event":
        return last_event_history(rng, label)
    if mode == "persistent_mean":
        return persistent_mean_history(rng, label)
    if mode == "rare_spike":
        return rare_spike_history(rng, label)
    if mode == "ordered_pattern":
        return ordered_pattern_history(rng, label)
    raise ValueError(f"Unknown v3 toy mode: {mode}")


def generate_dataset(mode, seed, num_trials=DEFAULT_TRIALS):
    rng = np.random.default_rng(seed)
    rows = []
    timestamp = 0.0

    for trial_idx in range(num_trials):
        split, local_idx = trial_split(trial_idx, num_trials)
        source, label = trial_source_and_label(trial_idx)
        history = build_history(rng, mode, label)

        for feature in history:
            timestamp += 1.0
            destination = int(rng.choice(SERVERS[1:]))
            append_event(
                rows, source, destination, timestamp, feature, 0,
                query_mask=0, split=split, trial=trial_idx, role="history",
            )

        for _ in range(QUERY_SIZE):
            timestamp += 1.0
            append_event(
                rows, source, QUERY_SERVER, timestamp, neutral_query_feature(rng), label,
                query_mask=1, split=split, trial=trial_idx, role="query",
            )

        for _ in range(FILLER_SIZE):
            timestamp += 1.0
            filler_source, filler_destination, feature = filler_edge(rng, source)
            append_event(
                rows, filler_source, filler_destination, timestamp, feature, 0,
                query_mask=0, split=split, trial=trial_idx, role="filler",
            )

    return pd.DataFrame(rows)


def write_preprocessed_dataset(df, data_name, output_dir):
    ml_df = df[["u", "i", "ts", "label", "query_mask", "split", "trial", "role"]].copy()
    ml_df["idx"] = np.arange(len(ml_df))

    # Match the existing toy preprocessing: index 0 is reserved for padding.
    ml_df["u"] = ml_df["u"] + 1
    ml_df["i"] = ml_df["i"] + 1
    ml_df["idx"] = ml_df["idx"] + 1

    feature_columns = [f"f{idx}" for idx in range(FEATURE_DIM)]
    features = df[feature_columns].to_numpy(dtype=float)
    features = np.vstack([np.zeros((1, FEATURE_DIM)), features])

    max_node_idx = int(max(ml_df["u"].max(), ml_df["i"].max()))
    node_features = np.zeros((max_node_idx + 1, NODE_FEATURE_DIM))

    ml_df.to_csv(output_dir / f"ml_{data_name}.csv", index=False)
    np.save(output_dir / f"ml_{data_name}.npy", features)
    np.save(output_dir / f"ml_{data_name}_node.npy", node_features)


def validate_dataset(data_name, df):
    split_parts = []
    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split]
        query_df = split_df[split_df["query_mask"] == 1]
        positives = int(query_df["label"].sum())
        rate = positives / max(1, len(query_df))
        split_parts.append(
            f"{split}_query={positives}/{len(query_df)} ({rate:.1%})"
        )

    query_count = int(df["query_mask"].sum())
    total_query_pos = int(df.loc[df["query_mask"] == 1, "label"].sum())
    print(
        f"{data_name}: {len(df)} edges | {query_count} query labels | "
        f"{total_query_pos} query positives ({total_query_pos / max(1, query_count):.1%}) | "
        + " | ".join(split_parts)
    )


def generate_all(output_dir, num_trials):
    output_dir.mkdir(parents=True, exist_ok=True)
    for data_name, mode, seed in DATASET_CONFIGS:
        df = generate_dataset(mode=mode, seed=seed, num_trials=num_trials)
        df.to_csv(output_dir / f"{data_name}.csv", index=False)
        write_preprocessed_dataset(df, data_name, output_dir)
        validate_dataset(data_name, df)


def parse_args():
    parser = argparse.ArgumentParser("Generate v3 isolated TGN label-prediction toy datasets")
    parser.add_argument("--output-dir", default=str(get_data_dir()),
                        help="Directory where raw and preprocessed toy datasets are written.")
    parser.add_argument("--num-trials", type=int, default=DEFAULT_TRIALS,
                        help="Number of two-batch trials per dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all(Path(args.output_dir), args.num_trials)
