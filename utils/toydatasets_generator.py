import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# These datasets are intentionally block-structured for TGN's memory update schedule.
# With bs=128, each trial writes one full history batch followed by one query batch.
# Query edges have neutral current-edge features; their labels depend on the previous
# history batch, where the selected message aggregator is the intended bottleneck.

NUM_NODES = 120
FEATURE_DIM = 6
NODE_FEATURE_DIM = 172
NUM_SERVERS = 15

HISTORY_SIZE = 128
QUERY_SIZE = 64
FILLER_SIZE = 64
DEFAULT_TRIALS = 32

SERVERS = np.arange(NUM_SERVERS)
DEVICES = np.arange(NUM_SERVERS, NUM_NODES)
TRIAL_DEVICES = np.arange(NUM_SERVERS, NUM_SERVERS + 16)
FILLER_DEVICES = np.arange(NUM_SERVERS + 32, NUM_NODES)

NEUTRAL = np.zeros(FEATURE_DIM)
LAST_POS = np.array([1.1, -1.1, 0.9, -0.9, 0.7, -0.7])
LAST_NEG = -LAST_POS
PERSISTENT_POS = np.array([0.35, 0.25, 0.20, -0.25, 0.20, -0.15])
PERSISTENT_NEG = -PERSISTENT_POS
PERSISTENT_DISTRACTOR = np.array([0.0, 0.0, 2.8, -2.8, 0.0, 0.0])
SPIKE_POS = np.array([4.0, -4.0, 3.5, -3.5, 3.2, -3.2])
SPIKE_NEG = -SPIKE_POS
ORDER_A = np.array([2.0, 0.0, 0.0, -1.0, 0.5, 0.0])
ORDER_B = np.array([0.0, 2.2, 0.0, 0.0, -1.0, 0.5])
ORDER_C = np.array([0.0, 0.0, 2.6, 0.5, 0.0, -1.6])


DATASET_CONFIGS = [
    ("toy_last_event", "last_event", 1101),
    ("toy_persistent_mean", "persistent_mean", 2202),
    ("toy_rare_spike", "rare_spike", 3303),
    ("toy_ordered_pattern", "ordered_pattern", 4404),
]


def append_event(rows, source, destination, timestamp, feature, label):
    rows.append({
        "u": int(source),
        "i": int(destination),
        "ts": float(timestamp),
        "label": int(label),
        **{f"f{j}": float(feature[j]) for j in range(FEATURE_DIM)},
    })


def jitter(rng, feature, scale):
    return np.asarray(feature, dtype=float) + rng.normal(0.0, scale, FEATURE_DIM)


def trial_source_and_label(trial_idx):
    # Eight recurring devices appear once per segment. The positive window rotates by segment,
    # so every device is positive in some trials and negative in others.
    segment_size = 8
    device_offset = trial_idx % segment_size
    segment = trial_idx // segment_size
    source = int(TRIAL_DEVICES[device_offset])
    positive_offsets = {(2 * segment + offset) % segment_size for offset in range(4)}
    label = int(device_offset in positive_offsets)
    return source, label


def neutral_query_feature(rng):
    return jitter(rng, NEUTRAL, 0.025)


def filler_feature(rng):
    return jitter(rng, NEUTRAL, 0.08)


def filler_edge(rng, trial_source):
    available = FILLER_DEVICES[FILLER_DEVICES != trial_source]
    source = int(rng.choice(available))
    if rng.random() < 0.8:
        destination = int(rng.choice(SERVERS))
    else:
        destination = int(rng.choice(available))
    return source, destination, filler_feature(rng)


def last_event_history(rng, label):
    history = []
    old_signal = LAST_NEG * 3.0 if label else LAST_POS * 3.0
    last_signal = LAST_POS if label else LAST_NEG

    for idx in range(HISTORY_SIZE - 1):
        if idx % 4 == 0:
            feature = jitter(rng, old_signal, 0.12)
        elif idx % 4 == 1:
            feature = jitter(rng, NEUTRAL, 0.35)
        else:
            feature = jitter(rng, old_signal * 0.55, 0.18)
        history.append(feature)

    # Only the final message carries the correct state. Older messages are deliberately stale.
    history.append(jitter(rng, last_signal, 0.08))
    return history


def persistent_mean_history(rng, label):
    majority = PERSISTENT_POS if label else PERSISTENT_NEG
    minority = PERSISTENT_NEG if label else PERSISTENT_POS
    history = []

    # The class is the average of many weak messages. The final message and the
    # high-magnitude distractors are intentionally non-diagnostic.
    message_types = (
        ["majority"] * 74
        + ["minority"] * 42
        + ["distractor"] * 11
        + ["neutral"]
    )
    rng.shuffle(message_types)
    message_types[-1] = "neutral"

    for message_type in message_types:
        if message_type == "majority":
            history.append(jitter(rng, majority, 0.16))
        elif message_type == "minority":
            history.append(jitter(rng, minority, 0.16))
        elif message_type == "distractor":
            sign = 1.0 if rng.random() < 0.5 else -1.0
            history.append(jitter(rng, sign * PERSISTENT_DISTRACTOR, 0.08))
        else:
            history.append(jitter(rng, NEUTRAL, 0.18))

    return history


def rare_spike_history(rng, label):
    history = [jitter(rng, NEUTRAL, 0.45) for _ in range(HISTORY_SIZE)]
    spike_positions = rng.choice(np.arange(8, HISTORY_SIZE - 8), size=4, replace=False)
    spike = SPIKE_POS if label else SPIKE_NEG

    for pos in spike_positions:
        history[pos] = jitter(rng, spike, 0.06)

    # Keep the last message non-diagnostic so LastMessageAggregator cannot solve it.
    history[-1] = jitter(rng, NEUTRAL, 0.20)
    return history


def ordered_pattern_history(rng, label):
    history = [jitter(rng, NEUTRAL, 0.18) for _ in range(HISTORY_SIZE)]

    # Same multiset and same final token in both classes. Only temporal order differs:
    # positive = A -> B -> C; negative = B -> A -> C.
    pattern = [ORDER_A, ORDER_B, ORDER_C] if label else [ORDER_B, ORDER_A, ORDER_C]
    positions = [HISTORY_SIZE - 9, HISTORY_SIZE - 5, HISTORY_SIZE - 1]

    for pos, token in zip(positions, pattern):
        history[pos] = jitter(rng, token, 0.05)

    # Add balanced older decoys so simple counts and magnitudes stay uninformative.
    decoy_positions = rng.choice(np.arange(8, HISTORY_SIZE - 16), size=9, replace=False)
    decoys = [ORDER_A, ORDER_B, ORDER_C] * 3
    rng.shuffle(decoys)
    for pos, token in zip(decoy_positions, decoys):
        history[pos] = jitter(rng, token, 0.10)

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
    raise ValueError(f"Unknown isolated toy mode: {mode}")


def generate_isolated_dataset(mode, seed, num_trials=DEFAULT_TRIALS):
    rng = np.random.default_rng(seed)
    rows = []
    timestamp = 0.0

    for trial_idx in range(num_trials):
        source, label = trial_source_and_label(trial_idx)
        query_destination = int(SERVERS[trial_idx % len(SERVERS)])
        history = build_history(rng, mode, label)

        # Batch A: all messages for the same source node. These are the messages
        # the aggregator compresses before the query batch starts.
        for feature in history:
            timestamp += 1.0
            destination = int(rng.choice(SERVERS))
            append_event(rows, source, destination, timestamp, feature, 0)

        # Batch B, first half: neutral current-edge features. The answer is in memory/history.
        for _ in range(QUERY_SIZE):
            timestamp += 1.0
            append_event(rows, source, query_destination, timestamp, neutral_query_feature(rng), label)

        # Batch B, second half: neutral filler for other nodes, preserving batch alignment.
        for _ in range(FILLER_SIZE):
            timestamp += 1.0
            filler_source, filler_destination, feature = filler_edge(rng, source)
            append_event(rows, filler_source, filler_destination, timestamp, feature, 0)

    return pd.DataFrame(rows)


def write_preprocessed_dataset(df, data_name, output_dir):
    ml_df = df[["u", "i", "ts", "label"]].copy()
    ml_df["idx"] = np.arange(len(ml_df))

    # Match the existing non-bipartite toy preprocessing: reserve row 0 for padding.
    ml_df["u"] = ml_df["u"] + 1
    ml_df["i"] = ml_df["i"] + 1
    ml_df["idx"] = ml_df["idx"] + 1

    features = df[[f"f{idx}" for idx in range(FEATURE_DIM)]].to_numpy(dtype=float)
    features = np.vstack([np.zeros((1, FEATURE_DIM)), features])

    max_node_idx = int(max(ml_df["u"].max(), ml_df["i"].max()))
    node_features = np.zeros((max_node_idx + 1, NODE_FEATURE_DIM))

    ml_df.to_csv(output_dir / f"ml_{data_name}.csv", index=False)
    np.save(output_dir / f"ml_{data_name}.npy", features)
    np.save(output_dir / f"ml_{data_name}_node.npy", node_features)


def validate_splits(data_name, df):
    val_time, test_time = np.quantile(df["ts"], [0.70, 0.85])
    splits = {
        "train": df[df["ts"] <= val_time],
        "val": df[(df["ts"] > val_time) & (df["ts"] <= test_time)],
        "test": df[df["ts"] > test_time],
    }
    split_text = []
    for split_name, split_df in splits.items():
        positives = int(split_df["label"].sum())
        rate = positives / max(1, len(split_df))
        split_text.append(f"{split_name}={positives}/{len(split_df)} ({rate:.1%})")

    total_pos = int(df["label"].sum())
    print(
        f"{data_name}: {len(df)} edges | {total_pos} positives "
        f"({total_pos / len(df):.1%}) | " + " | ".join(split_text)
    )


def generate_all(output_dir, num_trials):
    output_dir.mkdir(parents=True, exist_ok=True)
    for data_name, mode, seed in DATASET_CONFIGS:
        df = generate_isolated_dataset(mode=mode, seed=seed, num_trials=num_trials)
        df.to_csv(output_dir / f"{data_name}.csv", index=False)
        write_preprocessed_dataset(df, data_name, output_dir)
        validate_splits(data_name, df)


def parse_args():
    parser = argparse.ArgumentParser("Generate isolated TGN aggregator toy datasets")
    parser.add_argument("--output-dir", default="./data",
                        help="Directory where raw and preprocessed toy datasets are written.")
    parser.add_argument("--num-trials", type=int, default=DEFAULT_TRIALS,
                        help="Number of two-batch trials per dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all(Path(args.output_dir), args.num_trials)
