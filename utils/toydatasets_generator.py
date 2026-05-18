import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ======================
# Shared Parameters
# ======================
num_nodes = 120
feature_dim = 6
num_servers = 15

servers = np.arange(num_servers)
devices = np.arange(num_servers, num_nodes)


# ======================
# Core Generator
# ======================
def generate_dataset(num_edges, sparsity_level="sparse", mode="mixed"):
    """
    Generate a synthetic temporal graph dataset.

    Parameters
    ----------
    num_edges : int
        Approximate number of edges to generate before attack injection.
    sparsity_level : str
        "dense" or "sparse" — controls drop probability, time scale, burst rate.
    mode : str
        Structural mode that determines which aggregator is favoured:
        - "markov"       → favours LastMessageAggregator
        - "iid"          → favours MeanMessageAggregator
        - "mixed_noise"  → favours WeightedMessageAggregator
        - "sequential"   → favours AttentionMessageAggregator

    Returns
    -------
    pd.DataFrame with columns: u, i, ts, label, f0..f(feature_dim-1)
    """

    sources, destinations = [], []
    timestamps, features, labels = [], [], []

    t = 0

    # ----------------------
    # Sparsity config
    # ----------------------
    if sparsity_level == "dense":
        drop_prob = 0.05
        time_scale = 2.0
        burst_prob = 0.3
        device_server_prob = 0.85
    else:  # sparse
        drop_prob = 0.4
        time_scale = 10.0
        burst_prob = 0.1
        device_server_prob = 0.6

    # ----------------------
    # Mode-level overrides on top of sparsity
    # ----------------------
    if mode == "markov":
        # Isolated single events, no bursts — each interaction fully replaces prior state.
        # Long gaps between interactions so the "last" message is always the only relevant one.
        burst_prob = 0.02
        time_scale = max(time_scale, 15.0)

    elif mode == "iid":
        # Dense, uniform traffic — all messages equally informative, no temporal structure.
        burst_prob = 0.0
        time_scale = min(time_scale, 3.0)
        drop_prob = min(drop_prob, 0.05)

    elif mode == "mixed_noise":
        # Moderate interaction rate. Attack signal will be buried in junk messages.
        burst_prob = max(burst_prob, 0.2)
        time_scale = min(time_scale, 5.0)

    elif mode == "sequential":
        # Moderate rate. Attack unfolds in strict ordered phases across time.
        burst_prob = max(burst_prob, 0.15)
        time_scale = min(time_scale, 4.0)

    # ----------------------
    # Node state for markov mode (continuous drift)
    # ----------------------
    # Each node carries a latent state vector that drifts over time.
    # The most recent interaction's features reflect the current state.
    node_state = np.zeros((num_nodes, feature_dim))

    # ----------------------
    # Normal interaction factory
    # ----------------------
    def normal_interaction(step=0):
        if np.random.rand() < device_server_prob:
            u = np.random.choice(devices)
            v = np.random.choice(servers)
        else:
            u = np.random.choice(devices)
            v = np.random.choice(devices)

        if mode == "markov":
            # Features are the node's current drifted state — the latest message
            # is the only one that matters because previous states are obsolete.
            node_state[u] += np.random.normal(0, 0.3, feature_dim)
            node_state[v] += np.random.normal(0, 0.3, feature_dim)
            feat = (node_state[u] + node_state[v]) / 2.0 + np.random.normal(0, 0.05, feature_dim)

        elif mode == "iid":
            # Pure i.i.d. Gaussian — no node state, no temporal dependency.
            # Every message is equally informative; mean is the correct aggregator.
            feat = np.random.normal(0, 1, feature_dim)

        elif mode == "mixed_noise":
            # 75% of normal traffic is pure noise (random features).
            # 25% carries a weak directional signal correlated with node identity.
            if np.random.rand() < 0.75:
                feat = np.random.normal(0, 1, feature_dim)   # junk
            else:
                feat = np.random.normal(0.2, 0.3, feature_dim)  # weak signal

        elif mode == "sequential":
            # Normal traffic is unremarkable baseline noise.
            feat = np.random.normal(0, 1, feature_dim)

        else:
            feat = np.random.normal(0, 1, feature_dim)

        return u, v, feat, 0

    # ----------------------
    # Generate base traffic
    # ----------------------
    for i in range(num_edges):
        if np.random.rand() < drop_prob:
            continue

        u, v, feat, lbl = normal_interaction(step=i)

        if np.random.rand() < (1 - burst_prob):
            delta = np.random.exponential(time_scale)
        else:
            delta = np.random.exponential(0.5)

        t += delta

        sources.append(u)
        destinations.append(v)
        timestamps.append(t)
        features.append(feat)
        labels.append(lbl)

    features_arr = np.array(features)
    n = len(sources)

    # ----------------------
    # Attack injection (mode-specific)
    # ----------------------

    # Spread attack episodes across the whole timeline so the default
    # 70/15/15 time split contains positives in train, validation, and test.
    attack_windows = [(0.18, 0.30), (0.54, 0.66), (0.73, 0.82), (0.88, 0.96)]

    if mode == "markov":
        # ------------------------------------------------------------------
        # Abrupt state-change attacks.
        # A few devices change state sharply near the end of each episode.
        # Prior messages in the same batch are deliberately stale, so the
        # latest message remains the clearest signal.
        # ------------------------------------------------------------------
        attackers = np.random.choice(devices, size=3, replace=False)
        targets = np.random.choice(servers, size=3, replace=False)
        base_state = np.array([3.0, -3.0, 3.0, -3.0, 3.0, -3.0])

        for episode, (start_frac, end_frac) in enumerate(attack_windows):
            start = int(start_frac * n)
            end = max(start + 1, int(end_frac * n))
            attacker = attackers[episode % len(attackers)]
            target = targets[episode % len(targets)]
            compromised_state = base_state * (1 if episode % 2 == 0 else -1)

            for i in range(start, min(end, n)):
                if np.random.rand() < 0.35:
                    continue
                sources[i] = attacker
                destinations[i] = target
                progress = (i - start) / max(1, end - start)

                if progress < 0.45:
                    # Still mostly benign, making older messages misleading.
                    features_arr[i] = np.random.normal(0, 0.35, feature_dim)
                    labels[i] = 0
                else:
                    features_arr[i] = compromised_state + np.random.normal(0, 0.12, feature_dim)
                    labels[i] = 1

    elif mode == "iid":
        # ------------------------------------------------------------------
        # Slow statistical anomaly.
        # Many weakly shifted samples appear in every split. No individual
        # event is very strong, but the average over repeated messages is.
        # ------------------------------------------------------------------
        attack_nodes = np.random.choice(devices, size=12, replace=False)
        target_servers = np.random.choice(servers, size=3, replace=False)

        for episode, (start_frac, end_frac) in enumerate(attack_windows):
            start = int(start_frac * n)
            end = max(start + 1, int(end_frac * n))
            target = target_servers[episode % len(target_servers)]

            for i in range(start, min(end, n)):
                if np.random.rand() < 0.35:
                    continue
                progress = (i - start) / max(1, end - start)
                sources[i] = np.random.choice(attack_nodes)
                destinations[i] = target
                features_arr[i] = np.random.normal(0.45, 0.85, feature_dim)
                labels[i] = int(progress > 0.40)

    elif mode == "mixed_noise":
        # ------------------------------------------------------------------
        # Rare signal buried in noise.
        # Multiple compromised devices send mostly junk messages, with enough
        # high-signal anomalous messages in every split to make evaluation
        # stable across repeated runs.
        # ------------------------------------------------------------------
        attackers = np.random.choice(devices, size=3, replace=False)
        targets = np.random.choice(servers, size=3, replace=False)
        junk_ratio = 0.82
        signal_vecs = [
            np.array([4.0, -3.5, 3.2, -2.8, 3.8, -3.2]),
            np.array([-3.6, 3.4, -3.1, 2.9, -3.5, 3.0]),
            np.array([3.2, 3.0, -3.4, -3.2, 2.8, -2.9]),
        ]

        for episode, (start_frac, end_frac) in enumerate(attack_windows):
            start = int(start_frac * n)
            end = max(start + 1, int(end_frac * n))
            attacker = attackers[episode % len(attackers)]
            target = targets[episode % len(targets)]
            signal_vec = signal_vecs[episode % len(signal_vecs)]

            for i in range(start, min(end, n)):
                if np.random.rand() < 0.15:
                    continue
                progress = (i - start) / max(1, end - start)
                sources[i] = attacker
                destinations[i] = target

                if np.random.rand() < junk_ratio:
                    features_arr[i] = np.random.normal(0, 1.5, feature_dim)
                    labels[i] = 0
                else:
                    features_arr[i] = signal_vec + np.random.normal(0, 0.08, feature_dim)
                    labels[i] = int(progress > 0.15)

    elif mode == "sequential":
        # ------------------------------------------------------------------
        # Three-phase ordered attack: probe → escalate → exfiltrate.
        # The ordered sequence appears across train/val/test. Wrong-order
        # decoys use the same phase signatures but should not be positive.
        # Several attacker-target pairs prevent one pair from dominating.
        # ------------------------------------------------------------------
        attackers = np.random.choice(devices, size=4, replace=False)
        targets = np.random.choice(servers, size=4, replace=False)

        phase_features = {
            "probe":       np.array([ 2.0,  0.0,  0.0, -1.0,  0.5,  0.0]),
            "escalate":    np.array([ 0.0,  2.5,  0.0,  0.0, -1.0,  0.5]),
            "exfiltrate":  np.array([ 0.0,  0.0,  3.0,  0.5,  0.0, -2.0]),
        }

        attack_start = int(0.12 * n)
        attack_end = int(0.98 * n)
        n_cycles = 10
        cycle_length = max(1, (attack_end - attack_start) // n_cycles)

        for cycle in range(n_cycles):
            base = attack_start + cycle * cycle_length
            phase_size = max(4, cycle_length // 5)
            attacker = attackers[cycle % len(attackers)]
            target = targets[cycle % len(targets)]
            ordered = cycle % 4 != 1
            phases = ["probe", "escalate", "exfiltrate"] if ordered else [
                "probe", "exfiltrate", "escalate"
            ]

            for phase_idx, phase in enumerate(phases):
                phase_start = base + phase_idx * (phase_size + max(1, phase_size // 2))
                phase_end = min(phase_start + phase_size, n)
                for i in range(phase_start, phase_end):
                    if np.random.rand() < 0.20:
                        continue
                    sources[i] = attacker
                    destinations[i] = target
                    features_arr[i] = phase_features[phase] + np.random.normal(0, 0.15, feature_dim)
                    labels[i] = int(ordered and phase == "exfiltrate" and cycle >= 1)

    # ----------------------
    # Build DataFrame
    # ----------------------
    df = pd.DataFrame({
        'u': sources,
        'i': destinations,
        'ts': timestamps,
        'label': labels
    })

    for j in range(feature_dim):
        df[f'f{j}'] = features_arr[:, j]

    return df


# ======================
# Generate all datasets
# ======================
Path("./data").mkdir(parents=True, exist_ok=True)

configs = [
    ("markov",      5000, "sparse"),
    ("iid",         6000, "dense"),
    ("mixed_noise", 5000, "sparse"),
    ("sequential",  6000, "dense"),
]

for mode, num_edges, sparsity in configs:
    df = generate_dataset(num_edges=num_edges, sparsity_level=sparsity, mode=mode)
    path = f"./data/toy_{mode}.csv"
    df.to_csv(path, index=False)
    n_pos = df['label'].sum()
    print(f"toy_{mode}: {df.shape[0]} edges | {n_pos} positives ({100*n_pos/len(df):.1f}%) | sparsity={sparsity}")
