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

    if mode == "markov":
        # ------------------------------------------------------------------
        # Single abrupt state-change attack.
        # One node's state jumps discontinuously — only the LAST message
        # reflects the compromised state. Prior messages are misleading.
        # Mean aggregation would dilute the signal; last sees it clearly.
        # ------------------------------------------------------------------
        attacker = np.random.choice(devices)
        target = np.random.choice(servers)

        attack_start = int(0.6 * n)
        attack_end = int(0.9 * n)

        # Before attack: attacker has benign drifted state
        # After attack: abrupt jump to anomalous feature region
        compromised_state = np.array([3.0, -3.0, 3.0, -3.0, 3.0, -3.0])

        for i in range(attack_start, attack_end):
            if np.random.rand() < 0.4:
                continue
            sources[i] = attacker
            destinations[i] = target
            # Features jump to compromised region and stay there — Markovian
            progress = (i - attack_start) / (attack_end - attack_start)
            features_arr[i] = compromised_state * progress + np.random.normal(0, 0.1, feature_dim)
            if progress > 0.6:
                labels[i] = 1

    elif mode == "iid":
        # ------------------------------------------------------------------
        # Slow statistical anomaly — attack features are a shifted Gaussian.
        # No temporal structure; the shift is detectable only in aggregate.
        # Mean aggregation naturally accumulates the shifted signal over
        # many messages. Last is unreliable (one noisy sample); attention
        # has nothing sequential to attend to.
        # ------------------------------------------------------------------
        attack_nodes = np.random.choice(devices, size=12, replace=False)
        target_server = np.random.choice(servers)

        attack_start = int(0.3 * n)
        attack_end = int(0.85 * n)

        for i in range(attack_start, attack_end):
            if np.random.rand() < 0.5:
                continue
            sources[i] = np.random.choice(attack_nodes)
            destinations[i] = target_server
            # Consistently shifted features — detectable only in aggregate mean
            features_arr[i] = np.random.normal(0.4, 0.9, feature_dim)
            if i > int(0.6 * n):
                labels[i] = 1

    elif mode == "mixed_noise":
        # ------------------------------------------------------------------
        # Signal buried in noise. Attack node sends mostly junk messages
        # with occasional high-signal messages.
        # WeightedMessageAggregator should learn to upweight the signal
        # messages and suppress the junk via the learned scorer.
        # Mean dilutes the signal; last may pick a junk message.
        # ------------------------------------------------------------------
        attacker = np.random.choice(devices)
        target = np.random.choice(servers)
        JUNK_RATIO = 0.80   # 80% of attacker's messages are pure noise

        attack_start = int(0.25 * n)
        attack_end = int(0.90 * n)

        signal_vec = np.array([2.5, -2.0, 1.8, -1.5, 2.2, -1.9])  # strong directional signal

        for i in range(attack_start, attack_end):
            if np.random.rand() < 0.3:
                continue
            sources[i] = attacker
            destinations[i] = target

            if np.random.rand() < JUNK_RATIO:
                # Junk message — pure noise, no label
                features_arr[i] = np.random.normal(0, 1.5, feature_dim)
                labels[i] = 0
            else:
                # Signal message — consistent anomalous direction
                features_arr[i] = signal_vec + np.random.normal(0, 0.2, feature_dim)
                if i > int(0.55 * n):
                    labels[i] = 1

    elif mode == "sequential":
        # ------------------------------------------------------------------
        # Three-phase ordered attack: probe → escalate → exfiltrate.
        # Each phase produces a DISTINCT feature signature.
        # The label fires only when all three phases have occurred in order.
        # Attention can learn to weight the "escalate" message highly only
        # when preceded by "probe". Mean and last see only one phase at a time.
        # ------------------------------------------------------------------
        attacker = np.random.choice(devices)
        target = np.random.choice(servers)

        # Distinct feature vectors for each phase — orthogonal in feature space
        phase_features = {
            "probe":       np.array([ 2.0,  0.0,  0.0, -1.0,  0.5,  0.0]),
            "escalate":    np.array([ 0.0,  2.5,  0.0,  0.0, -1.0,  0.5]),
            "exfiltrate":  np.array([ 0.0,  0.0,  3.0,  0.5,  0.0, -2.0]),
        }

        n_cycles = 6   # number of full probe→escalate→exfiltrate cycles
        cycle_length = (n - int(0.2 * n)) // n_cycles
        attack_base = int(0.2 * n)

        for cycle in range(n_cycles):
            base = attack_base + cycle * cycle_length
            phase_size = max(1, cycle_length // 4)

            # Phase 1: probe (early in cycle)
            for i in range(base, min(base + phase_size, n)):
                if np.random.rand() < 0.35:
                    continue
                sources[i] = attacker
                destinations[i] = target
                features_arr[i] = phase_features["probe"] + np.random.normal(0, 0.15, feature_dim)
                labels[i] = 0   # probe alone is not yet an attack

            # Phase 2: escalate (middle of cycle)
            mid = base + phase_size
            for i in range(mid, min(mid + phase_size, n)):
                if np.random.rand() < 0.35:
                    continue
                sources[i] = attacker
                destinations[i] = target
                features_arr[i] = phase_features["escalate"] + np.random.normal(0, 0.15, feature_dim)
                labels[i] = 0   # escalate alone is not yet an attack

            # Phase 3: exfiltrate (late in cycle — label fires here)
            late = base + 2 * phase_size
            for i in range(late, min(late + phase_size, n)):
                if np.random.rand() < 0.35:
                    continue
                sources[i] = attacker
                destinations[i] = target
                features_arr[i] = phase_features["exfiltrate"] + np.random.normal(0, 0.15, feature_dim)
                if cycle >= 2:   # label only fires after pattern has repeated enough
                    labels[i] = 1

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