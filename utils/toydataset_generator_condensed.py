import numpy as np
import pandas as pd

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
# Core Generator Function
# ======================
def generate_dataset(num_edges, sparsity_level="dense"):

    sources, destinations = [], []
    timestamps, features, labels = [], [], []

    t = 0

    # ----------------------
    # Config based on sparsity
    # ----------------------
    if sparsity_level == "dense":
        drop_prob = 0.05
        time_scale = 2.0
        burst_prob = 0.3
        device_server_prob = 0.85

    elif sparsity_level == "sparse":
        drop_prob = 0.4
        time_scale = 10.0
        burst_prob = 0.1
        device_server_prob = 0.6

    # ----------------------
    # Normal interaction
    # ----------------------
    def normal_interaction():
        if np.random.rand() < device_server_prob:
            u = np.random.choice(devices)
            v = np.random.choice(servers)
        else:
            u = np.random.choice(devices)
            v = np.random.choice(devices)

        feat = np.random.normal(0, 1, feature_dim)
        return u, v, feat, 0

    # ----------------------
    # 1) Generate base traffic
    # ----------------------
    for i in range(num_edges):

        # Drop interactions (controls sparsity)
        if np.random.rand() < drop_prob:
            continue

        u, v, feat, lbl = normal_interaction()

        # Time dynamics
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

    # Convert to lists for indexing
    features_arr = np.array(features)

    # ----------------------
    # 2) Inject ATTACKS
    # ----------------------

    n = len(sources)

    # ---- A) Slow DDoS ----
    target_server = np.random.choice(servers)
    attack_nodes = np.random.choice(devices, size=10, replace=False)

    for i in range(int(0.1*n), int(0.4*n)):
        if np.random.rand() < (0.3 if sparsity_level=="dense" else 0.6):
            continue

        sources[i] = np.random.choice(attack_nodes)
        destinations[i] = target_server
        features_arr[i] += np.random.normal(0.15, 0.05, feature_dim)

        if i > int(0.3*n):
            labels[i] = 1

    # ---- B) Lateral Movement ----
    attacker = np.random.choice(devices)
    visited = set()

    for i in range(int(0.45*n), int(0.7*n)):
        if np.random.rand() < (0.25 if sparsity_level=="dense" else 0.5):
            continue

        new_target = np.random.choice(devices)
        sources[i] = attacker
        destinations[i] = new_target
        features_arr[i] += np.random.normal(0.1, 0.05, feature_dim)

        visited.add(new_target)

        if len(visited) > 20 and i > int(0.65*n):
            labels[i] = 1

    # ---- C) Data Exfiltration ----
    exf_node = np.random.choice(devices)
    target_server = np.random.choice(servers)

    for i in range(int(0.7*n), int(0.95*n)):
        if np.random.rand() < (0.35 if sparsity_level=="dense" else 0.6):
            continue

        sources[i] = exf_node
        destinations[i] = target_server

        drift = (i - int(0.7*n)) / (0.25*n)
        features_arr[i] += drift * np.array([0.3, 0.3, 0.3, 0, 0, 0])

        if drift > 0.75:
            labels[i] = 1

    # ----------------------
    # 3) Create DataFrame
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
# 4) Generate BOTH datasets
# ======================
dense_df = generate_dataset(num_edges=6000, sparsity_level="dense")
sparse_df = generate_dataset(num_edges=3000, sparsity_level="sparse")

# ======================
# 5) Save
# ======================
dense_df.to_csv('./data/toy_dense.csv', index=False)
sparse_df.to_csv('./data/toy_sparse.csv', index=False)

print("Datasets created:")
print("Dense:", dense_df.shape)
print("Sparse:", sparse_df.shape)