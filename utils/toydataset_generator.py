import numpy as np
import pandas as pd

np.random.seed(42)

# ======================
# Parameters
# ======================
num_nodes = 120
num_edges = 7500
feature_dim = 6

num_servers = 15
servers = np.arange(num_servers)
devices = np.arange(num_servers, num_nodes)

sources = []
destinations = []
timestamps = []
features = []
labels = []

t = 0

# ======================
# Helper: normal interaction (sparser)
# ======================
def normal_interaction():
    if np.random.rand() < 0.85:
        u = np.random.choice(devices)
        v = np.random.choice(servers)
    else:
        u = np.random.choice(devices)
        v = np.random.choice(devices)
    
    feat = np.random.normal(0, 1, feature_dim)
    return u, v, feat, 0


# ======================
# 1) Generate base traffic (sparse)
# ======================
for i in range(num_edges):
    u, v, feat, lbl = normal_interaction()
    
    # make most interactions far apart
    if np.random.rand() < 0.8:
        delta = np.random.exponential(5.0)  # long gap
    else:
        delta = np.random.exponential(0.5)  # bursts
    
    t += delta
    
    sources.append(u)
    destinations.append(v)
    timestamps.append(t)
    features.append(feat)
    labels.append(lbl)

# ======================
# 2) Inject ATTACKS (keep bursts for attacks)
# ======================

# ---- A) Slow DDoS (bursty in sparse background) ----
target_server = np.random.choice(servers)
attack_nodes = np.random.choice(devices, size=12, replace=False)

for i in range(500, 3000):
    if np.random.rand() < 0.3:
        continue
    sources[i] = np.random.choice(attack_nodes)
    destinations[i] = target_server
    features[i] += np.random.normal(0.15, 0.05, feature_dim)
    if i > 2200:
        labels[i] = 1

# ---- B) Lateral movement ----
attacker = np.random.choice(devices)
visited = set()

for i in range(3200, 5500):
    if np.random.rand() < 0.25:
        continue
    new_target = np.random.choice(devices)
    sources[i] = attacker
    destinations[i] = new_target
    features[i] += np.random.normal(0.1, 0.05, feature_dim)
    visited.add(new_target)
    if len(visited) > 25 and i > 4800:
        labels[i] = 1

# ---- C) Data exfiltration ----
exf_node = np.random.choice(devices)
target_server = np.random.choice(servers)

for i in range(5500, 7400):
    if np.random.rand() < 0.35:
        continue
    sources[i] = exf_node
    destinations[i] = target_server
    drift = (i - 5500) / 1900
    features[i] += drift * np.array([0.3, 0.3, 0.3, 0, 0, 0])
    if drift > 0.75:
        labels[i] = 1

# ======================
# 3) Convert to dataframe
# ======================
df = pd.DataFrame({
    'u': sources,
    'i': destinations,
    'ts': timestamps,
    'label': labels
})

features = np.array(features)

for j in range(feature_dim):
    df[f'f{j}'] = features[:, j]

# ======================
# 4) Save
# ======================
df.to_csv('./data/toy_sparse.csv', index=False)