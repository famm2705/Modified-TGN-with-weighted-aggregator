import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Header_Length",
    "Protocol_Type",
    "Duration",
    "Rate",
    "Srate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot_sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot_size",
    "IAT",
    "Number",
    "Magnitude",
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
]


ATTACK_DESCRIPTIONS = {
    "abrupt_last": (
        "One device is benign for most of a batch and changes state only at the end. "
        "The next batch contains assessment labels, so the most recent message matters most."
    ),
    "persistent_mean": (
        "Many weak anomaly messages from the same devices accumulate over a batch. "
        "No single row is decisive, so averaging the messages is favored."
    ),
    "rare_weighted": (
        "A compromised device sends mostly junk or benign-looking traffic with a few strong "
        "malformed/spoofing signals. The final messages are intentionally junk, so last is weak."
    ),
    "ordered_attention": (
        "The positive label depends on an ordered probe -> escalate -> exfiltrate pattern. "
        "Negative blocks contain the same phases in the wrong order or with missing context."
    ),
}


NODE_ROLES = {
    "iomt_devices": list(range(0, 40)),
    "gateway": 40,
    "mqtt_broker": 41,
    "cloud_service": 42,
    "monitor": 43,
    "malicious_pc": 44,
    "ble_phone": 45,
}


def _positive(value):
    return float(max(value, 0.0))


def _feature_vector(rng, kind="normal", intensity=1.0, protocol="tcp", profile="active"):
    """Create one CICIoMT-like numeric flow feature vector."""
    f = {name: 0.0 for name in FEATURE_COLUMNS}

    profile_rate = {
        "power": (8.0, 3.0),
        "idle": (12.0, 5.0),
        "active": (80.0, 25.0),
        "interaction": (160.0, 45.0),
    }
    rate_mu, rate_sigma = profile_rate.get(profile, profile_rate["active"])

    if protocol == "tcp":
        f["Protocol_Type"] = 6.0
        f["TCP"] = 1.0
        f["ack_flag_number"] = rng.binomial(1, 0.65)
        f["psh_flag_number"] = rng.binomial(1, 0.20)
        f["ack_count"] = _positive(rng.normal(0.25, 0.12))
    elif protocol == "udp":
        f["Protocol_Type"] = 17.0
        f["UDP"] = 1.0
    elif protocol == "icmp":
        f["Protocol_Type"] = 1.0
        f["ICMP"] = 1.0
    elif protocol == "arp":
        f["Protocol_Type"] = 0.0
        f["ARP"] = 1.0
        f["IPv"] = 0.0
        f["LLC"] = 1.0
    elif protocol == "ble":
        f["Protocol_Type"] = 17.0
        f["UDP"] = 1.0
    else:
        f["Protocol_Type"] = 6.0
        f["TCP"] = 1.0

    if protocol != "arp":
        f["IPv"] = 1.0
        f["LLC"] = 1.0

    f["Duration"] = _positive(rng.normal(64.0, 3.0))
    f["Rate"] = _positive(rng.normal(rate_mu, rate_sigma))
    f["Srate"] = f["Rate"] * _positive(rng.normal(1.0, 0.03))
    f["Header_Length"] = _positive(rng.normal(110.0, 40.0))
    f["Tot_sum"] = _positive(rng.normal(520.0, 70.0))
    f["Min"] = _positive(rng.normal(48.0, 5.0))
    f["Max"] = _positive(rng.normal(90.0, 45.0))
    f["AVG"] = _positive(rng.normal(62.0, 12.0))
    f["Std"] = _positive(rng.normal(6.0, 4.0))
    f["Tot_size"] = _positive(rng.normal(62.0, 12.0))
    f["IAT"] = _positive(rng.normal(90.0, 20.0))
    f["Number"] = _positive(rng.normal(9.5, 0.8))
    f["Magnitude"] = _positive(rng.normal(10.0, 1.4))
    f["Radius"] = _positive(rng.normal(4.0, 3.0))
    f["Covariance"] = _positive(rng.normal(120.0, 80.0))
    f["Variance"] = min(max(rng.normal(0.08, 0.08), 0.0), 1.0)
    f["Weight"] = _positive(rng.normal(140.0, 18.0))

    if kind == "syn_flood":
        f["TCP"] = 1.0
        f["UDP"] = 0.0
        f["Protocol_Type"] = 6.0
        f["syn_flag_number"] = 1.0
        f["ack_flag_number"] = 0.0
        f["syn_count"] = _positive(rng.normal(7.5 * intensity, 1.2))
        f["Rate"] = _positive(rng.normal(900.0 * intensity, 110.0))
        f["Srate"] = f["Rate"] * _positive(rng.normal(1.0, 0.02))
        f["IAT"] = _positive(rng.normal(2.0, 0.8))
        f["Header_Length"] = _positive(rng.normal(1800.0 * intensity, 250.0))
        f["Tot_sum"] = _positive(rng.normal(1250.0, 160.0))
    elif kind == "udp_flood":
        f["TCP"] = 0.0
        f["UDP"] = 1.0
        f["Protocol_Type"] = 17.0
        f["Rate"] = _positive(rng.normal(1000.0 * intensity, 130.0))
        f["Srate"] = f["Rate"] * _positive(rng.normal(1.0, 0.02))
        f["IAT"] = _positive(rng.normal(1.8, 0.7))
        f["Tot_sum"] = _positive(rng.normal(1500.0, 230.0))
        f["Max"] = _positive(rng.normal(900.0, 120.0))
    elif kind == "icmp_flood":
        f["TCP"] = 0.0
        f["ICMP"] = 1.0
        f["Protocol_Type"] = 1.0
        f["Rate"] = _positive(rng.normal(850.0 * intensity, 100.0))
        f["Srate"] = f["Rate"]
        f["IAT"] = _positive(rng.normal(2.4, 1.0))
        f["Tot_sum"] = _positive(rng.normal(900.0, 130.0))
    elif kind == "recon":
        f["TCP"] = 1.0
        f["Protocol_Type"] = 6.0
        f["syn_flag_number"] = rng.binomial(1, 0.75)
        f["ack_flag_number"] = rng.binomial(1, 0.15)
        f["syn_count"] = _positive(rng.normal(1.6 * intensity, 0.3))
        f["Rate"] = _positive(rng.normal(210.0 * intensity, 35.0))
        f["Srate"] = f["Rate"]
        f["IAT"] = _positive(rng.normal(15.0, 4.0))
        f["Header_Length"] = _positive(rng.normal(420.0, 65.0))
    elif kind == "arp_spoof":
        f["TCP"] = 0.0
        f["UDP"] = 0.0
        f["ARP"] = 1.0
        f["IPv"] = 0.0
        f["Protocol_Type"] = 0.0
        f["Rate"] = _positive(rng.normal(420.0 * intensity, 60.0))
        f["Srate"] = f["Rate"]
        f["IAT"] = _positive(rng.normal(5.0, 1.5))
        f["Tot_sum"] = _positive(rng.normal(420.0, 30.0))
        f["Min"] = _positive(rng.normal(42.0, 2.0))
        f["Max"] = _positive(rng.normal(64.0, 4.0))
        f["AVG"] = _positive(rng.normal(48.0, 2.0))
    elif kind == "mqtt_malformed":
        f["TCP"] = 1.0
        f["Protocol_Type"] = 6.0
        f["psh_flag_number"] = 1.0
        f["ack_flag_number"] = 1.0
        f["Rate"] = _positive(rng.normal(320.0 * intensity, 45.0))
        f["Srate"] = f["Rate"]
        f["IAT"] = _positive(rng.normal(8.0, 2.0))
        f["Std"] = _positive(rng.normal(95.0, 18.0))
        f["Radius"] = _positive(rng.normal(180.0, 30.0))
        f["Covariance"] = _positive(rng.normal(6000.0, 850.0))
        f["Variance"] = min(max(rng.normal(0.75, 0.08), 0.0), 1.0)
    elif kind == "weak_mqtt":
        f["TCP"] = 1.0
        f["Protocol_Type"] = 6.0
        f["Rate"] = _positive(rng.normal(190.0 * intensity, 20.0))
        f["Srate"] = f["Rate"]
        f["IAT"] = _positive(rng.normal(20.0, 4.0))
        f["Std"] = _positive(rng.normal(22.0, 5.0))
        f["Covariance"] = _positive(rng.normal(550.0, 110.0))
        f["Variance"] = min(max(rng.normal(0.20, 0.05), 0.0), 1.0)
    elif kind == "junk":
        for name in FEATURE_COLUMNS:
            if name in {"TCP", "UDP", "ARP", "ICMP", "IPv", "LLC"}:
                continue
            f[name] = _positive(rng.normal(0.0, 1.6))
        f["TCP"] = rng.binomial(1, 0.45)
        f["UDP"] = 1.0 - f["TCP"]
        f["Protocol_Type"] = 6.0 if f["TCP"] else 17.0
        f["IPv"] = 1.0
        f["LLC"] = 1.0

    return np.array([f[name] for name in FEATURE_COLUMNS], dtype=np.float32)


class EventWriter:
    def __init__(self, rng):
        self.rng = rng
        self.rows = []
        self.timestamp = 0.0

    def add(self, source, destination, features, label, burst=True):
        step = self.rng.exponential(0.15 if burst else 3.0)
        self.timestamp += step
        self.rows.append(
            [int(source), int(destination), float(self.timestamp), int(label), *features.tolist()]
        )

    def add_normal(self, source=None, destination=None, label=0, profile=None):
        devices = NODE_ROLES["iomt_devices"]
        services = [
            NODE_ROLES["gateway"],
            NODE_ROLES["mqtt_broker"],
            NODE_ROLES["cloud_service"],
            NODE_ROLES["monitor"],
        ]
        source = int(self.rng.choice(devices)) if source is None else int(source)
        destination = int(self.rng.choice(services)) if destination is None else int(destination)
        protocol = self.rng.choice(["tcp", "udp", "icmp"], p=[0.70, 0.22, 0.08])
        profile = profile or self.rng.choice(["idle", "active", "interaction"], p=[0.25, 0.55, 0.20])
        self.add(
            source,
            destination,
            _feature_vector(self.rng, "normal", protocol=protocol, profile=profile),
            label=label,
            burst=False,
        )

    def dataframe(self):
        return pd.DataFrame(self.rows, columns=["u", "i", "ts", "label", *FEATURE_COLUMNS])


def _fill_block_with_normal(writer, remaining):
    for _ in range(remaining):
        writer.add_normal()


def _generate_abrupt_last(num_events, batch_size, rng):
    writer = EventWriter(rng)
    attacker = int(rng.choice(NODE_ROLES["iomt_devices"]))
    target = NODE_ROLES["gateway"]
    pending_alert = False
    blocks = int(np.ceil(num_events / batch_size))

    for block in range(blocks):
        used = 0
        if pending_alert:
            for _ in range(max(8, batch_size // 4)):
                writer.add_normal(source=attacker, destination=NODE_ROLES["monitor"], label=1, profile="active")
                used += 1
            pending_alert = False

        attack_block = block > 3 and block % 4 in {1, 2}
        if attack_block:
            benign_count = max(1, batch_size - used - 10)
            for _ in range(benign_count):
                writer.add_normal(source=attacker, destination=NODE_ROLES["mqtt_broker"], label=0, profile="interaction")
                used += 1
            for _ in range(min(10, batch_size - used)):
                writer.add(attacker, target, _feature_vector(rng, "syn_flood", intensity=1.2), label=0)
                used += 1
            pending_alert = True

        _fill_block_with_normal(writer, batch_size - used)

    return writer.dataframe().iloc[:num_events].copy()


def _generate_persistent_mean(num_events, batch_size, rng):
    writer = EventWriter(rng)
    weak_nodes = rng.choice(NODE_ROLES["iomt_devices"], size=8, replace=False)
    target = NODE_ROLES["mqtt_broker"]
    pending_nodes = set()
    blocks = int(np.ceil(num_events / batch_size))

    for block in range(blocks):
        used = 0
        for node in sorted(pending_nodes):
            for _ in range(max(2, batch_size // 32)):
                writer.add_normal(source=node, destination=NODE_ROLES["monitor"], label=1, profile="active")
                used += 1
        pending_nodes = set()

        attack_block = block > 2 and block % 3 != 0
        if attack_block:
            per_node = max(4, (batch_size - used) // (len(weak_nodes) + 2))
            for node in weak_nodes:
                for _ in range(per_node):
                    writer.add(node, target, _feature_vector(rng, "weak_mqtt", intensity=1.0), label=0)
                    used += 1
                pending_nodes.add(int(node))

        _fill_block_with_normal(writer, batch_size - used)

    return writer.dataframe().iloc[:num_events].copy()


def _generate_rare_weighted(num_events, batch_size, rng):
    writer = EventWriter(rng)
    attacker = int(rng.choice(NODE_ROLES["iomt_devices"]))
    target = NODE_ROLES["mqtt_broker"]
    pending_alert = False
    blocks = int(np.ceil(num_events / batch_size))

    for block in range(blocks):
        used = 0
        if pending_alert:
            for _ in range(max(8, batch_size // 4)):
                writer.add_normal(source=attacker, destination=NODE_ROLES["monitor"], label=1, profile="active")
                used += 1
            pending_alert = False

        attack_block = block > 2 and block % 4 in {1, 2}
        if attack_block:
            attack_events = max(20, batch_size - used - 8)
            strong_positions = set(rng.choice(np.arange(4, attack_events - 4), size=4, replace=False))
            for pos in range(attack_events):
                if pos in strong_positions:
                    kind = rng.choice(["mqtt_malformed", "arp_spoof"])
                    dst = target if kind == "mqtt_malformed" else NODE_ROLES["gateway"]
                    writer.add(attacker, dst, _feature_vector(rng, kind, intensity=1.3), label=0)
                else:
                    writer.add(attacker, target, _feature_vector(rng, "junk"), label=0)
                used += 1
            for _ in range(min(8, batch_size - used)):
                writer.add(attacker, target, _feature_vector(rng, "junk"), label=0)
                used += 1
            pending_alert = True

        _fill_block_with_normal(writer, batch_size - used)

    return writer.dataframe().iloc[:num_events].copy()


def _phase_feature(rng, phase):
    if phase == "probe":
        return _feature_vector(rng, "recon", intensity=1.0)
    if phase == "escalate":
        return _feature_vector(rng, "arp_spoof", intensity=1.0)
    if phase == "exfiltrate":
        return _feature_vector(rng, "mqtt_malformed", intensity=1.0)
    return _feature_vector(rng, "normal")


def _generate_ordered_attention(num_events, batch_size, rng):
    writer = EventWriter(rng)
    attacker = int(rng.choice(NODE_ROLES["iomt_devices"]))
    target_by_phase = {
        "probe": NODE_ROLES["gateway"],
        "escalate": NODE_ROLES["gateway"],
        "exfiltrate": NODE_ROLES["mqtt_broker"],
    }
    pending_alert = False
    blocks = int(np.ceil(num_events / batch_size))

    for block in range(blocks):
        used = 0
        if pending_alert:
            for _ in range(max(8, batch_size // 4)):
                writer.add_normal(source=attacker, destination=NODE_ROLES["monitor"], label=1, profile="active")
                used += 1
            pending_alert = False

        sequence_block = block > 2 and block % 2 == 1
        if sequence_block:
            positive_order = block % 4 == 1
            phases = ["probe", "escalate", "exfiltrate"] if positive_order else ["probe", "exfiltrate", "escalate"]
            phase_repeats = max(3, (batch_size - used) // 12)
            for phase in phases:
                for _ in range(phase_repeats):
                    writer.add(attacker, target_by_phase[phase], _phase_feature(rng, phase), label=0)
                    used += 1
                for _ in range(max(1, phase_repeats // 2)):
                    writer.add_normal()
                    used += 1
            pending_alert = positive_order

        _fill_block_with_normal(writer, batch_size - used)

    return writer.dataframe().iloc[:num_events].copy()


GENERATORS = {
    "abrupt_last": _generate_abrupt_last,
    "persistent_mean": _generate_persistent_mean,
    "rare_weighted": _generate_rare_weighted,
    "ordered_attention": _generate_ordered_attention,
}


def write_tgn_files(df, dataset_name, output_dir, standardize=True, mode=None, description=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"{dataset_name}.csv"
    ml_csv_path = output_dir / f"ml_{dataset_name}.csv"
    edge_path = output_dir / f"ml_{dataset_name}.npy"
    node_path = output_dir / f"ml_{dataset_name}_node.npy"
    metadata_path = output_dir / f"{dataset_name}_metadata.json"

    df = df.sort_values("ts").reset_index(drop=True)
    df.to_csv(raw_path, index=False)

    graph_df = df[["u", "i", "ts", "label"]].copy()
    graph_df["u"] = graph_df["u"].astype(int) + 1
    graph_df["i"] = graph_df["i"].astype(int) + 1
    graph_df["idx"] = np.arange(1, len(graph_df) + 1, dtype=np.int64)
    graph_df.to_csv(ml_csv_path, index=False)

    edge_features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    feature_mean = edge_features.mean(axis=0)
    feature_std = edge_features.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    if standardize:
        edge_features = (edge_features - feature_mean) / feature_std
    edge_features = np.vstack([np.zeros((1, edge_features.shape[1]), dtype=np.float32), edge_features])
    np.save(edge_path, edge_features.astype(np.float32))

    max_node_id = int(max(graph_df["u"].max(), graph_df["i"].max()))
    node_features = np.zeros((max_node_id + 1, 172), dtype=np.float32)
    np.save(node_path, node_features)

    metadata = {
        "dataset": dataset_name,
        "mode": mode,
        "description": description,
        "raw_csv": str(raw_path),
        "ml_csv": str(ml_csv_path),
        "edge_features": str(edge_path),
        "node_features": str(node_path),
        "num_events": int(len(df)),
        "num_positive_labels": int(df["label"].sum()),
        "positive_label_rate": float(df["label"].mean()),
        "feature_columns": FEATURE_COLUMNS,
        "features_standardized_in_ml_npy": bool(standardize),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "node_roles": NODE_ROLES,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return metadata


def generate_all(num_events, batch_size, output_dir, seed, standardize=True):
    results = []
    for offset, (mode, generator) in enumerate(GENERATORS.items()):
        rng = np.random.default_rng(seed + offset)
        dataset_name = f"iomt_{mode}"
        df = generator(num_events=num_events, batch_size=batch_size, rng=rng)
        metadata = write_tgn_files(
            df,
            dataset_name,
            output_dir,
            standardize=standardize,
            mode=mode,
            description=ATTACK_DESCRIPTIONS[mode],
        )
        results.append(metadata)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CICIoMT-shaped extreme temporal graph datasets for TGN aggregators."
    )
    parser.add_argument("--num-events", type=int, default=12000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Design batch size used to place same-node message bursts. Train with the same --bs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Store raw edge feature scales in ml_*.npy instead of standardized features.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generated = generate_all(
        num_events=args.num_events,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        seed=args.seed,
        standardize=not args.no_standardize,
    )
    for item in generated:
        print(
            f"{item['dataset']}: {item['num_events']} events, "
            f"{item['num_positive_labels']} positives "
            f"({100.0 * item['positive_label_rate']:.2f}%)"
        )
