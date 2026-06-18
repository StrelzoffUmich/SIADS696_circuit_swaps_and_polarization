"""
===========================================================
Quantum Circuit Graph Regression with Multi-Task Targets
===========================================================

This script processes QASM files of quantum circuits, constructs
graph representations capturing qubit interactions and gate features,
and trains a GraphSAGE-based neural network to predict two targets:
  1. log1p(bare_routed_2q): log-transformed number of 2-qubit gates after routing
  2. polarization: some secondary property from a second dataset

Design decisions, hyperparameters, and feature engineering choices
are explained throughout.

Author: Steven Flack
===========================================================
"""

import os
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, JumpingKnowledge

from sklearn.metrics import r2_score, mean_absolute_error
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from pull_gate_types import extract_gates_with_legacy_support

##############################################################################
# PATHS & CACHE SETUP
##############################################################################

# CSVs with complementary targets: bare_routed_2q and polarization
csv_path1 = "/Users/steventf/Desktop/school/milestoneII/MQT_Loader_v3_2/MQT_Loader/results/run_2026-06-01_18-24-53/full_dataset.csv"
csv_path2 = "/Users/steventf/Desktop/school/milestoneII/MQT_Loader_v3_2/MQT_Loader/results/run_2026-06-01_18-31-57/full_dataset.csv"

# QASM source files directory
qasm_files_path = (
    "/Users/steventf/Desktop/school/milestoneII/MQT_Loader_v3_2/MQT_Loader/qasm"
)

# Cache for pre-built graph list (speeds up reruns)
cache_path = "/Users/steventf/Desktop/school/milestoneII/MQT_Loader_v3_2/MQT_Loader/cache/graph_list.pt"
os.makedirs(os.path.dirname(cache_path), exist_ok=True)

##############################################################################
# LOAD + MERGE DATA
##############################################################################

# Load first dataset (bare_routed_2q)
df1 = pd.read_csv(csv_path1)

# Load second dataset (polarization)
df2 = pd.read_csv(csv_path2)

# Merge on file name (inner join ensures only circuits present in both CSVs)
df = pd.merge(
    df1[["file", "bare_routed_2q"]],
    df2[["file", "polarization"]],
    on="file",
    how="inner",
)

# Drop any rows with missing target values
df = df.dropna(subset=["bare_routed_2q", "polarization"])
print("Dataset size:", len(df))

# Optionally save merged dataset for reproducibility
merged_csv_path = ".../swap_polarization_full_merged_dataset.csv"
df.to_csv(merged_csv_path, index=False)

##############################################################################
# GATE FEATURES
##############################################################################

# Extract the set of all gates in all circuits (legacy-safe) with script I wrote
global_gate_list = extract_gates_with_legacy_support(qasm_files_path)
num_gate_features = len(global_gate_list)  # total unique gates -> feature vector size


def gate_counts_to_feature_vector(gate_list):
    """
    Convert a list of gate names to a numeric feature vector.

    Each entry counts occurrences of a global gate.
    log1p is applied to reduce the impact of very frequent gates.
    Returns a torch tensor of shape [num_gate_features].
    """
    vec = np.zeros(num_gate_features, dtype=np.float32)
    gate2idx = {g: i for i, g in enumerate(global_gate_list)}

    for g in gate_list:
        if g in gate2idx:
            vec[gate2idx[g]] += 1

    # log1p = log(1 + x) avoids log(0) issues and compresses counts
    return torch.tensor(np.log1p(vec), dtype=torch.float)


##############################################################################
# GRAPH CONSTRUCTION
##############################################################################


def build_interaction_graph(file_name, qasm_dir, bare_routed_2q, polarization):
    """
    Build a PyTorch Geometric Data object representing a circuit graph.

    Nodes = qubits
    Edges = 2-qubit gates (undirected)
    Node features = [degree, weighted_degree, normalized_degree]
    Graph features = global properties (depth, width, gate counts, etc.)
    Multi-target output = [log1p(bare_routed_2q), polarization]
    """

    # Load quantum circuit from QASM
    qasm_path = os.path.join(qasm_dir, file_name)
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    dag = circuit_to_dag(qc)

    n_qubits = qc.num_qubits
    counter = Counter()

    # Count all 2-qubit interactions (for edges)
    for gate in qc.data:
        if len(gate.qubits) == 2:
            q1 = qc.find_bit(gate.qubits[0]).index
            q2 = qc.find_bit(gate.qubits[1]).index
            counter[tuple(sorted((q1, q2)))] += 1

    # Build undirected edge list and corresponding weights
    edge_list = []
    edge_weights = []
    for (u, v), w in counter.items():
        edge_list += [[u, v], [v, u]]  # make edges bidirectional
        edge_weights += [w, w]

    edge_index = (
        torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        if edge_list
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_weight = (
        torch.tensor(edge_weights, dtype=torch.float)
        if edge_weights
        else torch.empty((0,), dtype=torch.float)
    )

    # NODE FEATURES
    x = torch.zeros((n_qubits, 3), dtype=torch.float)
    degrees = [0] * n_qubits
    weighted = [0] * n_qubits

    for (u, v), w in counter.items():
        degrees[u] += 1
        degrees[v] += 1
        weighted[u] += w
        weighted[v] += w

    for i in range(n_qubits):
        x[i, 0] = degrees[i]  # number of connections
        x[i, 1] = weighted[i]  # sum of edge weights
        x[i, 2] = degrees[i] / max(1, n_qubits - 1)  # normalized degree

    # TARGET: multi-task learning
    y = torch.tensor(
        [
            np.log1p(bare_routed_2q),  # log-transform 2Q gates
            polarization,  # secondary target
        ],
        dtype=torch.float,
    )

    # GATE FEATURE VECTOR
    gate_list = [node.op.name for node in dag.op_nodes()]
    gate_attr = gate_counts_to_feature_vector(gate_list).view(1, -1)  # 2D tensor

    # GRAPH-LEVEL FEATURES
    depth = qc.depth()  # circuit depth
    num_ops = len(dag.op_nodes())  # total number of operations
    num_2q = sum(1 for node in dag.op_nodes() if len(node.qargs) == 2)  # 2Q gate count
    dag_width = dag.width()  # max parallelism
    density = (
        (2 * len(counter)) / (n_qubits * (n_qubits - 1)) if n_qubits > 1 else 0
    )  # graph connectivity
    avg_degree = np.mean(degrees)  # average node degree

    graph_attr = torch.tensor(
        [n_qubits, depth, num_ops, num_2q, dag_width, density, avg_degree],
        dtype=torch.float,
    ).view(1, -1)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        gate_attr=gate_attr,
        graph_attr=graph_attr,
    )


##############################################################################
# CACHE MANAGEMENT
##############################################################################


def get_cache_hash(csv_path, qasm_dir):
    """
    Compute a hash representing the state of CSV + QASM files.
    If files change, cache should be rebuilt.
    """
    m = hashlib.md5()
    m.update(str(os.path.getmtime(csv_path)).encode())

    for f in sorted(os.listdir(qasm_dir)):
        fpath = os.path.join(qasm_dir, f)
        if os.path.isfile(fpath):
            m.update(f.encode())
            m.update(str(os.path.getmtime(fpath)).encode())

    return m.hexdigest()


cache_hash_path = cache_path + ".hash"
current_hash = get_cache_hash(merged_csv_path, qasm_files_path)

# Load or build cached graph dataset
if os.path.exists(cache_path) and os.path.exists(cache_hash_path):
    with open(cache_hash_path) as f:
        saved = f.read().strip()

    if saved == current_hash:
        print("Loading cache...")
        graph_list = torch.load(cache_path, weights_only=False)
    else:
        print("Rebuilding cache...")
        graph_list = [
            build_interaction_graph(
                r["file"], qasm_files_path, r["bare_routed_2q"], r["polarization"]
            )
            for _, r in df.iterrows()
        ]
        torch.save(graph_list, cache_path)
        open(cache_hash_path, "w").write(current_hash)
else:
    print("Building cache...")
    graph_list = [
        build_interaction_graph(
            r["file"], qasm_files_path, r["bare_routed_2q"], r["polarization"]
        )
        for _, r in df.iterrows()
    ]
    torch.save(graph_list, cache_path)
    open(cache_hash_path, "w").write(current_hash)

# Ensure y is always shape [2] (multi-target)
for g in graph_list:
    g.y = g.y.view(2)

print("Graphs:", len(graph_list))

##############################################################################
# TRAIN/VAL/TEST SPLIT
##############################################################################

np.random.seed(42)  # deterministic shuffling
perm = np.random.permutation(len(graph_list))

# 70% train, 15% val, 15% test
n_train = int(0.7 * len(graph_list))
n_val = int(0.15 * len(graph_list))

train = [graph_list[i] for i in perm[:n_train]]
val = [graph_list[i] for i in perm[n_train : n_train + n_val]]
test = [graph_list[i] for i in perm[n_train + n_val :]]

# DataLoader for mini-batch training
train_loader = DataLoader(
    train, batch_size=32, shuffle=True
)  # batch size tuned after trial and error
val_loader = DataLoader(val, batch_size=32)
test_loader = DataLoader(test, batch_size=32)

##############################################################################
# MODEL DEFINITION
##############################################################################


class RoutingGraphSAGE(torch.nn.Module):
    """
    GraphSAGE-based multi-task regression model.

    Architecture:
    - 3 SAGEConv layers with residual connections
    - LayerNorm after each conv
    - Jumping Knowledge (concatenate) for multi-scale aggregation
    - Node features pooled via global_mean_pool
    - Concatenated with gate_attr and graph_attr
    - Fully connected head for final regression (2 outputs)
    """

    def __init__(
        self,
        in_channels,
        hidden=256,
        dropout=0.3,
        num_gate_features=num_gate_features,
        num_graph_features=7,
    ):

        super().__init__()

        # Graph convolution layers, 3 seemed to work the best
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)

        # LayerNorm stabilizes training (mainly used for with residuals)
        self.norm1 = torch.nn.LayerNorm(hidden)
        self.norm2 = torch.nn.LayerNorm(hidden)
        self.norm3 = torch.nn.LayerNorm(hidden)

        # JumpingKnowledge: concatenate multi-layer representations
        self.jk = JumpingKnowledge(mode="cat")

        # Fully connected layers
        # hidden*3 because of concatenation of x1,x2,x3
        self.lin1 = torch.nn.Linear(
            hidden * 3 + num_gate_features + num_graph_features, 512
        )
        self.lin2 = torch.nn.Linear(512, 2)  # two regression targets

        # Dropout probability
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1st conv + LayerNorm + ReLU + Dropout
        x1 = F.relu(self.norm1(self.conv1(x, edge_index)))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # 2nd conv + residual + LayerNorm + ReLU + Dropout
        x2 = F.relu(self.norm2(self.conv2(x1, edge_index)))
        x2 = x2 + x1
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # 3rd conv + residual
        x3 = F.relu(self.norm3(self.conv3(x2, edge_index)))
        x3 = x3 + x2

        # Concatenate multi-layer features via JumpingKnowledge
        x = self.jk([x1, x2, x3])

        # Aggregate node-level features to graph-level
        x = global_mean_pool(x, batch)

        # Concatenate with precomputed gate-level and graph-level features
        gate = data.gate_attr.to(x.device)
        graph = data.graph_attr.to(x.device)
        x = torch.cat([x, gate, graph], dim=1)

        # Fully connected head
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


##############################################################################
# TRAINING SETUP
##############################################################################

# just in case I run on GL with cuda card
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# in_channels=3 because x = torch.zeros((n_qubits, 3), dtype=torch.float) in graph
# degree (num distinct qubits it interacts with)
# weighted degree (how often those interactions occur)
# normalized degree (scale-invariant connectivity measure)
model = RoutingGraphSAGE(in_channels=3).to(device)

# Adam optimizer with very small learning rate (fine-tuning)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)
criterion = torch.nn.SmoothL1Loss()  # robust to outliers


def evaluate(loader):
    """
    Evaluate model on a DataLoader.
    Returns: (loss, (r2_0, r2_1), (mae_0, mae_1))
    """
    model.eval()
    preds, ys = [], []
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y.view(-1, 2))
            total += loss.item() * batch.num_graphs
            preds.append(pred.cpu())
            ys.append(batch.y.cpu())

    preds = torch.cat(preds).view(-1, 2).numpy()
    ys = torch.cat(ys).view(-1, 2).numpy()

    return (
        total / len(loader.dataset),
        (r2_score(ys[:, 0], preds[:, 0]), r2_score(ys[:, 1], preds[:, 1])),
        (
            mean_absolute_error(ys[:, 0], preds[:, 0]),
            mean_absolute_error(ys[:, 1], preds[:, 1]),
        ),
    )


##############################################################################
# TRAIN LOOP
##############################################################################

best = float("inf")
patience = (
    100  # early stopping patience, longer, but just because learning steps are small
)
no_improve = 0

for epoch in range(1, 5000):
    model.train()
    total = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.view(-1, 2))
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs

    train_loss = total / len(train_loader.dataset)
    val_loss, val_r2, val_mae = evaluate(val_loader)

    # unpack metrics
    r2_0, r2_1 = val_r2
    mae_0, mae_1 = val_mae

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:04d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val R2: [{r2_0:.3f}, {r2_1:.3f}] | "
            f"Val MAE: [{mae_0:.4f}, {mae_1:.4f}]"
        )

    # Early stopping
    if val_loss < best:
        best = val_loss
        best_state = model.state_dict()
        no_improve = 0
    else:
        no_improve += 1

    if no_improve > patience:
        print("Early stopping")
        break

# load best model
model.load_state_dict(best_state)

##############################################################################
# TEST
##############################################################################

test_loss, test_r2, test_mae = evaluate(test_loader)
test_r2_1, test_r2_2 = test_r2
test_mae_1, test_mae_2 = test_mae

print("\nFINAL RESULTS")
print(
    f"Test Loss {test_loss:.4f} | "
    f"Test R2 [log(bare_routed_2q), polarization]: [{test_r2_1:.4f}, {test_r2_2:.4f}] | "
    f"Test MAE [log(bare_routed_2q), polarization]: [{test_mae_1:.4f}, {test_mae_2:.4f}] "
)
