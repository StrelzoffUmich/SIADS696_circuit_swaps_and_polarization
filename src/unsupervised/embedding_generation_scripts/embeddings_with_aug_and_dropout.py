"""
This script turns quantum circuits (QASM files) into graph data and trains Graph Neural Networks
to learn useful circuit embeddings using contrastive learning (GraphCL).

It supports multiple encoders (GCN, GAT, GraphSAGE) and builds graph representations where
nodes are quantum gates and edges represent dependencies between operations.

The training process creates augmented versions of each graph to help the model learn
robust, invariant representations. After training, it extracts and saves graph-level
embeddings along with training loss history for later analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    global_mean_pool,
    JumpingKnowledge,
)
from torch_geometric.utils import dropout_edge

####################################################
# DEFINE ENCODER CLASSES

# these encoders encode the quantium circuit graphs into node embeddings
# GCN uses fixed normalized neighbor aggregation
# GAT uses attention weights over neighbors
# GraphSAGE samples and learns aggregators for inductive use
####################################################


####################################################
####################################################
# GCN ENCODER
# Learns node embeddings by aggregating neighbor information via graph convolutions on a graph structure.
####################################################
class GCNEncoder(torch.nn.Module):
    """
    Graph Convolutional Network encoder with:
    - 3 GCN layers
        - idea from https://kumo.ai/pyg/layers/gcn-conv/
            - layer 1 for 1-hop neighbors (local structure)
            - layer 2 for larger local to global neighbors
            - layer 3 for global neighbors
    - Residual connections
        - idea is then to
            - add the previous layer features to the current layer to preserve earlier information
            - improve gradient flow
            - educe over-smoothing in deeper GCN stacks.
    """

    def __init__(self, in_channels, hidden=128, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)

    def forward(self, x, edge_index, edge_attr=None):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)

        return x


####################################################
# GAT ENCODER
# Learns node embeddings using attention-weighted neighbor aggregation, allowing the model to focus on the most relevant neighbors in the graph.
####################################################
class GATEncoder(torch.nn.Module):
    """
    Graph Attention Network encoder with:
    - 3 GCN layers
        - idea from https://kumo.ai/pyg/layers/gcn-conv/
            - layer 1 for 1-hop neighbors (local structure)
            - layer 2 for larger local to global neighbors
            - layer 3 for global neighbors
    - Residual connections
        - idea is then to
            - add the previous layer features to the current layer to preserve earlier information
            - improve gradient flow
            - educe over-smoothing in deeper GCN stacks.
    """

    def __init__(self, in_channels, hidden=128, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        #######################
        # GAT layers
        # heads = number of attention heads (multi-head message passing)
        # concat = False --> average heads (keeps the output dim = hidden)
        # dropout = attention dropout for regularization
        #######################
        self.conv1 = GATConv(
            in_channels, hidden, heads=4, concat=False, dropout=self.dropout
        )
        self.conv2 = GATConv(
            hidden, hidden, heads=4, concat=False, dropout=self.dropout
        )
        self.conv3 = GATConv(
            hidden, hidden, heads=1, concat=False, dropout=self.dropout
        )

    def forward(self, x, edge_index, edge_attr=None):

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)

        return x


####################################################
# GRAPHSAGE ENCODER
# Learns node embeddings by sampling and aggregating neighbor features, enabling scalable inductive representation learning on large graphs.
####################################################
class GraphSAGEEncoder(torch.nn.Module):
    """
    GraphSAGE encoder with:
    - 3 GCN layers
        - idea from https://kumo.ai/pyg/layers/gcn-conv/
            - layer 1 for 1-hop neighbors (local structure)
            - layer 2 for larger local to global neighbors
            - layer 3 for global neighbors
    - Residual connections
        - idea is then to
            - add the previous layer features to the current layer to preserve earlier information
            - improve gradient flow
            - educe over-smoothing in deeper GCN stacks.
    """

    def __init__(self, in_channels, hidden=128, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)

    def forward(self, x, edge_index, edge_attr=None):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)

        return x


####################################################
# GRAPHCL WRAPPER
# Learns graph-level embeddings using a GNN encoder followed by a projection head for contrastive learning.
####################################################
class GraphCL(torch.nn.Module):
    """
    GraphCL model for graph-level contrastive learning.
        - following ideas from https://diveintographs.readthedocs.io/en/latest/tutorials/sslgraph.html

    - Encoder (GraphSAGE/GAT/GCN):
        - Learns node embeddings via message passing on the input graph
    - Global pooling:
        - Aggregates node embeddings into a single graph representation (mean pooling)
    - Projection head:
        - MLP (Multi-Layer Perceptron) that maps graph embeddings into a latent space for contrastive learning
        - includes Linear layers, BatchNorm, ReLU, and Dropout for stability and regularization
    """

    def __init__(self, encoder, dropout=0.3):
        super().__init__()

        self.encoder = encoder

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 128),
        )

    def forward(self, data):

        x = self.encoder(data.x, data.edge_index, data.edge_attr)
        x = global_mean_pool(x, data.batch)

        return self.proj(x)


###############################################
# LOAD DATA AND. BUILD GRAPHS

# Converts raw QASM files into graph representations for GNN training:
# 1) builds a gate vocabulary from all circuits
# 2) converts each circuit into node/edge graphs
# 3) assembles a clean dataset by filtering invalid samples
###############################################


def build_gate_vocab(file_list, qasm_dir):
    """
    Builds a vocabulary of quantum gates from QASM files by scanning all gate
    operations in the dataset and assigning each unique gate a unique integer ID.
    """

    vocab = {}

    def load_qasm(file):
        """
        opens the qasm file to work with
        """
        with open(qasm_dir / file, "r") as f:
            return f.read()

    for f in file_list:
        qasm = load_qasm(f)

        # splits each line of the qasm file into a distince part to learn a "vocab"
        for line in qasm.splitlines():
            line = line.strip()
            if not line or line.startswith(("include", "OPENQASM")):
                continue

            # is the new gate is not in the vocab, add it
            gate = line.split()[0]
            if gate not in vocab:
                vocab[gate] = len(vocab)

    return vocab


def qasm_to_graph(qasm_str, gate_vocab):
    """
    Converts a QASM circuit into a PyTorch Geometric graph representation.
        - Nodes represent quantum operations (gates)
        - Node features encode gate type and qubit count
        - Edges represent execution dependencies between operations
        - Edge features encode special gate types (e.g., CX, measure, barrier)
            - could add more, but with more, computation time increases greatly.
            - these specifically are key QASM operations (CX, measure, barrier)
                - since they strongly affect circuit structure, entanglement, and execution constraints.
    """

    # using qiskit, load in QASM file and convert to DAG
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    dag = circuit_to_dag(qc)

    # list the defined operations of the circuit
    ops = list(dag.op_nodes())

    #######################
    # Node features: [gate_id, is_two_qubit]
    #######################
    x = []
    for node in ops:
        gate_id = gate_vocab.get(node.op.name, 0)
        is_2q = 1 if len(node.qargs) >= 2 else 0
        x.append([gate_id, is_2q])

    x = torch.tensor(x, dtype=torch.float)

    #######################
    # Edge features
    #######################
    edge_index = []
    edge_attr = []

    # for each operation, go through the predecessors and attributes to build a neighborhood around the node
    for i, node in enumerate(ops):
        for pred in dag.predecessors(node):
            if pred in ops:
                j = ops.index(pred)
                edge_index.append([j, i])
                edge_attr.append(
                    [
                        float(node.op.name == "cx"),
                        float(node.op.name == "measure"),
                        float(node.op.name == "barrier"),
                    ]
                )

    #######################
    # handel empty cases
    #######################
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def load_graph_dataset(df, qasm_dir, gate_vocab):
    """
    Builds a PyTorch Geometric dataset from dataframe of QASM files.
        - uses build_gate_vocab() and qasm_to_graph() to:
            - Read QASM files from disk
            - Convert each circuit into a graph representation
            - Filter out invalid or unreadable samples
            - Return an aligned graph list and cleaned dataframe
    """

    graphs = []
    valid_rows = []

    def load_qasm(file):
        with open(qasm_dir / file, "r") as f:
            return f.read()

    for i, row in df.iterrows():
        try:
            qasm = load_qasm(row["file"])
            graphs.append(qasm_to_graph(qasm, gate_vocab))
            valid_rows.append(i)
        except:
            continue

    df = df.loc[valid_rows].reset_index(drop=True)

    return graphs, df


###############################################
# AUGMENTATION
# This section uses graph contrastive learning augmentations to combine
# structural corruption, feature masking, subcircuit dropout,
# and light noise to help generate diverse but semantically consistent circuit views
# per ideas introduces in https://arxiv.org/html/2604.23700v1 (Quantum Circuit Cutting: Complexity and Optimization)
###############################################


def augment(data):
    """
    Idea is from idea is from https://arxiv.org/html/2604.23700v1 and their thoughts on learning through augmentation of circuits
    Creates a random graph augmentation view for contrastive learning.
    """

    # randomly drop 20% of edges to simulate structural devation
    edge_index, mask = dropout_edge(data.edge_index, p=0.2)

    # keep edge attr that share retrained edges
    edge_attr = data.edge_attr[mask]

    # add a small ammount of noise for feature robustness
    x = data.x + torch.randn_like(data.x) * 0.01

    # augmented graph, edge indexes, and arrtibutes in the batch
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=data.batch)


###############################################
# CONTRASTIVE LOSS
###############################################
def contrastive_loss(z1, z2, temperature=0.2):
    """
    Followed ideas in: https://medium.com/@mlshark/infonce-explained-in-details-and-implementations-902f28199ce6

    This computes InfoNCE-style contrastive loss between two augmented graph views.
        - Encourages matching of embeddings from the same graph across views (augment function)
        - seperates embeddings from different graphs in the batch
        - Uses the cosine similarity with temperature scaling for stability
    """
    #######################
    # First normalize the embeddings using cosine similarity
    #######################
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    #######################
    # calculate a similarity matric between all pairs
    # temperature affects smoothness or sharpness of the similarity distribution
    #######################
    logits = torch.mm(z1, z2.T) / temperature

    #######################
    # align only the positive pairs by index
    #######################
    labels = torch.arange(z1.size(0), device=z1.device)

    #######################
    # calculate the cross_entropy over the similarity scores
    #######################
    return F.cross_entropy(logits, labels)


###############################################
# MAIN PIPELINE FUNCTION
###############################################
def run_graphcl_pipeline(
    csv_path,
    qasm_dir,
    output_path,
    # 128 balances stable contrastive loss estimates with memory limits while seeming to provide enough negatives.
    batch_size=128,
    # looking online, this seems like a standard Adam rate for stable yet responsive convergence in noisy GNN contrastive training.
    lr=1e-3,
    # 128 gives sufficient capacity for circuit structure encoding without overfitting our small dataset.
    hidden_dim=128,
    # gives some wiggle room without stopping too early or late
    patience=15,
    # Dropout is set relatively high (0.3) because with a small dataset and strong graph augmentations it
    # helps stabilize contrastive loss and helps prevent overfitting without harming representation quality.
    DROPOUT=0.3,
):
    """
    Full Graph Contrastive Learning pipeline for quantum circuits.

    Steps:
        1. Load dataset CSV
        2. Build gate vocabulary
        3. Convert QASM to a graph dataset
        4. Train GraphCL model
        5. Extract embeddings
        6. Save embeddings to CSV

    For each GNN (GCN, GAT, GraphSAGE) Writes:
        1. CSV with embeddings and metadata
        2. CSV with epochs and contrastive_loss
    """

    ###############################################
    # Load dataset
    ###############################################
    df = pd.read_csv(csv_path)
    qasm_dir = Path(qasm_dir)

    ###############################################
    # Build vocabulary of quantum gates from dataset
    ###############################################
    gate_vocab = build_gate_vocab(df["file"], qasm_dir)

    ###############################################
    # Convert QASM files into graph representations
    ###############################################
    graphs, df = load_graph_dataset(df, qasm_dir, gate_vocab)

    print(f"Loaded {len(graphs)} graphs")

    # PyG DataLoader batches multiple graphs for training and shuffles to help randomize more
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True, drop_last=True)

    # option to use gpu if ever available to speed things up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################################
    # Define the available graph encoders from near start of code
    ###############################################
    ENCODERS = {
        "gcn": GCNEncoder,
        "gat": GATEncoder,
        "graphsage": GraphSAGEEncoder,
    }

    for model_name, encoder_cls in ENCODERS.items():
        print("\n")
        print("#" * 60)
        print("Training:", model_name)
        print("#" * 60)

        # GraphCL is the encoder + projection head (MLP) for contrastive learning
        model = GraphCL(encoder_cls(in_channels=2), dropout=DROPOUT).to(device)

        # use ADAM as the optimizer for contrastive pretraining
        # per: https://apxml.com/courses/graph-neural-networks-gnns/chapter-3-gnn-training-complexities/gnn-optimization-strategies
        # set the learning rate as the same as the model learning rate
        # after guessing and checking from 1 to 1e-8 this seemed to work well
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # track out loss history to see how well the embeddings are learning
        loss_history = []

        # set up early stopping variables
        best_loss = float("inf")
        epochs_without_improve = 0
        # 15 seems to be the sweet spot, where it allows for some bumps but does't die early or go on forever
        patience = 15
        epoch = 0

        ###############################################
        # contrastive learning loop
        ###############################################
        # Learn graph embeddings using augmentation views
        while True:
            model.train()
            total_loss = 0

            for data in loader:
                data = data.to(device)

                ###############################################
                # create two augmented views of same graph for representation invariance under slight deviations
                ###############################################
                aug1 = augment(data)
                aug2 = augment(data)

                ###############################################
                # encode both augmented views
                ###############################################
                z1 = model(aug1)
                z2 = model(aug2)

                ###############################################
                # calculate the contrapositive loss
                # ideas from https://medium.com/@mlshark/infonce-explained-in-details-and-implementations-902f28199ce6
                ###############################################
                loss = contrastive_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            ###############################################
            # calculate our average loss for the epoch, write to loss_history, and print out result
            ###############################################
            avg_loss = total_loss / len(loader)
            loss_history.append({"epoch": epoch, "loss": avg_loss})
            print(f"{model_name} Epoch {epoch} Loss {avg_loss:.4f}")

            ###############################################
            # early stopping
            # if training hasn't improved in the patience peroid, stop
            ###############################################
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_without_improve = 0

            else:
                epochs_without_improve += 1

            if epochs_without_improve >= patience:
                print("Early stopping")
                break

            epoch += 1

        ###############################################
        # Extract embeddings
        ###############################################

        model.eval()
        embeddings = []

        with torch.no_grad():
            for g in graphs:
                ###############################################
                # single batch format to ensure each graph and node embeddings are processed independently
                ###############################################
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
                g = g.to(device)

                ###############################################
                # gather node embeddings from encoder
                ###############################################
                z = model.encoder(g.x, g.edge_index, g.edge_attr)

                ###############################################
                # make graph level embedding by dogin mean pooling
                ###############################################
                embeddings.append(z.mean(dim=0).cpu().numpy())

        embeddings = np.array(embeddings)

        ###############################################
        # save the embeddings and metadata to a csv for later use
        ###############################################

        # convert the embeddings into a df
        emb_df = pd.DataFrame(embeddings)
        emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]

        # attach origional metadata identifiers
        final_df = pd.concat(
            [
                df.reset_index(drop=True)[["file", "algo", "level", "target", "n"]],
                emb_df,
            ],
            axis=1,
        )

        # create path for file
        final_output_path = Path(output_path).with_name(
            f"{model_name}_aug_dropout_and_norm_embeddings.csv"
        )

        # save to csv
        final_df.to_csv(final_output_path, index=False)

        ###############################################
        # save the loss data to a csv for later use
        ###############################################
        # make list df
        loss_df = pd.DataFrame(loss_history)

        # create path for file
        loss_output_path = Path(output_path).with_name(
            f"{model_name}_aug_dropout_and_norm_training_loss.csv"
        )
        # save to csv
        loss_df.to_csv(loss_output_path, index=False)

        ###############################################
        # callout to where the files are saved and their names
        ###############################################
        print("Saved training history to:", loss_output_path)
        print("Saved embeddings to:", final_output_path)


# if __name__ == "__main__":
#     CSV_PATH = "/Users/steventf/Desktop/school/milestoneII/SIADS696_circuit_swaps_and_polarization-main/data/datasets/train_swap_FakeBrisbane.csv"

#     QASM_DIR = Path(
#         "/Users/steventf/Desktop/school/milestoneII/SIADS696_circuit_swaps_and_polarization-main/src/mqtloader/qasm"
#     )

#     OUTPUT_DIR = Path(
#         "/Users/steventf/Desktop/school/milestoneII/SIADS696_circuit_swaps_and_polarization-main/data/datasets/test_with_comments"
#     )

#     run_graphcl_pipeline(csv_path=CSV_PATH, qasm_dir=QASM_DIR, output_path=OUTPUT_DIR)


import sys
from pathlib import Path

if __name__ == "__main__":
    # Check if the correct number of arguments were passed
    if len(sys.argv) < 4:
        print(
            "Usage: python embeddings_with_aug_and_dropout.py <csv_path> <qasm_dir> <output_dir>"
        )
        sys.exit(1)

    # Read paths from command-line arguments and resolve them
    CSV_PATH = Path(sys.argv[1]).resolve()
    QASM_DIR = Path(sys.argv[2]).resolve()
    OUTPUT_DIR = Path(sys.argv[3]).resolve()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    run_graphcl_pipeline(
        csv_path=str(CSV_PATH), qasm_dir=QASM_DIR, output_path=OUTPUT_DIR
    )
