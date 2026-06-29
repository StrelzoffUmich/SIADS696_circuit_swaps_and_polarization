## Embedding Generation Scripts

This directory contains scripts for converting QASM files into DAG representations and generating graph embeddings using various Graph Neural Network (GNN) configurations.

### Files

- **`embeddings_with_aug_and_dropout.py`**  
  Converts QASM files into DAGs and generates embeddings using GCN, GAT, and GraphSAGE with augmentation and dropout.

- **`embeddings_with_aug_dropout_and_norm.py`**  
  Extends the above pipeline by adding feature normalization along with augmentation and dropout.

- **`embeddings_with_aug_dropout_and_norm_with_improved_edges.py`**  
  Adds improved DAG edge construction for more robust graph representations, in addition to augmentation, dropout, and normalization.

- **`embeddings_with_GraphCL.py`**  
  Baseline implementation using a GraphCL-style contrastive learning approach for generating embeddings from QASM-derived DAGs.

## scripts intended to be run through the generate_all_embeddings.sh script one level up
