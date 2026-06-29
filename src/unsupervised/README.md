# Unsupervised Learning Pipeline

This directory contains the scripts and notebooks used to generate graph neural network (GNN) embeddings and perform unsupervised clustering analyses on the resulting embedding vectors.

## Directory Structure

```text
unsupervised/
├── generate_all_embeddings.sh
├── run_model_scripts.sh
├── embedding_generation_scripts/
├── model_run_scripts/
└── unsupervised_learning_results_walkthrough.ipynb
```

### Files

| File / Directory                                  | Description                                                                                                                                      |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `generate_all_embeddings.sh`                      | Bash script that automates generation of all GNN embeddings by executing the scripts in `embedding_generation_scripts/`.                         |
| `run_model_scripts.sh`                            | Bash script that executes all unsupervised learning scripts contained in `model_run_scripts/`.                                                   |
| `embedding_generation_scripts/`                   | Scripts for generating graph embeddings from DAGs constructed from QASM circuits.                                                                |
| `model_run_scripts/`                              | Scripts that perform clustering using K-Means, Gaussian Mixture Models (GMM), and HDBSCAN on the generated embeddings.                           |
| `unsupervised_learning_results_walkthrough.ipynb` | Notebook demonstrating the post-processing workflow, clustering analysis, evaluation, and preparation of results used in the accompanying paper. |

---

## Running the Pipeline

### 1. Generate GNN Embeddings

Generate embeddings for all configured GNN models and circuit complexity levels.

```bash
bash src/unsupervised/generate_all_embeddings.sh
```

> **Note:** Embedding generation can take a significant amount of time depending on the number of circuits and GNN architectures.

---

### 2. Run Unsupervised Learning Models

After embeddings have been generated, execute the clustering analyses.

```bash
bash src/unsupervised/run_model_scripts.sh
```

This runs the following clustering methods:

* K-Means
* Gaussian Mixture Models (GMM)
* HDBSCAN

The clustering results are compared using the **Adjusted Rand Index (ARI)**.

---

## Output Files

### Generated Embeddings

Embedding CSV files are saved to:

```text
data/datasets/embedding_data/embeddings/
    {GNN_type}_{complexity_levels}.csv
```

Training loss values for each GNN model are saved to:

```text
data/datasets/embedding_data/loss/
    {GNN_type}_{complexity_levels}_loss.csv
```

---

## Unsupervised Learning Results

All clustering outputs are written to:

```text
data/datasets/unsupervised_learning_results/
```

Many intermediate files are generated during execution. The primary summary files are:

| File                              | Description                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------- |
| `ALL_cluster_assignments.csv`     | Cluster assignments for every embedding across all clustering methods.            |
| `ALL_summary_comparison.csv`      | Summary statistics and clustering performance metrics, including ARI comparisons. |
| `embedding_comparison_ranked.csv` | Ranked comparison of embedding methods based on clustering performance.           |

---

## Analysis Notebook

The notebook

```text
unsupervised_learning_results_walkthrough.ipynb
```

provides a complete walkthrough of:

* Loading the generated embedding datasets
* Running post-processing analyses
* Comparing clustering performance across embedding methods
* Computing and interpreting Adjusted Rand Index (ARI) scores
* Generating figures and summary tables used for publication
