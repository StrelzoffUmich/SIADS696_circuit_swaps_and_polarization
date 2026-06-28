"""
HDBSCAN clustering pipeline for embedding and swap data.

This script merges embedding vectors with swap metadata, preprocesses features
(using normalization and optional PCA), and applies HDBSCAN to discover
density-based clusters without requiring a predefined number of clusters.

It produces cluster assignments (including noise points and probabilities),
cluster-level summaries, structural profiling, and visualization outputs
for interpretability and downstream evaluation such as ARI.

note for me / users:
- row alignment is critical otherwise ARI will break
- hdbscan will naturally label outliers as -1 (noise)
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


###############################################
# PIPELINE FUNCTION
###############################################


def run_hdbscan_pipeline(swap_path, embedding_path, output_dir, PCA_VAL=False):
    """
    HDBSCAN clustering pipeline for embedding-based circuit analysis.

    This pipeline:
    - merges swap + embedding data
    - enforces strict row alignment (ARI-safe across methods)
    - normalizes embeddings
    - applies PCA
    - runs HDBSCAN clustering
    - computes cluster + noise statistics
    - produces structural summaries
    - saves outputs, plots, and metadata

    Returns
    -------
    main_df : pd.DataFrame
        Full dataset with HDBSCAN labels.

    output_df : pd.DataFrame
        Lightweight inference-style output.

    summary_df : pd.DataFrame
        Run-level metrics.
    """

    ###############################################
    # setup output directories
    ###############################################
    # create folder structure for reproducibility of outputs
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    data_dir = os.path.join(output_dir, "data")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    ###############################################
    # load and align data (needed for ARI evaluation)
    ###############################################
    # read in the csv files
    swap_df = pd.read_csv(swap_path)
    embedding_df = pd.read_csv(embedding_path)

    # merge on file identifier col to ensure row alignment
    base_df = swap_df.merge(embedding_df, on="file", how="inner")

    # identify the embedding feature columns
    emb_features = base_df.filter(regex=r"^emb_").columns.tolist()

    # drop rows with missing embedding values to avoid instability
    # should not be any, but just in case
    base_df = base_df.dropna(subset=emb_features).reset_index(drop=True)

    ###############################################
    # embedding matrix extraction
    ###############################################
    # convert embedding columns into np matrix
    X = base_df[emb_features].to_numpy(dtype=float)

    ###############################################
    # NORMALIZATION
    ###############################################
    # l2 normalization via sklearn to improve kmeans distance behavior
    X = normalize(X)

    ###############################################
    # PCA - optional and no longer needed
    ###############################################
    # this is a holdover from when we were testing with our
    # calculated results vs the qasm files themselves, False by default now
    if PCA_VAL == True:
        pca = PCA(n_components=0.90, random_state=42)
        X_used = pca.fit_transform(X)
    else:
        X_used = X

    ###############################################
    # hdbscan clustering (density-based method)
    ###############################################
    # hdbscan does not require predefined number of clusters (unlike kmeans/gmm)
    # but we do have to choose some numbers:

    # min_cluster_size=28:
    # tried to find a number that ensured the clusters weren't too small or noisy.
    # 1% of total datasize came up in a few posts, so I went with 28.
    # the hope is that using this for for gnn embeddings will help
    # avoid overfitting to any local density artifacts and to forces only
    # the meaningful, stable semantic groups to form

    # min_samples=10:
    # as i didn't want random small noise clusters, I chose 10, large enough
    # to prevent noise from taking over, but hopefully large enough to allow
    # meaningful groups to form. the thought is that higher values make
    # clustering more conservative and increase noise labeling,
    # which is useful for gnn embeddings that often have fuzzy boundaries

    # metric="euclidean":
    # this assumes that euclidean distance is meaningful in embedding space;
    # combined with l2 normalization, it should approximate cosine similarity
    # and works well for GNN embeddings, since the embeddings are normalized
    # to unit length.

    # cluster_selection_method="eom":
    # honestly, this is the default, but it's description is worth mentioning:
    # The standard approach for HDBSCAN* is to use an Excess of Mass ("eom")
    # algorithm to find the most persistent clusters. Alternatively you can instead
    # select the clusters at the leaves of the tree – this provides the most
    # fine grained and homogeneous clusters.
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=28,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    # fit model and assign cluster labels
    labels = clusterer.fit_predict(X_used)
    # probability-like confidence score per point (0 to 1)
    probs = clusterer.probabilities_

    base_df = base_df.copy()
    base_df["hdbscan_cluster"] = labels
    base_df["hdbscan_prob"] = probs

    ###############################################
    # metrics analysis - cluster vs noise separation
    # metrics defined by our teams research as important for the non-polarized dataset
    ###############################################
    # label -1 corresponds to noise points (not assigned to any cluster)
    noise = base_df[base_df["hdbscan_cluster"] == -1]
    clean = base_df[base_df["hdbscan_cluster"] != -1]

    metrics = ["bare_routed_2q", "bare_routed_depth", "mirror_over_bare_2q"]

    metric_rows = []

    for m in metrics:
        # compute teh descriptive statistics per cluster
        grouped = (
            base_df.groupby("hdbscan_cluster")[m]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )

        grouped["metric"] = m
        metric_rows.append(grouped)

    metric_df = pd.concat(metric_rows, ignore_index=True)

    # save cluster metric summaries for downstream analysis
    metric_df.to_csv(os.path.join(data_dir, "hdbscan_metric_summary.csv"), index=False)

    ###############################################
    # structural profiling of clusters
    ###############################################
    # these features describe graph / circuit topology behavior
    structural_cols = [
        "critical_depth",
        "program_communication",
        "parallelism",
        "fiedler_topology",
        "effective_resistance",
        "graph_density",
        "graph_diameter",
        "avg_clustering",
        "twoq_temporal_locality",
        "spectral_entropy_topology",
        "log_spanning_trees",
    ]

    # compute mean structural profile per cluster
    cluster_profile = (
        base_df.groupby("hdbscan_cluster")[structural_cols].mean().round(3)
    )

    # save cluster interpretability profiles
    cluster_profile.to_csv(os.path.join(data_dir, "hdbscan_cluster_profiles.csv"))

    ###############################################
    # 2d visualization
    ###############################################
    # just like k-means clustering script,
    # project the high-dim embeddings into 2d for visualization to see shape
    # for setup, follow ideas from: https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/
    X_2d = PCA(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=base_df["hdbscan_cluster"],
        palette="tab10",
        s=40,
        alpha=0.7,
    )

    plt.title("HDBSCAN Clusters (Embedding Space)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, "hdbscan_clusters_2d.png"))
    plt.close()

    ###############################################
    # create a summary df
    ###############################################
    # summary for logging and tracking for the code run
    summary_df = pd.DataFrame(
        [
            {
                "n_samples": len(base_df),
                "n_features": len(emb_features),
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "noise_points": len(noise),
                "noise_ratio": len(noise) / len(base_df),
            }
        ]
    )

    summary_df.to_csv(os.path.join(data_dir, "hdbscan_summary.csv"), index=False)

    ###############################################
    # create the full output for all out computed data and save to csv
    ###############################################
    # save the full enriched dataset for possible analysis and debugging

    main_df = base_df.copy()

    main_df.to_csv(os.path.join(data_dir, "hdbscan_clustered.csv"), index=False)

    ###############################################
    # pipeline output view (for ari)
    ###############################################
    # create a minimal view used for clustering comparison metric ari

    output_df = base_df[["file", "hdbscan_cluster", "hdbscan_prob"]].copy()

    output_df.to_csv(os.path.join(data_dir, "hdbscan_output_view.csv"), index=False)

    ###############################################
    # return the created dfs
    ###############################################

    return main_df, output_df, summary_df


###############################################
# MAIN ENTRY POINT
###############################################

if __name__ == "__main__":
    SWAP_PATH = "/Users/steventf/Desktop/school/milestoneII/SIADS696_circuit_swaps_and_polarization-main/data/datasets/train_swap_FakeBrisbane.csv"
    EMBEDDING_PATH = "/Users/steventf/Desktop/school/milestoneII/SIADS696_circuit_swaps_and_polarization-main/data/datasets/circuits_gat_augmented_and_normalized_embeddings_with_improved_edges.csv"
    OUTPUT_DIR = "/Users/steventf/Desktop/school/milestoneII/SIADS696_circuit_swaps_and_polarization-main/data/datasets/test"

    main_df, output_df, summary_df = run_hdbscan_pipeline(
        SWAP_PATH, EMBEDDING_PATH, OUTPUT_DIR
    )

    print("HDBSCAN pipeline complete.")
    print(summary_df)
