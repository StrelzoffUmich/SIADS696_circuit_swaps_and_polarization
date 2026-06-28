"""
KMeans clustering pipeline for embedding data.

This script merges swap and embedding datasets, preprocesses and normalizes
feature vectors, optionally applies PCA, and runs KMeans clustering with
automatic K selection using silhouette scores.

It outputs cluster assignments, distance-to-centroid scores, summary stats,
and basic visualizations for analysis and downstream evaluation.

note for me / users:
- row alignment is critical otherwise ARI will break
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


###############################################
# PIPELINE FUNCTION
###############################################


def run_kmeans_pipeline(swap_path, embedding_path, output_dir, PCA_VAL=False):
    """
    kmeans clustering pipeline with silhouette-based k selection.

    this function builds a full clustering workflow over embedding data:
    - loads swap and embedding datasets and merges them on file id
    - enforces strict alignment for reproducibility (important for ari comparisons)
    - extracts embedding features and cleans missing rows
    - normalizes feature vectors (l2 normalization)
    - optionally reduces dimensionality using pca
    - selects optimal k using silhouette score over a fixed range
    - fits final kmeans model using best k
    - assigns clusters and computes distance-to-centroid scores
    - generates diagnostic plots and summary metrics
    - exports all artifacts for downstream analysis

    parameters
    ----------
    swap_path : str
        path to swap dataset (csv)

    embedding_path : str
        path to embedding dataset (csv)

    output_dir : str
        directory where outputs (plots, csvs) will be saved

    PCA_VAL : bool
        if true, applies pca (90% variance retained), otherwise skips it

    returns
    -------
    main_df : pd.DataFrame
        full dataset including cluster assignments and diagnostics

    output_df : pd.DataFrame
        minimal inference-ready view (file, cluster, distance)

    summary_df : pd.DataFrame
        run-level summary (best k, silhouette score, feature stats)
    """

    ###############################################
    # setup output dirs
    ###############################################
    # create the main output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    data_dir = os.path.join(output_dir, "data")

    # separate the outputs into plots and tabular data
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
    # create the embeddings matrix
    ###############################################
    # convert embedding columns into np matrix
    X = base_df[emb_features].to_numpy(dtype=float)

    ###############################################
    # normalization
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
    # silhouette search for best k
    ###############################################

    k_range = range(2, 21)

    scores = []
    best_k = None
    best_score = -1
    best_model = None

    # evaluate kmeans for multiple k values, using silhouette score as the judge
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)

        labels = km.fit_predict(X_used)

        # silhouette score measures cluster separation quality
        score = silhouette_score(X_used, labels)
        scores.append(score)

        # keep track best performing model
        if score > best_score:
            best_score = score
            best_k = k
            best_model = km

    ###############################################
    # final model output
    ###############################################
    # assign the final cluster labels
    labels = best_model.predict(X_used)
    # base_df["kmeans_cluster"] = labels

    # base_df = base_df.copy()
    # # calculate the distance to nearest centroid to help determine confidence
    distances = best_model.transform(X_used)
    # base_df["kmeans_distance"] = distances.min(axis=1)

    base_df = pd.concat(
        [
            base_df,
            pd.DataFrame(
                {"kmeans_cluster": labels, "kmeans_distance": distances.min(axis=1)},
                index=base_df.index,
            ),
        ],
        axis=1,
    )

    ###############################################
    # 2d visualization
    ###############################################
    # project the high-dim embeddings into 2d for visualization to see shape
    # for setup, i follow ideas from: https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/
    X_2d = PCA(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=base_df["kmeans_cluster"],
        palette="tab10",
        s=40,
        alpha=0.7,
    )

    plt.title(f"KMeans Clusters (K={best_k})")
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, "kmeans_clusters_2d.png"))
    plt.close()

    ###############################################
    # silhouette curve
    ###############################################
    # visualize how the silhouette score changes with k
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), scores, marker="o")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.title("KMeans Silhouette Search")

    plt.grid(True)

    plt.savefig(os.path.join(plot_dir, "kmeans_silhouette_curve.png"))
    plt.close()

    ###############################################
    # metrics analysis
    # metrics defined by our teams research as important for the non-polarized dataset
    ###############################################
    # use important metrics to summarize per cluster
    metrics = ["bare_routed_2q", "bare_routed_depth", "mirror_over_bare_2q"]

    metric_rows = []

    # compute the cluster-level statistics for each metric
    for m in metrics:
        grouped = (
            base_df.groupby("kmeans_cluster")[m]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )

        grouped["metric"] = m
        metric_rows.append(grouped)

    metric_df = pd.concat(metric_rows, ignore_index=True)

    # save the cluster metric summaries to csv
    metric_df.to_csv(os.path.join(data_dir, "kmeans_metric_summary.csv"), index=False)

    ###############################################
    # create a summary df
    ###############################################
    # summary for logging and tracking for the code run
    summary_df = pd.DataFrame(
        [
            {
                "best_k": best_k,
                "silhouette": best_score,
                "n_samples": len(base_df),
                "n_features": len(emb_features),
                "pca_dims": X_used.shape[1],
            }
        ]
    )

    summary_df.to_csv(os.path.join(data_dir, "kmeans_summary.csv"), index=False)

    ###############################################
    # create the full output for all out computed data and save to csv
    ###############################################
    # save the full enriched dataset for possible analysis and debugging
    main_df = base_df.copy()

    main_df.to_csv(os.path.join(data_dir, "kmeans_clustered.csv"), index=False)

    ###############################################
    # pipeline output view (for ari)
    ###############################################
    # create a minimal view used for clustering comparison metric ari
    output_df = base_df[["file", "kmeans_cluster", "kmeans_distance"]].copy()
    output_df.to_csv(os.path.join(data_dir, "kmeans_output_view.csv"), index=False)

    ###############################################
    # return the created dfs
    ###############################################
    return main_df, output_df, summary_df


###############################################
# MAIN ENTRY POINT
###############################################

if __name__ == "__main__":
    SWAP_PATH = "/Users/steventf/Desktop/school/milestoneII/SIADS696_circuit_swaps_and_polarization-main/data/datasets/train_swap_FakeBrisbane.csv"
    EMBEDDING_PATH = "/Users/steventf/Desktop/school/milestoneII/SIADS696_circuit_swaps_and_polarization-main/data/datasets/embedding_data/embeddings/graphsage_aug_dropout_and_norm_with_improved_edges_training.csv"
    OUTPUT_DIR = "/Users/steventf/Desktop/school/milestoneII/SIADS696_circuit_swaps_and_polarization-main/data/datasets/test"

    main_df, output_df, summary_df = run_kmeans_pipeline(
        SWAP_PATH, EMBEDDING_PATH, OUTPUT_DIR, PCA_VAL=False
    )

    print("KMeans pipeline complete.")
    print(summary_df)
