"""
GMM clustering pipeline for embedding and swap data.

This script merges embeddings with swap metadata, preprocesses features,
optionally applies PCA, and fits Gaussian Mixture Models with BIC-based
model selection.

It outputs cluster assignments (with soft confidence), diagnostics, and
summary files for downstream evaluation like ARI.

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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


###############################################
# PIPELINE FUNCTION
###############################################


def run_gmm_pipeline(swap_path, embedding_path, output_dir, PCA_VAL=False):
    """
    GMM clustering pipeline with BIC-based model selection.

    This pipeline:
    - merges swap + embedding data
    - ensures strict row alignment for cross-method comparison (ARI-safe)
    - normalizes embeddings
    - applies PCA --> no longer done!
    - selects best GMM via BIC
    - assigns soft + hard cluster labels
    - exports cluster outputs + summaries + plots

    Returns
    -------
    main_df : pd.DataFrame
        Full dataset with GMM assignments.

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
    # gmm model selection using bic
    ###############################################
    # try multiple cluster sizes and select best via lowest bic
    n_components_range = range(2, 20)

    bics = []
    best_gmm = None
    best_bic = np.inf
    best_k = None

    for k in n_components_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)

        gmm.fit(X_used)
        bic = gmm.bic(X_used)

        bics.append(bic)

        # keep best performing model
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

    ###############################################
    # final model inference
    ###############################################
    # assign the hard clusters and confidence scores
    labels = best_gmm.predict(X_used)
    probs = best_gmm.predict_proba(X_used).max(axis=1)

    base_df = base_df.copy()
    base_df["gmm_cluster"] = labels
    base_df["gmm_prob"] = probs

    ###############################################
    # silhouette score - might not use, but good to have
    ###############################################
    # not used for selection here
    # measures cluster separation quality
    sil = silhouette_score(X_used, labels)

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
        hue=base_df["gmm_cluster"],
        palette="tab10",
        s=40,
        alpha=0.7,
    )

    plt.title(f"GMM Clusters (K={best_k})")
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, "gmm_clusters_2d.png"))
    plt.close()

    ###############################################
    # metrics analysis
    # metrics defined by our teams research as important for the non-polarized dataset
    ###############################################
    # use important metrics to summarize per cluster
    metrics = ["bare_routed_2q", "bare_routed_depth", "mirror_over_bare_2q"]

    metric_rows = []

    for m in metrics:
        grouped = (
            base_df.groupby("gmm_cluster")[m]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )

        grouped["metric"] = m
        metric_rows.append(grouped)

    metric_df = pd.concat(metric_rows, ignore_index=True)

    # save the cluster metric summaries to csv
    metric_df.to_csv(os.path.join(data_dir, "gmm_metric_summary.csv"), index=False)

    ###############################################
    # create a summary df
    ###############################################
    # summary for logging and tracking for the code run

    summary_df = pd.DataFrame(
        [
            {
                "best_k": best_k,
                "bic": best_bic,
                "silhouette": sil,
                "n_samples": len(base_df),
                "n_features": len(emb_features),
                "pca_dims": X_used.shape[1],
            }
        ]
    )

    summary_df.to_csv(os.path.join(data_dir, "gmm_summary.csv"), index=False)

    ###############################################
    # create the full output for all out computed data and save to csv
    ###############################################
    # save the full enriched dataset for possible analysis and debugging

    main_df = base_df.copy()

    main_df.to_csv(os.path.join(data_dir, "gmm_clustered.csv"), index=False)

    ###############################################
    # pipeline output view (for ari)
    ###############################################
    # create a minimal view used for clustering comparison metric ari

    output_df = base_df[["file", "gmm_cluster", "gmm_prob"]].copy()

    output_df.to_csv(os.path.join(data_dir, "gmm_output_view.csv"), index=False)

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

    main_df, output_df, summary_df = run_gmm_pipeline(
        SWAP_PATH, EMBEDDING_PATH, OUTPUT_DIR
    )

    print("GMM pipeline complete.")
    print(summary_df)
