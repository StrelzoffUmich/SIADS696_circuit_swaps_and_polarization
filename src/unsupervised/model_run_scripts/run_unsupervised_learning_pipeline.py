"""
Across all embeddings:
    - Builds a ranked comparison table of embedding quality
    - Saves all clustering outputs, summaries, and comparisons
"""

import os
from pathlib import Path
import pandas as pd

pd.set_option("display.max_colwidth", None)

from kmeans_clustering_pipeline import run_kmeans_pipeline
from gmm_pipeline import run_gmm_pipeline
from hdbscan_pipeline import run_hdbscan_pipeline

from sklearn.metrics import adjusted_rand_score


def run_unsupervised_learning_analysis(embedding_dir, csv_path, output_dir):
    """
    Runs kmeans, gmm, and hdbscan clustering pipelines across all embedding files
    in the given directory, computes pairwise ARI scores, and returns ranked results.

    Parameters
    ----------
    embedding_dir : str or Path
        Directory containing embedding CSV files.
    csv_path : str
        Path to the swap dataset CSV.
    output_dir : str
        Directory where per-embedding summaries will be saved.

    Returns
    -------
    comparison_df : pd.DataFrame
        Embeddings ranked by mean ARI agreement score across methods.
    final_results : pd.DataFrame
        Full cluster assignments for all embeddings.
    final_summaries : pd.DataFrame
        Per-method summaries for all embeddings.
    """

    ###############################################
    # CONFIGURATION
    ###############################################
    PCA_VAL = False

    # easier than listing out each file when the names change
    embedding_files = [f.name for f in EMBEDDING_DIR.iterdir() if f.is_file()]

    # Output subdirectories for each clustering method
    KMEANS_DIR = os.path.join(OUTPUT_DIR, "kmeans")
    GMM_DIR = os.path.join(OUTPUT_DIR, "gmm")
    HDBSCAN_DIR = os.path.join(OUTPUT_DIR, "hdbscan")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ###############################################
    # STORAGE FOR GLOBAL RESULTS
    ###############################################

    # Stores per-embedding comparison metrics
    all_comparisons = []

    # Stores full clustering outputs for all embeddings
    all_results = []

    # Stores full summaries for all embeddings
    all_summaries = []

    ###############################################
    # MAIN EXPERIMENT LOOP
    ###############################################

    for emb_file in embedding_files:
        # Construct full path to embedding file
        EMBEDDING_PATH = os.path.join(EMBEDDING_DIR, emb_file)

        print("\n======================================")
        print(f"Running pipelines for: {emb_file}")
        print("======================================\n")

        ###############################################
        # RUN CLUSTERING PIPELINES
        ###############################################

        # KMeans clustering
        kmeans_main, kmeans_out, kmeans_summary = run_kmeans_pipeline(
            CSV_PATH, EMBEDDING_PATH, KMEANS_DIR
        )

        # Gaussian Mixture Model clustering
        gmm_main, gmm_out, gmm_summary = run_gmm_pipeline(
            CSV_PATH, EMBEDDING_PATH, GMM_DIR
        )

        # HDBSCAN clustering
        hdbscan_main, hdbscan_out, hdbscan_summary = run_hdbscan_pipeline(
            CSV_PATH, EMBEDDING_PATH, HDBSCAN_DIR
        )

        ###############################################
        # MERGE CLUSTER ASSIGNMENTS
        ###############################################

        # Create unified dataframe for ARI comparison
        results_df = kmeans_main[["file"]].copy()

        results_df["kmeans_cluster"] = kmeans_main["kmeans_cluster"]
        results_df["gmm_cluster"] = gmm_main["gmm_cluster"]
        results_df["hdbscan_cluster"] = hdbscan_main["hdbscan_cluster"]

        # Track which embedding generated these results
        results_df["embedding_file"] = emb_file

        all_results.append(results_df)

        ###############################################
        # COMPUTE ARI MATRIX
        ###############################################

        methods = ["kmeans", "gmm", "hdbscan"]

        ari_matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)

        # Pairwise ARI between clustering methods
        for a in methods:
            for b in methods:
                ari_matrix.loc[a, b] = adjusted_rand_score(
                    results_df[f"{a}_cluster"], results_df[f"{b}_cluster"]
                )

        print("\nARI Matrix:")
        print(ari_matrix)

        ###############################################
        # EXTRACT SUMMARY METRICS
        ###############################################

        # Pairwise agreement scores
        k_g = ari_matrix.loc["kmeans", "gmm"]
        k_h = ari_matrix.loc["kmeans", "hdbscan"]
        g_h = ari_matrix.loc["gmm", "hdbscan"]

        # Mean agreement score (embedding stability score)
        mean_score = (k_g + k_h + g_h) / 3

        # Store comparison results for global ranking
        all_comparisons.append(
            {
                "embedding_file": emb_file,
                "mean_score": mean_score,
                "kmeans_vs_gmm": k_g,
                "kmeans_vs_hdbscan": k_h,
                "gmm_vs_hdbscan": g_h,
            }
        )

        ###############################################
        # SAVE PER-EMBEDDING OUTPUTS
        ###############################################

        # Save ARI matrix
        ari_matrix.to_csv(os.path.join(OUTPUT_DIR, f"ari_matrix_{emb_file}.csv"))

        # Save clustering assignments
        results_df.to_csv(
            os.path.join(OUTPUT_DIR, f"clusters_{emb_file}.csv"), index=False
        )

        # Combine per-method summaries into one table
        summary_df = pd.concat(
            [
                kmeans_summary.assign(method="kmeans", embedding=emb_file),
                gmm_summary.assign(method="gmm", embedding=emb_file),
                hdbscan_summary.assign(method="hdbscan", embedding=emb_file),
            ]
        )

        summary_df.to_csv(
            os.path.join(OUTPUT_DIR, f"summary_{emb_file}.csv"), index=False
        )

        all_summaries.append(summary_df)

    ###############################################
    # GLOBAL EMBEDDING COMPARISON TABLE
    ###############################################

    comparison_df = pd.DataFrame(all_comparisons)

    # Rank embeddings by highest mean stability score
    comparison_df = comparison_df.sort_values("mean_score", ascending=False)

    # Reorder columns for readability
    comparison_df = comparison_df[
        [
            "embedding_file",
            "mean_score",
            "kmeans_vs_gmm",
            "kmeans_vs_hdbscan",
            "gmm_vs_hdbscan",
        ]
    ]

    print("\n======================================")
    print("=== EMBEDDING RANKING (BEST → WORST) ===")
    print("======================================\n")
    print(comparison_df)

    # Save ranked comparison
    comparison_df.to_csv(
        os.path.join(OUTPUT_DIR, "embedding_comparison_ranked.csv"), index=False
    )

    ###############################################
    # FINAL AGGREGATED EXPORTS
    ###############################################

    # Full clustering results across all embeddings
    final_results = pd.concat(all_results)

    # Full method summaries across all embeddings
    final_summaries = pd.concat(all_summaries)

    final_results.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_cluster_assignments.csv"), index=False
    )

    final_summaries.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_summary_comparison.csv"), index=False
    )

    ###############################################
    # COMPLETION MESSAGE
    ###############################################

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Processed {len(embedding_files)} embedding files")
    print("\nTop-performing embedding:")
    print(comparison_df.iloc[0])


# ###############################################
# # MAIN ENTRY POINT
# ###############################################

# if __name__ == "__main__":
#     EMBEDDING_DIR = Path(
#         "/Users/steventf/Desktop/school/milestoneII/SIADS696_circuit_swaps_and_polarization-main/data/datasets/embedding_data/embeddings"
#     )
#     CSV_PATH = "/Users/steventf/Desktop/school/milestoneII/SIADS696_circuit_swaps_and_polarization-main/data/datasets/train_swap_FakeBrisbane.csv"
#     OUTPUT_DIR = "/Users/steventf/Desktop/school/milestoneII/SIADS696_circuit_swaps_and_polarization-main/data/datasets/TEST-unsupervised_results_pipeline_script_test"

#     comparison_df, final_results, final_summaries = run_unsupervised_learning_analysis(
#         EMBEDDING_DIR, CSV_PATH, OUTPUT_DIR
#     )

#     print("run_unsupervised_learning_analysis pipeline complete.")


import sys
from pathlib import Path

if __name__ == "__main__":
    # Check if the correct number of arguments were passed
    if len(sys.argv) < 4:
        print(
            "Usage: python embeddings_with_autoencoder.py <embedding_dir> <csv_path> <output_dir>"
        )
        sys.exit(1)

    # Read paths from command-line arguments and resolve them
    EMBEDDING_DIR = Path(sys.argv[1]).resolve()
    CSV_PATH = Path(sys.argv[2]).resolve()
    OUTPUT_DIR = Path(sys.argv[3]).resolve()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    run_unsupervised_learning_analysis(
        embedding_dir=str(EMBEDDING_DIR), csv_path=CSV_PATH, output_dir=OUTPUT_DIR
    )
