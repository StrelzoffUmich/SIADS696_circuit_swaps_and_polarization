# Unsupervised Clustering Pipelines (Embeddings + Swap Data)

This project contains four scripts for clustering embedding and swap data and comparing results across methods.

The key requirement across all scripts is correct row alignment between embeddings and swap metadata. If alignment is broken, ARI evaluation will fail.

## gmm_pipeline.py

Gaussian Mixture Model clustering for embedding and swap data.

Merges embeddings with swap metadata, preprocesses features, optionally applies PCA, and fits GMMs with BIC-based model selection.

Outputs cluster assignments (with soft confidence), diagnostics, and summary files for downstream evaluation like ARI.

Note:
- Row alignment is critical otherwise ARI will break


## hdbscan_pipeline.py

HDBSCAN clustering for embedding and swap data.

Merges embeddings with swap metadata, preprocesses features (normalization and optional PCA), and applies HDBSCAN for density-based clustering without specifying number of clusters.

Outputs cluster assignments including noise points, probabilities, cluster summaries, structural profiling, and visualization outputs.

Notes:
- Row alignment is critical otherwise ARI will break
- HDBSCAN labels outliers as -1 (noise)


## kmeans_clustering_pipeline.py

KMeans clustering for embedding data.

Merges swap and embedding datasets, preprocesses and normalizes features, optionally applies PCA, and runs KMeans with automatic K selection using silhouette scores.

Outputs cluster assignments, distance-to-centroid scores, summary stats, and basic visualizations.

Note:
- Row alignment is critical otherwise ARI will break


## run_unsupervised_learning_pipeline.py

Runs across all embeddings.

Builds a ranked comparison table of embedding quality and saves all clustering outputs, summaries, and comparisons.
