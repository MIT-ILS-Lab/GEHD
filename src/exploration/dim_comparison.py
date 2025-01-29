import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    TSNE,
    SpectralEmbedding,
    MDS,
)
from sklearn.decomposition import KernelPCA
import umap.umap_ as umap
from utils import load_config, get_nearest_neighbors, load_manifold_data
from sklearn.metrics import jaccard_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import pickle


def compute_jaccard_similarity(original_neighbors, reduced_neighbors, distance_matrix):
    jaccard_similarities = []
    for orig_neighbors, red_neighbors in zip(original_neighbors, reduced_neighbors):
        orig_binary = np.zeros(len(distance_matrix))
        red_binary = np.zeros(len(distance_matrix))
        orig_binary[orig_neighbors] = 1
        red_binary[red_neighbors] = 1
        jaccard_similarities.append(jaccard_score(orig_binary, red_binary))
    return np.mean(jaccard_similarities)


def compute_relative_distances(dist_matrix, original_neighbors, k=300):
    # Calculate the pairwise Euclidean distances for the original and reduced embeddings
    relative_diffs = []
    for i, neighbors in enumerate(original_neighbors):
        original_dists = dist_matrix[i, neighbors]
        reduced_dists = np.sort(dist_matrix[i])[
            :k
        ]  # Nearest k neighbors in the reduced embedding
        diff = np.abs(original_dists - reduced_dists)
        relative_diffs.append(
            np.mean(diff)
        )  # Compute the mean difference for the neighbors
    return np.mean(relative_diffs)


def normalize_distances(dist_matrix):
    # Normalize the distance matrix to the range [0, 1]
    scaler = MinMaxScaler()
    return scaler.fit_transform(dist_matrix)


def save_results(results, filename):
    with open(filename, "wb") as f:
        pickle.dump(results, f)


# Load config
config = load_config("config.yaml")

if __name__ == "__main__":

    # Load data
    distance_matrix, graph, locations = load_manifold_data(config)
    # Initialize parameters
    k = 200  # Number of neighbors
    original_neighbors = get_nearest_neighbors(distance_matrix, k)

    # Dimensions to check
    dimensions = [40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2]

    # Initialize dictionary to store results
    results = {
        "jaccard_scores_lle_standard": [],
        "jaccard_scores_lle_hessian": [],
        "jaccard_scores_lle_modified": [],
        "rec_error_lle_standard": [],
        "rec_error_lle_hessian": [],
        "rec_error_lle_modified": [],
        "jaccard_scores_tsne": [],
        "kl_divergence_tsne": [],
        "jaccard_scores_ltsa": [],
        "rec_error_ltsa": [],
        "jaccard_scores_isomap": [],
        "rec_error_isomap": [],
        "jaccard_scores_mds": [],
        "stress_mds": [],
        "distance_errors_lle_standard": [],
        "distance_errors_lle_hessian": [],
        "distance_errors_lle_modified": [],
        "distance_errors_tsne": [],
        "distance_errors_ltsa": [],
        "distance_errors_isomap": [],
        "distance_errors_mds": [],
    }

    # Isomap
    for dim in tqdm(dimensions, desc="Running Isomap"):
        print(f"\nRunning Isomap for dimension {dim}")
        isomap = Isomap(n_neighbors=k, n_components=dim, n_jobs=-1)
        reduced_embedding_isomap = isomap.fit_transform(distance_matrix)

        reduced_neighbors_isomap = get_nearest_neighbors(reduced_embedding_isomap, k)
        dist_matrix_emb_isomap = pairwise_distances(reduced_embedding_isomap)
        dist_matrix_emb_isomap = normalize_distances(dist_matrix_emb_isomap)

        rec_error_isomap = isomap.reconstruction_error()
        print(f"Reconstruction error: {rec_error_isomap}")

        jaccard_similarity_isomap = compute_jaccard_similarity(
            original_neighbors, reduced_neighbors_isomap, distance_matrix
        )
        print(f"Jaccard similarity: {jaccard_similarity_isomap}")

        relative_distance_error_isomap = compute_relative_distances(
            dist_matrix_emb_isomap, original_neighbors, k
        )
        print(f"Relative distance error: {relative_distance_error_isomap}")

        results["rec_error_isomap"].append(rec_error_isomap)
        results["jaccard_scores_isomap"].append(jaccard_similarity_isomap)
        results["distance_errors_isomap"].append(relative_distance_error_isomap)

        # Save the results after the Isomap loop
        save_results(results, "all_results.pkl")

    # Loop through dimensions and apply manifold learning models
    for dim in tqdm(dimensions, desc="Running manifold learning models"):
        # LLE Standard
        print(f"\nRunning LLE Standard for dimension {dim}")
        lle_standard = LocallyLinearEmbedding(
            n_neighbors=k, n_components=dim, method="standard", n_jobs=-1
        )
        reduced_embedding_lle_standard = lle_standard.fit_transform(distance_matrix)

        reduced_neighbors_lle_standard = get_nearest_neighbors(
            reduced_embedding_lle_standard, k
        )
        dist_matrix_emb_lle_standard = pairwise_distances(
            reduced_embedding_lle_standard
        )
        dist_matrix_emb_lle_standard = normalize_distances(dist_matrix_emb_lle_standard)

        rec_error_lle_standard = lle_standard.reconstruction_error_
        print(f"Reconstruction error: {rec_error_lle_standard}")

        jaccard_similarity_lle_standard = compute_jaccard_similarity(
            original_neighbors, reduced_neighbors_lle_standard, distance_matrix
        )
        print(f"Jaccard similarity: {jaccard_similarity_lle_standard}")

        relative_distance_error_lle_standard = compute_relative_distances(
            dist_matrix_emb_lle_standard, original_neighbors, k
        )
        print(f"Relative distance error: {relative_distance_error_lle_standard}")

        results["rec_error_lle_standard"].append(rec_error_lle_standard)
        results["jaccard_scores_lle_standard"].append(jaccard_similarity_lle_standard)
        results["distance_errors_lle_standard"].append(
            relative_distance_error_lle_standard
        )

        # Save the results after LLE Standard loop
        save_results(results, "all_results.pkl")

        # LLE Hessian
        print(f"\nRunning LLE Hessian for dimension {dim}")
        lle_hessian = LocallyLinearEmbedding(
            n_neighbors=k, n_components=dim, method="hessian", n_jobs=-1
        )
        reduced_embedding_lle_hessian = lle_hessian.fit_transform(distance_matrix)

        reduced_neighbors_lle_hessian = get_nearest_neighbors(
            reduced_embedding_lle_hessian, k
        )
        dist_matrix_emb_lle_hessian = pairwise_distances(reduced_embedding_lle_hessian)
        dist_matrix_emb_lle_hessian = normalize_distances(dist_matrix_emb_lle_hessian)

        rec_error_lle_hessian = lle_hessian.reconstruction_error_
        print(f"Reconstruction error: {rec_error_lle_hessian}")

        jaccard_similarity_lle_hessian = compute_jaccard_similarity(
            original_neighbors, reduced_neighbors_lle_hessian, distance_matrix
        )
        print(f"Jaccard similarity: {jaccard_similarity_lle_hessian}")

        relative_distance_error_lle_hessian = compute_relative_distances(
            dist_matrix_emb_lle_hessian, original_neighbors, k
        )
        print(f"Relative distance error: {relative_distance_error_lle_hessian}")

        results["rec_error_lle_hessian"].append(rec_error_lle_hessian)
        results["jaccard_scores_lle_hessian"].append(jaccard_similarity_lle_hessian)
        results["distance_errors_lle_hessian"].append(
            relative_distance_error_lle_hessian
        )

        # Save the results after LLE Hessian loop
        save_results(results, "all_results.pkl")

        # LLE Modified
        print(f"\nRunning LLE Modified for dimension {dim}")
        lle_modified = LocallyLinearEmbedding(
            n_neighbors=k, n_components=dim, method="modified", n_jobs=-1
        )
        reduced_embedding_lle_modified = lle_modified.fit_transform(distance_matrix)

        reduced_neighbors_lle_modified = get_nearest_neighbors(
            reduced_embedding_lle_modified, k
        )
        dist_matrix_emb_lle_modified = pairwise_distances(
            reduced_embedding_lle_modified
        )
        dist_matrix_emb_lle_modified = normalize_distances(dist_matrix_emb_lle_modified)

        rec_error_lle_modified = lle_modified.reconstruction_error_
        print(f"Reconstruction error: {rec_error_lle_modified}")

        jaccard_similarity_lle_modified = compute_jaccard_similarity(
            original_neighbors, reduced_neighbors_lle_modified, distance_matrix
        )
        print(f"Jaccard similarity: {jaccard_similarity_lle_modified}")

        relative_distance_error_lle_modified = compute_relative_distances(
            dist_matrix_emb_lle_modified, original_neighbors, k
        )
        print(f"Relative distance error: {relative_distance_error_lle_modified}")

        results["rec_error_lle_modified"].append(rec_error_lle_modified)
        results["jaccard_scores_lle_modified"].append(jaccard_similarity_lle_modified)
        results["distance_errors_lle_modified"].append(
            relative_distance_error_lle_modified
        )

        # Save the results after LLE Modified loop
        save_results(results, "all_results.pkl")

        # t-SNE
        print(f"\nRunning t-SNE for dimension {dim}")
        tsne = TSNE(
            n_components=dim,
            metric="precomputed",
            random_state=42,
            init="random",
            perplexity=k,
            n_jobs=-1,
        )
        reduced_embedding_tsne = tsne.fit_transform(distance_matrix)

        reduced_neighbors_tsne = get_nearest_neighbors(reduced_embedding_tsne, k)
        dist_matrix_emb_tsne = pairwise_distances(reduced_embedding_tsne)
        dist_matrix_emb_tsne = normalize_distances(dist_matrix_emb_tsne)

        kl_divergence_tsne = tsne.kl_divergence_
        print(f"KL divergence: {kl_divergence_tsne}")

        jaccard_similarity_tsne = compute_jaccard_similarity(
            original_neighbors, reduced_neighbors_tsne, distance_matrix
        )
        print(f"Jaccard similarity: {jaccard_similarity_tsne}")

        relative_distance_error_tsne = compute_relative_distances(
            dist_matrix_emb_tsne, original_neighbors, k
        )
        print(f"Relative distance error: {relative_distance_error_tsne}")

        results["kl_divergence_tsne"].append(kl_divergence_tsne)
        results["jaccard_scores_tsne"].append(jaccard_similarity_tsne)
        results["distance_errors_tsne"].append(relative_distance_error_tsne)

        # Save the results after t-SNE loop
        save_results(results, "all_results.pkl")

    # LTSA (Local Tangent Space Alignment)
    for dim in tqdm(dimensions, desc="Running LTSA"):
        print(f"\nRunning LTSA for dimension {dim}")
        ltsa = LocallyLinearEmbedding(
            n_neighbors=k, n_components=dim, method="ltsa", n_jobs=-1
        )
        reduced_embedding_ltsa = ltsa.fit_transform(distance_matrix)

        reduced_neighbors_ltsa = get_nearest_neighbors(reduced_embedding_ltsa, k)
        dist_matrix_emb_ltsa = pairwise_distances(reduced_embedding_ltsa)
        dist_matrix_emb_ltsa = normalize_distances(dist_matrix_emb_ltsa)

        rec_error_ltsa = ltsa.reconstruction_error_
        print(f"Reconstruction error: {rec_error_ltsa}")

        jaccard_similarity_ltsa = compute_jaccard_similarity(
            original_neighbors, reduced_neighbors_ltsa, distance_matrix
        )
        print(f"Jaccard similarity: {jaccard_similarity_ltsa}")

        relative_distance_error_ltsa = compute_relative_distances(
            dist_matrix_emb_ltsa, original_neighbors, k
        )
        print(f"Relative distance error: {relative_distance_error_ltsa}")

        results["rec_error_ltsa"].append(rec_error_ltsa)
        results["jaccard_scores_ltsa"].append(jaccard_similarity_ltsa)
        results["distance_errors_ltsa"].append(relative_distance_error_ltsa)

        # Save the results after LTSA loop
        save_results(results, "all_results.pkl")

    # MDS
    for dim in tqdm(dimensions, desc="Running MDS"):
        print(f"\nRunning MDS for dimension {dim}")
        mds = MDS(
            n_components=dim, dissimilarity="precomputed", random_state=42, n_jobs=-1
        )
        reduced_embedding_mds = mds.fit_transform(distance_matrix)

        reduced_neighbors_mds = get_nearest_neighbors(reduced_embedding_mds, k)
        dist_matrix_emb_mds = pairwise_distances(reduced_embedding_mds)
        dist_matrix_emb_mds = normalize_distances(dist_matrix_emb_mds)

        stress_mds = mds.stress_
        print(f"Stress: {stress_mds}")

        jaccard_similarity_mds = compute_jaccard_similarity(
            original_neighbors, reduced_neighbors_mds, distance_matrix
        )
        print(f"Jaccard similarity: {jaccard_similarity_mds}")

        relative_distance_error_mds = compute_relative_distances(
            dist_matrix_emb_mds, original_neighbors, k
        )
        print(f"Relative distance error: {relative_distance_error_mds}")

        results["stress_mds"].append(stress_mds)
        results["jaccard_scores_mds"].append(jaccard_similarity_mds)
        results["distance_errors_mds"].append(relative_distance_error_mds)

        # Save the results after MDS loop
        save_results(results, "all_results.pkl")
