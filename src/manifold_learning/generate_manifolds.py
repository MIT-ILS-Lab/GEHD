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
from manifold_learning.visualize_manifolds import (
    plot_interactive,
    plot_subplots,
    plot_2d_locations,
)
from sklearn.metrics import jaccard_score


class ManifoldLearning:
    def __init__(self, distance_matrix, k=300, n_components=3):
        self.distance_matrix = distance_matrix
        self.k = k
        self.n_components = n_components

    def apply_isomap(self):
        model = Isomap(n_neighbors=self.k, n_components=self.n_components)
        return model.fit_transform(self.distance_matrix)

    def apply_lle(self, method: str = "standard"):
        model = LocallyLinearEmbedding(
            n_neighbors=self.k, n_components=self.n_components, method=method
        )
        return model.fit_transform(self.distance_matrix)

    def apply_tsne(self):
        model = TSNE(
            n_components=self.n_components,
            metric="precomputed",
            random_state=42,
            init="random",
            perplexity=self.k,
        )
        return model.fit_transform(self.distance_matrix)

    def apply_umap(self):
        model = umap.UMAP(
            n_neighbors=self.k,
            n_components=self.n_components,
            metric="precomputed",
            random_state=42,
        )
        return model.fit_transform(self.distance_matrix)

    def apply_laplacian(self):
        model = SpectralEmbedding(
            n_components=self.n_components, affinity="precomputed"
        )
        return model.fit_transform(self.distance_matrix)

    def apply_kernel_pca(self):
        model = KernelPCA(n_components=self.n_components, kernel="precomputed")
        return model.fit_transform(self.distance_matrix)

    def apply_mds(self, metric=True):
        model = MDS(
            n_components=self.n_components,
            dissimilarity="precomputed",
            random_state=42,
            metric=metric,
        )
        return model.fit_transform(self.distance_matrix)


def compute_jaccard_similarity(original_neighbors, reduced_neighbors, distance_matrix):
    jaccard_similarities = []
    for orig_neighbors, red_neighbors in zip(original_neighbors, reduced_neighbors):
        orig_binary = np.zeros(len(distance_matrix))
        red_binary = np.zeros(len(distance_matrix))
        orig_binary[orig_neighbors] = 1
        red_binary[red_neighbors] = 1
        jaccard_similarities.append(jaccard_score(orig_binary, red_binary))
    return np.mean(jaccard_similarities)


if __name__ == "__main__":
    # Load config
    config = load_config("config.yaml")

    # Load data
    distance_matrix, graph, locations = load_manifold_data(config)

    # Parameters
    k = config["manifold"]["n_neighbors"]  # Reduced k value
    n_components = config["manifold"]["n_components"]  # Number of components

    manifold_learning = ManifoldLearning(distance_matrix, k, n_components)

    # Choose methods to test
    models = [
        # (manifold_learning.apply_isomap, "Isomap"),
        (manifold_learning.apply_lle, "LLE"),
        (manifold_learning.apply_lle(method="hessian"), "Hessian LLE"),
        (manifold_learning.apply_lle(method="modified"), "Modified LLE"),
        # (manifold_learning.apply_lle(method="ltsa"), "LTSA"),
        # (manifold_learning.apply_tsne, "t-SNE"),
        # (manifold_learning.apply_umap, "UMAP"),
        # (manifold_learning.apply_laplacian, "Laplacian Eigenmaps"),
        # (manifold_learning.apply_kernel_pca, "Kernel PCA"),
        # (lambda: manifold_learning.apply_mds(metric=True), "MDS (Metric)"),
        # (lambda: manifold_learning.apply_mds(metric=False), "MDS (Non-Metric)"),
    ]

    embeddings = []
    titles = []

    for method, title in models:
        try:
            embedding = method()
            embeddings.append(embedding)
            titles.append(title)
            print(f"{title} completed")
        except Exception as e:
            print(f"{title} failed: {e}")

    # Visualization
    plot_subplots(embeddings, titles, plot_shape=(1, 3), n_components=n_components)
    plot_2d_locations(locations, title="Original Locations")

    # Compute and print Jaccard similarities
    original_neighbors = get_nearest_neighbors(distance_matrix, k)

    for embedding, title in zip(embeddings, titles):
        reduced_neighbors = get_nearest_neighbors(embedding, k)
        jaccard_similarity = compute_jaccard_similarity(
            original_neighbors, reduced_neighbors, distance_matrix
        )
        print(f"{title} Jaccard Similarity: {jaccard_similarity}")
