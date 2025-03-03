"""
Utility functions for the project using igraph.
"""

import numpy as np
import igraph as ig
import yaml
import osmnx as ox
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def load_config(config_file: str) -> dict:
    """
    Loads a YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_graph(file_path: str) -> nx.MultiDiGraph:
    """
    Load the graph from the given file path.

    Args:
        file_path: The file path to load the graph from.
    """
    graph = ox.load_graphml(file_path)
    return graph


def load_manifold_data(config):
    distance_matrix = np.load(config["manifold"]["matrix_file"])
    graph = load_graph(config["manifold"]["graph_file"])
    locations = np.load(config["manifold"]["locations_file"])

    # Get (y, x) coordinates fo the locations
    locations = [
        [graph.nodes[location]["y"], graph.nodes[location]["x"]]
        for location in graph.nodes
    ]
    locations = np.array(locations)

    return distance_matrix, graph, locations


def sample_points(graph: ig.Graph, num_points: int) -> list[int]:
    """
    Sample points (vertices) from the igraph graph.

    Args:
        graph: The igraph graph to sample the points from.
        num_points: The number of points to sample.

    Returns:
        list[int]: The sampled vertex IDs from the graph.
    """
    assert (
        num_points <= graph.vcount()
    ), "num_points must be less than the number of vertices in the graph"

    # Randomly sample vertex IDs (indices in igraph)
    nodes = np.random.choice(graph.vs.indices, num_points, replace=False)

    return nodes.tolist()


def convert_nx_to_ig(G_nx: nx.MultiDiGraph) -> ig.Graph:
    """Converts a networkx graph to igraph."""
    osmids = list(G_nx.nodes)

    # give each node its original osmid as attribute since we relabeled them
    osmid_values = {k: v for k, v in zip(G_nx.nodes, osmids)}
    nx.set_node_attributes(G_nx, osmid_values, "osmid")

    G_nx = nx.relabel.convert_node_labels_to_integers(G_nx)

    # Convert networkx graph to igraph
    G_ig = ig.Graph(directed=True)
    G_ig.add_vertices(list(G_nx.nodes))
    G_ig.add_edges(list(G_nx.edges()))
    G_ig.vs["name"] = [str(node) for node in G_nx.nodes]
    G_ig.vs["osmid"] = osmids
    G_ig.es["travel_time"] = list(nx.get_edge_attributes(G_nx, "travel_time").values())

    return G_ig


def get_nearest_neighbors(vectors: np.ndarray, k: int) -> np.ndarray:
    """Returns k-nn neighbors of each point in the dataset."""
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(vectors)
    _, indices = nn.kneighbors(vectors)
    return indices
