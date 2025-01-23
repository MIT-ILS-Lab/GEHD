"""
Utility functions for the project using igraph.
"""

import numpy as np
import igraph as ig
import yaml


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


def load_graph(file_path: str) -> ig.Graph:
    """
    Load an igraph object from a file.

    Args:
        file_path: The file path to load the graph from.

    Returns:
        ig.Graph: The loaded igraph object.
    """
    graph = ig.Graph.Read_Pickle(file_path)
    return graph


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
