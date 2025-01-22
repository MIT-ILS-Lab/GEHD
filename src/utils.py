"""
Utility functions for the project.
"""

import numpy as np
import osmnx as ox
import networkx as nx

import yaml

from typing import Literal
from shapely import Polygon, MultiPolygon


def load_config(config_file):
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


def sample_points(graph: nx.MultiDiGraph, num_points: int) -> list[int]:
    """
    Sample points from the graph.

    Args:
        num_points: The number of points to sample.
        graph: The graph to sample the points from.

    Returns:
        nodes: The sampled node IDs from the underlying graph.
    """
    nodes = np.random.choice(list(graph.nodes), num_points, replace=False)

    # nodes = [(graph.nodes[node]["y"], graph.nodes[node]["x"]) for node in nodes]

    return nodes  # type: ignore
