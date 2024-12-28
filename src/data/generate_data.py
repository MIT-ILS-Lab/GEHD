"""
A module to generate the reallife data set
"""

import numpy as np
import osmnx as ox
import networkx as nx

from typing import Literal
from shapely import Polygon, MultiPolygon


def create_graph(
    graph_area: (
        str | dict[str, str] | list[str | dict[str, str]] | Polygon | MultiPolygon
    ),
    area_type: Literal["location", "polygon"] = "location",
    file_path: str = "",
    simplify: bool = False,
):
    """
    Load the underlying graph from the given file path. This graph is then used as the base to calculate the distances between coordinates.

    Args:
        graph_area: The area to load the graph from. It can be either a location name, a polygon or a multipolygon.
        area_type: The type of the area to load the graph from. It can be either a location or a polygon shape.
        file_path: The file path to save the graph to.
        simplify: Whether to simplify the graph or not.
    """

    if area_type == "location":
        assert isinstance(
            graph_area, (str, dict, list)
        ), "graph_area cannot be a polygon, please use a location name instead"
        graph = ox.graph_from_place(graph_area, network_type="drive", simplify=simplify)
    else:
        assert isinstance(
            graph_area, (Polygon, MultiPolygon)
        ), "graph_area must be a polygon"
        graph = ox.graph_from_polygon(
            graph_area, network_type="drive", simplify=simplify
        )

    graph = ox.add_edge_speeds(graph)  # Adding missing edge speeds
    graph = ox.add_edge_travel_times(graph)  # Adding missing edge travel times

    if file_path is not "":
        ox.save_graphml(graph, file_path)

    return graph


def compute_edge_travel_times(
    graph: nx.MultiDiGraph, source: tuple[float, float], target: tuple[float, float]
) -> float:
    """
    Compute the travel time between the source and target nodes in the graph.

    Args:
        graph: The graph to compute the travel time on
        source: The source node coordinates (x,y)
        target: The target node coordinates (x,y)

    Returns:
        The travel time between the source and target nodes
    """
    source_node = ox.nearest_nodes(graph, X=source[0], Y=source[1])
    target_node = ox.nearest_nodes(graph, X=target[0], Y=target[1])

    travel_time = nx.shortest_path_length(
        graph, source_node, target_node, weight="travel_time"
    )

    return travel_time


def generate_distance_matrix(
    graph: nx.MultiDiGraph,
    locations: list[tuple[float, float]],
    depot: tuple[float, float],
    type: Literal["symmetric", "asymmetric"],
) -> np.ndarray:
    """
    Generate the distance matrix D for the given set of customer locations.

    The distance matrix D has entries D_{i,j} representing the travel time between location i and location j. If the type is symmetric, than this is the mean travel between i and j as well as j and i.

    Args:
        graph: The graph to compute the travel time on
        locations: List of tuples of coordinates (x,y) representing the different locations
        depot: Depot location (x,y) to serve the locations
        type: Type of distances. It can be either symmetric or asymmetric.
    """

    D_mat = np.zeros((len(locations) + 1, len(locations) + 1))

    # We put the depot at the first element in D_mat
    if type == "symmetric":
        for i in range(len(locations)):
            for j in range(i, len(locations)):
                travel_time = 0.5 * (
                    compute_edge_travel_times(graph, locations[i], locations[j])
                    + compute_edge_travel_times(graph, locations[j], locations[i])
                )
                D_mat[i + 1, j + 1] = travel_time
                D_mat[j + 1, i + 1] = travel_time

        for i in range(len(locations)):
            travel_time_to_depot = 0.5 * (
                compute_edge_travel_times(graph, depot, locations[i])
                + compute_edge_travel_times(graph, locations[i], depot)
            )
            D_mat[0, i + 1] = travel_time_to_depot
            D_mat[i + 1, 0] = travel_time_to_depot

    elif type == "asymmetric":
        for i in range(len(locations)):
            for j in range(len(locations)):
                if i != j:
                    D_mat[i + 1, j + 1] = compute_edge_travel_times(
                        graph, locations[i], locations[j]
                    )

        for i in range(len(locations)):
            D_mat[0, i + 1] = compute_edge_travel_times(graph, depot, locations[i])
            D_mat[i + 1, 0] = compute_edge_travel_times(graph, locations[i], depot)

    return D_mat
