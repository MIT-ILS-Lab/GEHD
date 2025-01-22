"""
A module to generate the reallife data set
"""

import numpy as np
import osmnx as ox
import networkx as nx

from typing import Literal
from shapely import Polygon, MultiPolygon

from utils import load_config, sample_points


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

    if area_type not in ["location", "polygon"]:
        raise ValueError(f"area_type must be 'location' or 'polygon', got {area_type}")

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

    if file_path != "":
        ox.save_graphml(graph, file_path)

    return graph


def load_graph(file_path: str) -> nx.MultiDiGraph:
    """
    Load the graph from the given file path.

    Args:
        file_path: The file path to load the graph from.
    """
    graph = ox.load_graphml(file_path)
    return graph


def compute_edge_travel_times(
    graph: nx.MultiDiGraph,
    source: tuple[float, float] | int,
    target: tuple[float, float] | int,
    type: Literal["id", "coords"] = "coords",
) -> float:
    """
    Compute the travel time between the source and target nodes in the graph.

    Args:
        graph: The graph to compute the travel time on
        source: The source node id or coordinates (y,x) (lat,long)
        target: The target node id or coordinates (y,x) (lat,long)
        type: The type of the source and target nodes. It can be either id or coordinates

    Returns:
        The travel time between the source and target nodes
    """
    assert type in ["id", "coords"], f"type must be 'id' or 'coords', got {type}"

    if type == "coords":
        source = ox.nearest_nodes(graph, Y=source[0], X=source[1])  # type: ignore
        target = ox.nearest_nodes(graph, Y=target[0], X=target[1])  # type: ignore

    travel_time = nx.shortest_path_length(graph, source, target, weight="travel_time")

    return travel_time


def generate_distance_matrix(
    graph: nx.MultiDiGraph,
    locations: list[tuple[float, float]] | list[int],
    depot: tuple[float, float] | int,
    distance_type: Literal["symmetric", "asymmetric"],
    node_type: Literal["id", "coords"] = "coords",
) -> np.ndarray:
    """
    Generate the distance matrix D for the given set of customer locations.

    The distance matrix D has entries D_{i,j} representing the travel time between location i and location j. If the type is symmetric, than this is the mean travel between i and j as well as j and i.

    Args:
        graph: The graph to compute the travel time on
        locations: List of tuples of coordinates (y,x) (lat,long) representing the different locations
        depot: Depot location (y,x) (lat,long) to serve the locations
        type: Type of distances. It can be either symmetric or asymmetric.
        node_type: Type of node representation. It can be either be id or coords.
    """

    D_mat = np.zeros((len(locations) + 1, len(locations) + 1))

    if distance_type not in ["symmetric", "asymmetric"]:
        raise ValueError(
            f"distance_type must be 'symmetric' or 'asymmetric', got {distance_type}"
        )

    # We put the depot at the first element in D_mat
    if distance_type == "symmetric":
        for i in range(len(locations)):
            for j in range(i, len(locations)):
                if i != j:
                    travel_time = 0.5 * (
                        compute_edge_travel_times(
                            graph, locations[i], locations[j], node_type
                        )
                        + compute_edge_travel_times(
                            graph, locations[j], locations[i], node_type
                        )
                    )
                    D_mat[i + 1, j + 1] = travel_time
                    D_mat[j + 1, i + 1] = travel_time

        for i in range(len(locations)):
            travel_time_to_depot = 0.5 * (
                compute_edge_travel_times(graph, depot, locations[i], node_type)
                + compute_edge_travel_times(graph, locations[i], depot, node_type)
            )
            D_mat[0, i + 1] = travel_time_to_depot
            D_mat[i + 1, 0] = travel_time_to_depot

    elif distance_type == "asymmetric":
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


if __name__ == "__main__":
    #     # Example for the location Zurich, Switzerland
    config = load_config("config.yaml")
    graph = create_graph(
        config["data"]["location"],
        file_path=config["data"]["graph_file"],
        simplify=True,
    )
    graph = load_graph(config["data"]["graph_file"])
    locations = sample_points(graph, config["data"]["num_points"])
    depot = sample_points(graph, 1)[0]
    #     locations = [(47.352810, 8.530466), (47.361336, 8.551344), (47.392781, 8.528951)]
    #     depot = (47.374267, 8.541208)
    D_mat = generate_distance_matrix(
        graph, locations, depot, distance_type="symmetric", node_type="id"
    )
    # save the distance matrix and node choices in a good file format
    np.save(config["data"]["matrix_file"], D_mat)
    np.save(config["data"]["locations_file"], locations)
