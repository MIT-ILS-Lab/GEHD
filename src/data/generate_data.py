"""
A module to generate the reallife data set
"""

import numpy as np
import osmnx as ox
import networkx as nx

from typing import Literal
from shapely import Polygon, MultiPolygon

from tqdm import tqdm
from multiprocessing import Pool, Manager
from functools import partial

from utils import load_config, sample_points, load_graph


def create_graph(
    graph_area: (
        str | dict[str, str] | list[str | dict[str, str]] | Polygon | MultiPolygon
    ),
    file_path: str = "",
    area_type: Literal["location", "polygon"] = "location",
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
    if type == "coords":
        source = ox.nearest_nodes(graph, Y=source[0], X=source[1])  # type: ignore
        target = ox.nearest_nodes(graph, Y=target[0], X=target[1])  # type: ignore

    return nx.shortest_path_length(graph, source, target, weight="travel_time")


def generate_distance_matrix_worker(params):
    """
    Worker function for multiprocessing to compute a distance matrix entry.

    Args:
        params: A tuple containing graph, source, target, and type.

    Returns:
        A tuple of indices and the computed travel time.
    """
    graph, i, j, source, target, node_type = params
    travel_time = compute_edge_travel_times(graph, source, target, node_type)
    return (i, j, travel_time)


def compute_distance_task(
    task, graph: nx.MultiDiGraph, node_type: Literal["id", "coords"]
):
    """
    Compute travel time for a given source and target pair.

    Args:
        task: Tuple containing (source, target, source_id, target_id symmetric_flag)
        graph: The graph to compute the travel time on
        node_type: Type of node representation. It can be either id or coords.

    Returns:
        Tuple containing (source_index, target_index, travel_time)
    """
    source, target, source_id, target_id, symmetric = task

    if symmetric:
        travel_time = 0.5 * (
            compute_edge_travel_times(graph, source, target, node_type)
            + compute_edge_travel_times(graph, target, source, node_type)
        )
    else:
        travel_time = compute_edge_travel_times(graph, source, target, node_type)

    return source_id, target_id, travel_time


from functools import partial


def generate_distance_matrix(
    graph: nx.MultiDiGraph,
    locations: list[tuple[float, float]] | list[int],
    depot: tuple[float, float] | int,
    distance_type: Literal["symmetric", "asymmetric"],
    node_type: Literal["id", "coords"] = "coords",
    use_multiprocessing: bool = True,
) -> np.ndarray:
    """
    Generate the distance matrix D for the given set of customer locations using multiprocessing.

    Args:
        graph: The graph to compute the travel time on
        locations: List of tuples of coordinates (y,x) (lat,long) representing the different locations
        depot: Depot location (y,x) (lat,long) to serve the locations
        distance_type: Type of distances. It can be either symmetric or asymmetric.
        node_type: Type of node representation. It can be either be id or coords.
        use_multiprocessing: Whether to use multiprocessing for computation

    Returns:
        np.ndarray: Distance matrix with travel times between all locations and the depot.
    """

    D_mat = np.zeros((len(locations) + 1, len(locations) + 1))

    if distance_type not in ["symmetric", "asymmetric"]:
        raise ValueError(
            f"distance_type must be 'symmetric' or 'asymmetric', got {distance_type}"
        )

    # Convert coordinates to graph nodes if node_type is "coords"
    if node_type == "coords":
        locations = [ox.nearest_nodes(graph, X=loc[1], Y=loc[0]) for loc in locations]  # type: ignore
        depot = ox.nearest_nodes(graph, X=depot[1], Y=depot[0])  # type: ignore

    # Validate that all nodes exist in the graph
    all_nodes = set(graph.nodes)

    # Prepare tasks for multiprocessing with actual locations and depot
    tasks = []
    for i in range(len(locations)):
        for j in range(i if distance_type == "symmetric" else 0, len(locations)):
            if i != j:
                tasks.append(
                    (locations[i], locations[j], i, j, distance_type == distance_type)
                )

        # Include depot for both symmetric and asymmetric calculations
        if distance_type == "symmetric":
            tasks.append((depot, locations[i], 0, i, distance_type == distance_type))
        else:
            tasks.append((depot, locations[i], 0, i, distance_type == distance_type))
            tasks.append((locations[i], depot, i, 0, distance_type == distance_type))

    total_tasks = len(tasks)

    # Process tasks with or without multiprocessing
    if use_multiprocessing:
        with Pool() as pool, tqdm(
            total=total_tasks, desc="Generating Distance Matrix"
        ) as pbar:
            func = partial(compute_distance_task, graph=graph, node_type=node_type)

            for result in pool.imap_unordered(func, tasks):
                source_id, target_id, travel_time = result
                if distance_type == "symmetric":
                    D_mat[source_id, target_id] = travel_time
                    D_mat[target_id, source_id] = travel_time
                else:
                    D_mat[source_id, target_id] = travel_time
                pbar.update(1)
    else:
        for task in tqdm(tasks, desc="Generating Distance Matrix"):
            source, target, travel_time = compute_distance_task(
                task, graph=graph, node_type=node_type
            )
            D_mat[source, target] = travel_time

    return D_mat


if __name__ == "__main__":
    config = load_config("config.yaml")
    # graph = create_graph(config["data"]["location"], config["data"]["graph_file"])
    graph = ox.load_graphml(config["data"]["graph_file"])
    points = sample_points(graph, config["data"]["num_points"] + 1)
    locations = points[1:]
    depot = points[0]

    use_multiprocessing = config.get("use_multiprocessing", True)

    D_mat = generate_distance_matrix(
        graph,
        locations,
        depot,
        distance_type="symmetric",
        node_type="id",
        use_multiprocessing=use_multiprocessing,
    )
    np.save(config["data"]["matrix_file"], D_mat)
    np.save(config["data"]["locations_file"], [depot] + locations)
