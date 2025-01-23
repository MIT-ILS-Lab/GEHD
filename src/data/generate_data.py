"""
A module to generate the reallife data set using osmnx (ox).
Uses vectorized shortest_path for efficiency.
Considers only asymmetric distances.
"""

import numpy as np
import osmnx as ox
import networkx as nx

from typing import Literal
from shapely import Polygon, MultiPolygon

from tqdm import tqdm
import time

from utils import load_config, sample_points, load_graph


def create_graph(
    graph_area: (
        str | dict[str, str] | list[str | dict[str, str]] | Polygon | MultiPolygon
    ),
    file_path: str = "",
    area_type: Literal["location", "polygon"] = "location",
    simplify: bool = False,
):
    """Creates a graph from a location or polygon using osmnx."""
    if area_type not in ["location", "polygon"]:
        raise ValueError(f"area_type must be 'location' or 'polygon', got {area_type}")

    if area_type == "location":
        assert isinstance(
            graph_area, (str, dict, list)
        ), "graph_area must be str, dict, or list for location"
        G = ox.graph_from_place(graph_area, network_type="drive", simplify=simplify)
    else:
        assert isinstance(
            graph_area, (Polygon, MultiPolygon)
        ), "graph_area must be Polygon or MultiPolygon"
        G = ox.graph_from_polygon(graph_area, network_type="drive", simplify=simplify)

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Reduce graph to biggest strongly connected component
    scc = max(nx.strongly_connected_components(G), key=len)
    G.remove_nodes_from(set(G.nodes) - scc)

    if file_path:
        ox.save_graphml(G, file_path)
    return G


def compute_travel_times_vectorized(G, origins, destinations):
    """Computes travel times using vectorized shortest_path."""
    try:
        routes = ox.shortest_path(
            G, origins, destinations, weight="travel_time", cpus=None
        )  # Let ox handle CPU usage
        travel_times = []
        for route in routes:
            if route:
                if len(route) == 1:
                    travel_times.append(0)
                else:
                    travel_times.append(
                        sum(
                            ox.routing.route_to_gdf(G, route, weight="travel_time")[
                                "travel_time"
                            ]
                        )
                    )
            else:
                travel_times.append(np.inf)
        return np.array(travel_times)
    except Exception as e:
        print(f"Error in vectorized routing: {e}")
        return np.full(len(origins), np.inf)  # Return infinities in case of error


def generate_distance_matrix(G, points):
    """Generates the *asymmetric* distance matrix using vectorized routing."""
    n = len(points)
    D_mat = np.full((n, n), np.inf)

    for i in tqdm(range(n), desc="Generating Distance Matrix"):
        origins = [points[i]] * n  # Repeat the origin n times
        destinations = points  # All points as destinations
        travel_times = compute_travel_times_vectorized(G, origins, destinations)
        D_mat[i, :] = travel_times  # Assign to the i-th row
    return D_mat


if __name__ == "__main__":
    config = load_config("config.yaml")
    graph = create_graph(config["data"]["location"], config["data"]["graph_file"])

    points = sample_points(graph, config["data"]["num_customers"] + 1)

    start_time = time.time()

    D_mat = generate_distance_matrix(graph, points)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    np.save(config["data"]["matrix_file"], D_mat)
    np.save(config["data"]["locations_file"], points)
    print("Distance matrix and locations saved.")
