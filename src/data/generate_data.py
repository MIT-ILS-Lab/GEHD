import igraph as ig
import networkx as nx
import numpy as np
from typing import Literal
from shapely import Polygon, MultiPolygon
from utils import load_config, sample_points, load_graph, convert_nx_to_ig
import osmnx as ox


def create_graph(
    graph_area: (
        str | dict[str, str] | list[str | dict[str, str]] | Polygon | MultiPolygon
    ),
    file_path: str = "",
    area_type: Literal["location", "polygon"] = "location",
    simplify: bool = False,
) -> nx.MultiDiGraph:
    """Creates a graph from a location or polygon using osmnx."""
    if area_type not in ["location", "polygon"]:
        raise ValueError(f"area_type must be 'location' or 'polygon', got {area_type}")

    if area_type == "location":
        graph = ox.graph_from_place(graph_area, network_type="drive", simplify=simplify)  # type: ignore
    else:
        graph = ox.graph_from_polygon(
            graph_area, network_type="drive", simplify=simplify  # type: ignore
        )

    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)

    # Reduce to the largest strongly connected component
    scc = max(nx.strongly_connected_components(graph), key=len)
    graph = graph.subgraph(scc).copy()

    if file_path:
        ox.save_graphml(graph, file_path)

    return graph


def compute_travel_times_igraph(G: ig.Graph, origins: list, destinations: list):
    """Computes travel times using igraph's distances method."""
    origin_indices = [G.vs.find(name=str(origin)).index for origin in origins]
    destination_indices = [
        G.vs.find(name=str(destination)).index for destination in destinations
    ]

    # Compute pairwise distances
    travel_times = G.distances(
        source=origin_indices, target=destination_indices, weights="travel_time"
    )
    return np.array(travel_times)


def generate_distance_matrix(G: ig.Graph, points: list):
    """Generates the *asymmetric* distance matrix using igraph."""
    print("Generating Distance Matrix...")
    D_mat = compute_travel_times_igraph(G, points, points)
    print("Done generating Distance Matrix")
    return D_mat


def generate_symmetric_matrix(D_mat: np.ndarray, normalize: bool = False):
    """Generates a symmetric matrix by averaging distances between i to j and j to i."""
    print("Generating Symmetric Matrix...")
    symmetric_D_mat = (D_mat + D_mat.T) / 2
    if normalize:
        symmetric_D_mat = symmetric_D_mat / symmetric_D_mat.max()
    print("Done generating Symmetric Matrix")
    return symmetric_D_mat


if __name__ == "__main__":
    config = load_config("config.yaml")
    graph = create_graph(config["data"]["location"], config["data"]["graph_file"])
    # graph = load_graph(config["data"]["graph_file"])

    graph = convert_nx_to_ig(graph)

    points = sample_points(graph, config["data"]["num_customers"] + 1)

    D_mat = generate_distance_matrix(graph, points)

    D_mat = generate_symmetric_matrix(D_mat, normalize=True)

    np.save(config["data"]["matrix_file"], D_mat)
    np.save(config["data"]["locations_file"], points)
    print("Distance matrix and locations saved.")
