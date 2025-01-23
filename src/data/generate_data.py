import igraph as ig
import networkx as nx
import numpy as np
from tqdm import tqdm
import time
from typing import Literal
from shapely import Polygon, MultiPolygon
from utils import load_config, sample_points


def create_graph(
    graph_area: (
        str | dict[str, str] | list[str | dict[str, str]] | Polygon | MultiPolygon
    ),
    file_path: str = "",
    area_type: Literal["location", "polygon"] = "location",
    simplify: bool = False,
):
    """Creates a graph from a location or polygon using osmnx and converts to igraph."""
    import osmnx as ox

    if area_type not in ["location", "polygon"]:
        raise ValueError(f"area_type must be 'location' or 'polygon', got {area_type}")

    if area_type == "location":
        G_nx = ox.graph_from_place(graph_area, network_type="drive", simplify=simplify)
    else:
        G_nx = ox.graph_from_polygon(
            graph_area, network_type="drive", simplify=simplify
        )

    G_nx = ox.add_edge_speeds(G_nx)
    G_nx = ox.add_edge_travel_times(G_nx)

    # Reduce to the largest strongly connected component
    scc = max(nx.strongly_connected_components(G_nx), key=len)
    G_nx = G_nx.subgraph(scc).copy()

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

    if file_path:
        G_ig.write_pickle(file_path)

    return G_ig


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


def generate_symmetric_matrix(D_mat: np.ndarray):
    """Generates a symmetric matrix by averaging distances between i to j and j to i."""
    print("Generating Symmetric Matrix...")
    symmetric_D_mat = (D_mat + D_mat.T) / 2
    return symmetric_D_mat


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
