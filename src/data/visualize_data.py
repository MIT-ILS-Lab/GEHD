"""
A module to visualize graph data and routes
"""

import osmnx as ox
import networkx as nx

import matplotlib.pyplot as plt

from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure


from data.generate_data import load_graph


def visualize_graph(
    graph: nx.MultiDiGraph, file_path_save: str | None = None, show: bool = True
) -> tuple[Figure, Axes]:
    """
    Visualize the given graph.

    Args:
        graph: The graph to visualize.
        file_path_save: The file path to save the visualization to.
        show: Whether to show the visualization or not.

    Returns:
        fig: The figure object.
        ax: The axis object.
    """

    save = False
    if file_path_save is not None:
        save = True

    fig, ax = ox.plot_graph(
        graph,
        figsize=(10, 10),
        node_size=0.5,
        edge_color="teal",
        edge_linewidth=0.3,
        bgcolor="none",
        save=save,
        show=show,
        filepath=file_path_save,
    )

    return fig, ax


def visualize_locations(
    graph: nx.MultiDiGraph,
    locations: list[tuple[float, float]],
    depot: tuple[float, float],
    file_path_save: str | None = None,
    show: bool = True,
):
    """
    Visualize the given graph with locations and depot.

    Args:
        graph: The graph to visualize.
        locations: The locations to visualize.
        depot: The depot to visualize.
        file_path_save: The file path to save the visualization to.
        show: Whether to show the visualization or not.

    Returns:
        fig: The figure object.
        ax: The axis object.
    """

    fig, ax = visualize_graph(graph, file_path_save=None, show=False)

    # Plot depot
    if depot:
        nearest_depot = ox.nearest_nodes(graph, Y=depot[0], X=depot[1])
        nearest_depot_x = graph.nodes[nearest_depot]["x"]
        nearest_depot_y = graph.nodes[nearest_depot]["y"]

        ax.scatter(
            depot[1],
            depot[0],
            color="red",
            s=60,
            label="Depot",
            alpha=0.8,
        )
        ax.scatter(
            nearest_depot_x,
            nearest_depot_y,
            color="magenta",
            s=60,
            marker="x",
            label="Closest depot on the grid",
        )

    # Plot customer locations
    if locations:
        nearest_locations = [
            ox.nearest_nodes(graph, Y=loc[0], X=loc[1]) for loc in locations
        ]
        nearest_x = [graph.nodes[loc]["x"] for loc in nearest_locations]
        nearest_y = [graph.nodes[loc]["y"] for loc in nearest_locations]

        x = [location[1] for location in locations]
        y = [location[0] for location in locations]

        ax.scatter(x, y, color="orange", s=60, alpha=0.8, label="Customers")
        ax.scatter(
            nearest_x,
            nearest_y,
            color="gold",
            s=60,
            marker="x",
            label="Closest customer drop off",
        )

    ax.legend()

    if file_path_save is not None:
        fig.savefig(file_path_save)

    if show:
        plt.show()

    return fig, ax


# if __name__ == "__main__":
#     graph = load_graph("src/data/zurich.graphml")
#     locations = [(47.352810, 8.530466), (47.361336, 8.551344), (47.392781, 8.528951)]
#     depot = (47.374267, 8.541208)
#     fig, ax = visualize_locations(
#         graph,
#         locations,
#         depot,
#         show=True,
#         file_path_save="src/data/zurich_customers.png",
#     )
# visualize_graph(graph, file_path_save="src/data/zurich.png")
