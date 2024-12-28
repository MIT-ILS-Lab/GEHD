"""
A module to visualize graph data and routes
"""

import osmnx as ox
import networkx as nx

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
        node_size=1,
        edge_color="magenta",
        edge_linewidth=0.3,
        bgcolor="none",
        save=save,
        show=show,
        filepath=file_path_save,
    )

    return fig, ax


# if __name__ == "__main__":
#     graph = load_graph("src/data/zurich.graphml")
#     visualize_graph(graph, file_path_save="src/data/zurich.png")
