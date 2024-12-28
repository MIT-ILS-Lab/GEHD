import pytest
import matplotlib.pyplot as plt
import networkx as nx
from data.visualize_data import visualize_graph
from data.generate_data import create_graph

from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

pytest.fixture


@pytest.fixture
def sample_graph():
    return create_graph("Zurich, Switzerland", simplify=True)


@pytest.fixture
def temp_file_path(tmp_path):
    return tmp_path / "test_graph.png"


def test_visualize_graph_basic(sample_graph):
    # Test basic visualization
    fig, ax = visualize_graph(sample_graph, show=False)

    # Check return types
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    # Check figure properties
    assert ax.get_xbound() != 0 and ax.get_ybound() != 0  # Has axis limits

    plt.close(fig)  # Cleanup


def test_visualize_graph_save(sample_graph, temp_file_path):
    # Test save functionality
    fig, ax = visualize_graph(
        sample_graph, file_path_save=str(temp_file_path), show=False
    )

    # Check if file was created
    assert temp_file_path.exists()
    assert temp_file_path.stat().st_size > 0

    plt.close(fig)  # Cleanup


def test_visualize_graph_properties(sample_graph):
    # Test specific visualization properties
    fig, ax = visualize_graph(sample_graph, show=False)

    # Check edge properties
    edges = ax.collections[0]  # First collection should be edges
    assert edges.get_linewidth() == [0.3]

    plt.close(fig)  # Cleanup
