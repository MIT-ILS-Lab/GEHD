import pytest
import matplotlib.pyplot as plt
import networkx as nx
from data.visualize_data import visualize_graph, visualize_locations
from data.generate_data import create_graph

from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

from unittest.mock import patch

pytest.fixture


@pytest.fixture
def sample_locations():
    return [(47.352810, 8.530466), (47.361336, 8.551344), (47.392781, 8.528951)]


@pytest.fixture
def sample_depot():
    return (47.374267, 8.541208)


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


def test_visualize_locations_basic(sample_graph, sample_locations, sample_depot):
    # Test basic visualization
    fig, ax = visualize_locations(
        sample_graph, sample_locations, sample_depot, show=False
    )

    # Check return types
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    # Check legend
    assert ax.get_legend() is not None
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "Depot" in legend_texts
    assert "Customers" in legend_texts

    plt.close(fig)


def test_visualize_locations_save(
    sample_graph, sample_locations, sample_depot, tmp_path
):
    # Test save functionality with show=True
    file_path = tmp_path / "test_locations.png"

    with patch("matplotlib.pyplot.show") as mock_show:
        fig, ax = visualize_locations(
            sample_graph,
            sample_locations,
            sample_depot,
            file_path_save=str(file_path),
            show=True,
        )

        # Verify show was called
        assert mock_show.called
        # Verify file was saved
        assert file_path.exists()
        assert file_path.stat().st_size > 0

        plt.close(fig)
