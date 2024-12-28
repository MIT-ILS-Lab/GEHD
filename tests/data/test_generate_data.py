import pytest
import numpy as np
import networkx as nx
from shapely.geometry import Polygon
from data.generate_data import (
    create_graph,
    load_graph,
    compute_edge_travel_times,
    generate_distance_matrix,
)

pytest.fixture


@pytest.fixture
def sample_location_name():
    return "Zurich, Switzerland"


@pytest.fixture
def sample_graph(sample_location_name):
    return create_graph(sample_location_name, simplify=True)


@pytest.fixture
def sample_locations():
    return [(47.352810, 8.530466), (47.361336, 8.551344), (47.392781, 8.528951)]


@pytest.fixture
def sample_polygon():
    return Polygon([(8.5, 47.3), (8.6, 47.3), (8.6, 47.4), (8.5, 47.4)])


@pytest.fixture
def sample_depot():
    return (47.374267, 8.541208)


def test_create_graph_with_location(sample_location_name):
    graph = create_graph(sample_location_name)
    assert isinstance(graph, nx.MultiDiGraph)
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0


def test_create_graph_with_polygon(sample_polygon):
    graph = create_graph(sample_polygon, area_type="polygon")
    assert isinstance(graph, nx.MultiDiGraph)
    assert len(graph.nodes) > 0


def test_load_graph(tmp_path, sample_location_name):
    file_path = tmp_path / "test_graph_zurich.graphml"
    _ = create_graph(sample_location_name, file_path=str(file_path))
    loaded_graph = load_graph(str(file_path))
    assert isinstance(loaded_graph, nx.MultiDiGraph)
    assert len(loaded_graph.nodes) > 0


def test_compute_edge_travel_times(sample_graph, sample_locations, sample_depot):
    source = sample_depot
    target = sample_locations[0]
    travel_time = compute_edge_travel_times(sample_graph, source, target)
    assert travel_time > 0
    assert isinstance(travel_time, float)


def test_generate_distance_matrix_symmetric(
    sample_graph, sample_locations, sample_depot
):
    D_mat = generate_distance_matrix(
        sample_graph, sample_locations, sample_depot, distance_type="symmetric"
    )
    assert isinstance(D_mat, np.ndarray)
    assert D_mat.shape == (len(sample_locations) + 1, len(sample_locations) + 1)
    # Test symmetry
    assert np.allclose(D_mat, D_mat.T)


def test_generate_distance_matrix_asymmetric(
    sample_graph, sample_locations, sample_depot
):
    D_mat = generate_distance_matrix(
        sample_graph, sample_locations, sample_depot, distance_type="asymmetric"
    )
    assert isinstance(D_mat, np.ndarray)
    assert D_mat.shape == (len(sample_locations) + 1, len(sample_locations) + 1)


def test_invalid_area_type(sample_location_name):
    # Test invalid combination of polygon with location type
    with pytest.raises(AssertionError):
        create_graph(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), area_type="location")

    # Test invalid area_type value
    with pytest.raises(ValueError):
        create_graph(sample_location_name, area_type="invalid")


def test_invalid_matrix_type(sample_graph, sample_locations, sample_depot):
    with pytest.raises(ValueError):
        generate_distance_matrix(
            sample_graph, sample_locations, sample_depot, distance_type="invalid"
        )
