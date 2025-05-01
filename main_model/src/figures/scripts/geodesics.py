import numpy as np
import trimesh
import polyscope as ps
from pygeodesic import geodesic
import matplotlib.pyplot as plt

# --- Colors ---
pink = [0.96, 0, 0.42]
cyan = [0.22, 0.827, 0.835]
blue = [0.216, 0.522, 0.882]

# --- Parameters ---
subdivisions = 2
mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
vertices = mesh.vertices
faces = mesh.faces


# --- Compute geodesic distances from node 0 ---
def compute_geodesic_distances(vertices, faces, source_index=0):
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
    source_indices = np.array([source_index])
    target_indices = None
    distances, _ = geoalg.geodesicDistances(source_indices, target_indices)
    return distances


geodesic_distances = compute_geodesic_distances(vertices, faces, source_index=0)
norm_geodist = geodesic_distances / np.max(geodesic_distances)

# --- Compute Euclidean distances from node 0 ---
euclidean_distances = np.linalg.norm(vertices - vertices[0], axis=1)
norm_eucdist = euclidean_distances / np.max(euclidean_distances)


# --- Make the gradient more cyan-heavy by applying a power curve ---
norm_dist = norm_geodist
power = 0.8  # Lower means more cyan, higher means more pink
adjusted = norm_dist**power

# --- Use a custom colormap: from cyan to pink ---
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list(
    "cyan_pink",
    [
        (0.0, cyan),  # Cyan
        (0.5, blue),  # Vibrant blue (optional)
        (0.8, pink),  # Pink
        (1.0, pink),  # Pink
    ],
)

node_colors = cmap(adjusted)[:, :3]  # Drop alpha

# --- Visualize with Polyscope ---
ps.init()
ps_mesh = ps.register_surface_mesh("trimesh_mesh", vertices, faces)
ps_mesh.set_material("clay")
ps_mesh.set_edge_width(1.05)
ps.set_ground_plane_mode("none")
ps_mesh.add_color_quantity("geodesic distance", node_colors, enabled=True)
ps.show()
