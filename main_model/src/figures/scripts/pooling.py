import numpy as np
import trimesh
import polyscope as ps
from scipy.spatial import Delaunay

# Parameters
width = 7
height = 10
n_x = 8  # Number of points along x (increase for finer mesh)
n_y = 10  # Number of points along y
z_top = 0
# z_top = 0.4
z_bottom = 0

# Generate grid points for top and bottom
x = np.linspace(-width / 2, width / 2, n_x)
y = np.linspace(-height / 2, height / 2, n_y)
xx, yy = np.meshgrid(x, y)
zz_top = np.full_like(xx, z_top)
zz_bottom = np.full_like(xx, z_bottom)

# Flatten and stack for vertices
vertices_top = np.column_stack([xx.ravel(), yy.ravel(), zz_top.ravel()])
vertices_bottom = np.column_stack([xx.ravel(), yy.ravel(), zz_bottom.ravel()])

# Combine vertices
vertices = np.vstack([vertices_top, vertices_bottom])
n_vertices_per_layer = vertices_top.shape[0]

# Triangulate the grid (2D Delaunay on x, y)
tri = Delaunay(np.column_stack([xx.ravel(), yy.ravel()]))
faces_top = tri.simplices
faces_bottom = tri.simplices + n_vertices_per_layer  # Offset for bottom layer


# Create side faces by connecting corresponding border vertices
def border_indices(n_x, n_y):
    # Returns the indices of the outer border in order
    idx = []
    # Top row
    idx.extend(range(n_x))
    # Right column
    idx.extend([i * n_x + (n_x - 1) for i in range(1, n_y - 1)])
    # Bottom row (reversed)
    idx.extend(range(n_x * (n_y - 1) + n_x - 1, n_x * (n_y - 1) - 1, -1))
    # Left column (reversed)
    idx.extend([i * n_x for i in range(n_y - 2, 0, -1)])
    return np.array(idx)


border = border_indices(n_x, n_y)
n_border = len(border)

side_faces = []
for i in range(n_border):
    i_top = border[i]
    i_top_next = border[(i + 1) % n_border]
    i_bottom = i_top + n_vertices_per_layer
    i_bottom_next = i_top_next + n_vertices_per_layer
    # Two triangles per quad
    side_faces.append([i_top, i_top_next, i_bottom])
    side_faces.append([i_top_next, i_bottom_next, i_bottom])

# Combine all faces
faces = np.vstack([faces_top, faces_bottom, np.array(side_faces)])

# Create mesh
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# Visualize
# sandwich_mesh.show(viewer="gl", flags={"wireframe": True})


pink = [0.96, 0, 0.42]
cyan = [0.22, 0.827, 0.835]
blue = [0.216, 0.522, 0.882]

# Initialize Polyscope
ps.init()

# # Register the mesh with Polyscope
# ps_mesh = ps.register_surface_mesh("trimesh_mesh", mesh.vertices, mesh.faces)

# # --- Color Faces ---
# ps_mesh.set_color(cyan)

# # --- Set Edge Width ---
# edge_width = 1.05
# ps_mesh.set_edge_width(edge_width)


# # --- Set Material to 'clay' ---
# ps_mesh.set_material("clay")

# --- Register Curve Network for Edges ---
# curve_net = ps.register_curve_network("edges_only", mesh.vertices, mesh.edges_unique)
# curve_net.set_color([0, 0, 0])
# curve_net.set_radius(0.0006)
# curve_net.set_transparency(0.95)

# --- Highlight edge between node 48 and 348 ---
# --- Register curve network for vertex 48 in pink ---
# nodes_48 = np.array([mesh.vertices[48]])
# edges_48 = np.array([[0, 0]])  # self-loop on the only node
# ps_48 = ps.register_curve_network(
#     "vertex_48", nodes_48, edges_48, color=pink, radius=0.012
# )

# # --- Register curve network for vertex 348 in cyan ---
# nodes_348 = np.array([mesh.vertices[348]])
# edges_348 = np.array([[0, 0]])  # self-loop on the only node
# ps_348 = ps.register_curve_network(
#     "vertex_348", nodes_348, edges_348, color=cyan, radius=0.012
# )


# For poolings
curve_net = ps.register_curve_network("edges_only", mesh.vertices, mesh.edges_unique)
curve_net.set_color([0.33, 0.33, 0.33])
curve_net.set_radius(0.0025)
curve_net.set_transparency(0.95)

# Highlight all nodes
ps_nodes = ps.register_point_cloud("vertices", mesh.vertices, color=blue, radius=0.012)

# nodes_5 = np.array([mesh.vertices[5]])
# edges_5 = np.array([[0, 0]])  # self-loop on the only node
# ps_5 = ps.register_curve_network("vertex_5", nodes_5, edges_5, color=pink, radius=0.012)

# --- Register curve network for vertex 25 in cyan ---
# nodes_25 = np.array([mesh.vertices[25]])
# edges_25 = np.array([[0, 0]])  # self-loop on the only node
# ps_25 = ps.register_curve_network(
#     "vertex_25", nodes_25, edges_25, color=cyan, radius=0.012
# )

nodes_18 = np.array([mesh.vertices[18]])
edges_18 = np.array([[0, 0]])  # self-loop on the only node
ps_18 = ps.register_curve_network(
    "vertex_18", nodes_18, edges_18, color=pink, radius=0.012
)


# --- Disable Ground Plane ---
ps.set_ground_plane_mode("none")

# Show the viewer
ps.show()
