import numpy as np
import trimesh
import polyscope as ps

# subdivisions = 2
# mesh = trimesh.creation.icosphere(subdivisions=subdivisions)

# Vertices: center, top, bottom, and two fans
vertices = np.array(
    [
        [0, 0, 0],
        [1, -1, 0],
        [2, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
    ]
)

# Triangles (fans above and below center vertex)
faces = np.array(
    [
        [0, 2, 1],
        [0, 2, 3],
        [0, 2, 4],
    ]
)

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

pink = [0.96, 0, 0.42]
cyan = [0.22, 0.827, 0.835]

# Initialize Polyscope
ps.init()

# Register the mesh with Polyscope
ps_mesh = ps.register_surface_mesh("trimesh_mesh", mesh.vertices, mesh.faces)

# --- Color Faces ---
ps_mesh.set_color(cyan)

# --- Set Edge Width ---
edge_width = 1.05
ps_mesh.set_edge_width(edge_width)

highlight_edge = np.array([[0, 1]])

# Register the edge as a curve network
highlight_curve = ps.register_curve_network(
    "highlight_edge",
    nodes=vertices[[0, 2]],
    edges=highlight_edge,
)

# Style the highlighted edge
highlight_curve.set_radius(0.01)  # Thicker than normal mesh edges
highlight_curve.set_enabled(True)

# --- Set Material to 'clay' ---
ps_mesh.set_material("clay")

# --- Disable Ground Plane ---
ps.set_ground_plane_mode("none")

# Show the viewer
ps.show()
