import numpy as np
import trimesh
import polyscope as ps

# subdivisions = 2
# mesh = trimesh.creation.icosphere(subdivisions=subdivisions)

# Vertices: center, top, bottom, and two fans
vertices = np.array(
    [
        (0, 0, 0),
        (1.0000, 0.0000, 1),
        (0.3090, 0.9511, 1),
        (-0.8090, 0.5878, 1),
        (-0.8090, -0.5878, 1),
        (0.3090, -0.9511, 1),
        (1.5000, 0.0000, -1),
        (0.4635, 1.4266, -1),
        (-1.2135, 0.8817, -1),
        (-1.2135, -0.8817, -1),
        (0.4635, -1.4266, -1),
    ]
)

# Triangles (fans above and below center vertex)
faces = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 1],
        [0, 6, 7],
        [0, 7, 8],
        [0, 8, 9],
        [0, 9, 10],
        [0, 10, 6],
    ]
)

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

pink = [0.96, 0, 0.42]
blue = [0.216, 0.522, 0.882]
cyan = [0.22, 0.827, 0.835]

# Initialize Polyscope
ps.init()

# Register the mesh with Polyscope
ps_mesh = ps.register_surface_mesh("trimesh_mesh", mesh.vertices, mesh.faces)

# --- Color Faces ---
ps_mesh.set_color(blue)

# --- Set Edge Width ---
edge_width = 1.05
ps_mesh.set_edge_width(edge_width)

vertex_0 = vertices[0].reshape(1, 3)
ps_point = ps.register_point_cloud("vertex_0", vertex_0)
ps_point.set_color(pink)
ps_point.set_radius(0.05)  # Adjust size as needed

# --- Set Material to 'clay' ---
ps_mesh.set_material("clay")

# --- Disable Ground Plane ---
ps.set_ground_plane_mode("none")

# Show the viewer
ps.show()
