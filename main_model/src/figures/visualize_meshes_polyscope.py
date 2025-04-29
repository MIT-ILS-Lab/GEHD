import numpy as np
import trimesh
import polyscope as ps

subdivisions = 2
mesh = trimesh.creation.icosphere(subdivisions=subdivisions)

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

# --- Set Material to 'clay' ---
ps_mesh.set_material("clay")

# --- Disable Ground Plane ---
ps.set_ground_plane_mode("none")

# Show the viewer
ps.show()
