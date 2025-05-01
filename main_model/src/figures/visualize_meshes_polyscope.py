import numpy as np
import trimesh
import polyscope as ps
from scipy.spatial import Delaunay


def create_flat_mesh(points_2d):
    """Create 3D mesh from 2D points with z=0 and vertical normals"""
    # Convert to 3D coordinates
    vertices = np.hstack([points_2d, np.zeros((len(points_2d), 1))])

    # Generate triangular faces
    tri = Delaunay(points_2d)
    faces = tri.simplices

    # Create mesh with explicit normals
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
    )

    return mesh


# Plane
num_points_per_side = 100  # 100 x 100 = 10,000 points
x = np.linspace(0, 1, num_points_per_side)
y = np.linspace(0, 1, num_points_per_side)
grid_x, grid_y = np.meshgrid(x, y)
sample_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
mesh = create_flat_mesh(sample_points)

# Sphere
subdivisions = 5
mesh = trimesh.creation.icosphere(subdivisions=subdivisions)

# Torus
major_radius = 1.0
minor_radius = 0.3
major_sections = 100  # circles around main ring
minor_sections = 100  # points along each circle

mesh = trimesh.creation.torus(
    major_radius=major_radius,
    minor_radius=minor_radius,
    major_sections=major_sections,
    minor_sections=minor_sections,
)

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

# --- Set Material to 'clay' ---
ps_mesh.set_material("clay")

# --- Disable Ground Plane ---
ps.set_ground_plane_mode("none")

# Show the viewer
ps.show()
