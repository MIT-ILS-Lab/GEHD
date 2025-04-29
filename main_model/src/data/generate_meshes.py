import os
import trimesh
import numpy as np
from scipy.spatial import Delaunay

from main_model.src.utils.config import load_config, parse_args


def create_flat_mesh(points_2d):
    """Create 3D mesh from 2D points with z=0 and vertical normals"""
    # Convert to 3D coordinates
    vertices = np.hstack([points_2d, np.zeros((len(points_2d), 1))])

    # rotate the vertices
    rotation_angle = np.pi / 4  # 45 degrees
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle), np.cos(rotation_angle), 0],
            [0, 0, 1],
        ]
    )
    vertices = np.dot(vertices, rotation_matrix)

    # Generate triangular faces
    tri = Delaunay(points_2d)
    faces = tri.simplices

    # Create mesh with explicit normals
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
    )

    return mesh


if __name__ == "__main__":
    # initialize global settings
    args = parse_args("main_model/configs/config_encoder.yaml")
    config = load_config(args.config)

    # Define path to mesh directory
    PATH_TO_MESH = config["data"]["preparation"]["path_to_mesh"]
    TYPE = config["data"]["preparation"]["type"]

    # Ensure the directory exists
    os.makedirs(PATH_TO_MESH, exist_ok=True)

    if TYPE == "sphere":
        # Generate simple meshes and save them as .obj files
        subdivisions = 5
        spheres = trimesh.creation.icosphere(subdivisions=subdivisions)
        shapes = {
            "sphere": trimesh.creation.icosphere(subdivisions=subdivisions),
            "sphere_tilde": spheres.copy(),
        }
    elif TYPE == "2d":
        # Sample points in 2D space
        num_points_per_side = 100  # 100 x 100 = 10,000 points
        x = np.linspace(0, 1, num_points_per_side)
        y = np.linspace(0, 1, num_points_per_side)
        grid_x, grid_y = np.meshgrid(x, y)
        sample_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        mesh = create_flat_mesh(sample_points)
        shapes = {
            "2d_plane": mesh,
            "2d_plane_tilde": mesh.copy(),
        }
    elif TYPE == "torus":
        # Generate a torus with around 10k vertices
        # Control sections to get approximately 10k faces
        major_radius = 1.0
        minor_radius = 0.3
        major_sections = 100  # circles around main ring
        minor_sections = 100  # points along each circle

        torus = trimesh.creation.torus(
            major_radius=major_radius,
            minor_radius=minor_radius,
            major_sections=major_sections,
            minor_sections=minor_sections,
        )
        shapes = {
            "torus": torus,
            "torus_tilde": torus.copy(),
        }
    elif TYPE == "cube":
        raise NotImplementedError("Cube mesh generation is not implemented yet.")
    else:
        raise ValueError(f"Invalid type: {TYPE}. Must be 'sphere', '2d', or 'torus'.")

    # Save meshes as .obj files
    for name, mesh in shapes.items():
        path = os.path.join(PATH_TO_MESH, f"{name}.obj")
        mesh.export(path)
        print(f"Saved {name} mesh to {path}")
