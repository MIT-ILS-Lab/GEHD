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
        # vertex_normals=np.tile([0, 0, 1], (len(vertices), 1)),
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
        shapes = {
            "sphere": trimesh.creation.icosphere(subdivisions=subdivisions),
            "sphere_tilde": trimesh.creation.icosphere(subdivisions=subdivisions),
        }
    elif TYPE == "2d":
        # Sample points in 2D space
        num_points = 10000
        sample_points = np.random.rand(num_points, 2)
        mesh = create_flat_mesh(sample_points)
        shapes = {
            "2d_plane": mesh,
            "2d_plane_tilde": mesh,
        }
    else:
        raise ValueError(f"Invalid type: {TYPE}. Must be 'sphere' or '2d'.")

    # Save meshes as .obj files
    for name, mesh in shapes.items():
        mesh.export(os.path.join(PATH_TO_MESH, f"{name}.obj"))
