"""
This file contains the code to generate the meshes for the encoder and decoder.
Call this function before the generate_object_data.py file.
"""

import os
import trimesh
import numpy as np
from scipy.spatial import Delaunay

from main_model.src.utils.config import load_config, parse_args


def create_flat_mesh(points_2d: np.ndarray) -> trimesh.Trimesh:
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
        # Create a regular 2D grid mesh on the [0,1] x [0,1] square
        n = 100  # 100 x 100 = 10,000 vertices
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        grid_x, grid_y = np.meshgrid(x, y)
        vertices_2d = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        vertices_3d = np.hstack([vertices_2d, np.zeros((len(vertices_2d), 1))])

        # Apply 45-degree rotation
        rotation_angle = np.pi / 4
        rotation_matrix = np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                [0, 0, 1],
            ]
        )
        vertices_3d = vertices_3d @ rotation_matrix.T

        # Manually build triangle faces
        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                idx0 = i * n + j
                idx1 = idx0 + 1
                idx2 = idx0 + n
                idx3 = idx2 + 1
                faces.append([idx0, idx2, idx1])
                faces.append([idx1, idx2, idx3])

        mesh = trimesh.Trimesh(vertices=vertices_3d, faces=np.array(faces))

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
        # Generate a unit cube shell mesh with ~10,000 vertices
        n = 41  # 6 * 41^2 ≈ 10,086 vertices total

        lin = np.linspace(0, 1, n)
        u, v = np.meshgrid(lin, lin)
        u = u.ravel()
        v = v.ravel()

        vertices = []
        faces = []

        def add_face(offset, normal_axis, flip=False):
            start_index = len(vertices)

            if normal_axis == 0:  # x = const
                coords = np.column_stack([np.full_like(u, offset), u, v])
            elif normal_axis == 1:  # y = const
                coords = np.column_stack([u, np.full_like(u, offset), v])
            elif normal_axis == 2:  # z = const
                coords = np.column_stack([u, v, np.full_like(u, offset)])

            if flip:
                coords = coords[:, [0, 2, 1]]

            vertices.extend(coords)

            for i in range(n - 1):
                for j in range(n - 1):
                    idx0 = start_index + i * n + j
                    idx1 = idx0 + 1
                    idx2 = idx0 + n
                    idx3 = idx2 + 1
                    faces.append([idx0, idx2, idx1])
                    faces.append([idx1, idx2, idx3])

        # Add all 6 faces
        add_face(0, normal_axis=0, flip=True)
        add_face(1, normal_axis=0)
        add_face(0, normal_axis=1, flip=True)
        add_face(1, normal_axis=1)
        add_face(0, normal_axis=2, flip=True)
        add_face(1, normal_axis=2)

        cube_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

        shapes = {
            "cube": cube_mesh,
            "cube_tilde": cube_mesh.copy(),
        }
    else:
        raise ValueError(f"Invalid type: {TYPE}. Must be 'sphere', '2d', or 'torus'.")

    # Save meshes as .obj files
    for name, mesh in shapes.items():
        path = os.path.join(PATH_TO_MESH, f"{name}.obj")
        mesh.export(path)
        print(f"Saved {name} mesh to {path}")
