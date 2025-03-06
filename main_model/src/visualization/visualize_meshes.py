import os
import trimesh

from main_model.src.utils.config import load_config, parse_args


if __name__ == "__main__":
    # Load the config file
    args = parse_args()
    config = load_config(args.config)

    # Define path to mesh directory
    PATH_TO_MESH = config["data"]["preparation"]["path_to_mesh"]

    # List mesh files
    mesh_files = [f for f in os.listdir(PATH_TO_MESH) if f.endswith(".obj")]

    if not mesh_files:
        raise FileNotFoundError("No .obj files found in the specified mesh directory.")

    # Display each mesh in the directory
    for mesh_file in mesh_files:
        # Load the mesh
        mesh = trimesh.load_mesh(os.path.join(PATH_TO_MESH, mesh_file))

        # Create a scene with the mesh
        scene = trimesh.Scene([mesh])

        # Render the scene
        scene.show(viewer="gl", flags={"wireframe": True})
