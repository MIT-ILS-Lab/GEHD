import os
import trimesh

from main_model.src.utils.config import load_config, parse_args

if __name__ == "__main__":
    # initialize global settings
    args = parse_args()
    config = load_config(args.config)

    # Define path to mesh directory
    PATH_TO_MESH = config["DATA"]["preparation"]["path_to_mesh"]

    # Ensure the directory exists
    os.makedirs(PATH_TO_MESH, exist_ok=True)

    # Generate simple meshes and save them as .obj files
    shapes = {
        # "cube": trimesh.creation.box(),
        "sphere": trimesh.creation.icosphere(subdivisions=5),
        "sphere_tilde": trimesh.creation.icosphere(subdivisions=5),
        # "cylinder": trimesh.creation.cylinder(radius=0.5, height=1.0),
    }

    # Save meshes as .obj files
    for name, mesh in shapes.items():
        mesh.export(os.path.join(PATH_TO_MESH, f"{name}.obj"))
