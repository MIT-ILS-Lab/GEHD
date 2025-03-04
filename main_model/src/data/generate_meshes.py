import os
import trimesh

from gegnn.utils.thsolver import default_settings
from gegnn.utils.thsolver.config import parse_args

import inspect

# Get the file path of the function definition
file_path = inspect.getfile(parse_args)

# Print the path
print(f"The function 'parse_args' is defined in: {file_path}")

# initialize global settings
default_settings._init()
FLAGS = parse_args(config_path="main_model/config.yaml")

if __name__ == "__main__":

    default_settings.set_global_values(FLAGS)

    # Define path to mesh directory
    PATH_TO_MESH = FLAGS.DATA.preparation.path_to_mesh

    # Ensure the directory exists
    os.makedirs(PATH_TO_MESH, exist_ok=True)

    # Generate simple meshes and save them as .obj files
    shapes = {
        # "cube": trimesh.creation.box(),
        "sphere_train": trimesh.creation.icosphere(subdivisions=5),
        "sphere_val": trimesh.creation.icosphere(subdivisions=5),
        # "cylinder": trimesh.creation.cylinder(radius=0.5, height=1.0),
    }

    # Save meshes as .obj files
    for name, mesh in shapes.items():
        mesh.export(os.path.join(PATH_TO_MESH, f"{name}.obj"))
