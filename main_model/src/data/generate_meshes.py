import os
import trimesh
import sys

print("System path: %s" % sys.path)

from gegnn.utils.thsolver import default_settings

# initialize global settings
default_settings._init()
from gegnn.utils.thsolver.config import parse_args

if __name__ == "__main__":
    FLAGS = parse_args(config_path="main_model/config.yaml")
    default_settings.set_global_values(FLAGS)

    # Define path to mesh directory
    PATH_TO_MESH = "main_model/disk/meshes"

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
