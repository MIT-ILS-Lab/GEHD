import os
import trimesh

from gegnn.utils.thsolver import default_settings
from gegnn.utils.thsolver.config import parse_args

if __name__ == "__main__":
    # initialize global settings
    default_settings._init()
    FLAGS = parse_args(config_path="main_model/config.yaml")
    default_settings.set_global_values(FLAGS)

    # Define path to mesh directory
    PATH_TO_MESH = FLAGS.DATA.preparation.path_to_mesh

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
