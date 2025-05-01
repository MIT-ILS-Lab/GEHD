import numpy as np
import torch
import torch.nn as nn
import trimesh
import argparse
import os

from main_model.src.utils.config import load_config, parse_args
from main_model.src.utils.general_utils import read_mesh
from main_model.src.architecture.encoder_architecture import GraphUNet
from main_model.src.utils.hgraph.hgraph import Data, HGraph


# a wrapper for the pretrained model
class PretrainedGeGnn(nn.Module):
    def __init__(self, ckpt_path, config):
        super(PretrainedGeGnn, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GraphUNet(
            config["in_channels"], config["hidden_channels"], config["out_channels"]
        ).to(self.device)
        self.embds = None
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_dict"])

    def embd_decoder_func(self, i, j, embedding):
        i = i.long()
        j = j.long()
        embd_i = embedding[i].squeeze(-1)
        embd_j = embedding[j].squeeze(-1)
        embd = (embd_i - embd_j) ** 2
        pred = self.model.embedding_decoder_mlp(embd)
        return pred.squeeze(-1)

    def precompute(self, mesh):
        with torch.no_grad():
            vertices = mesh["vertices"]
            normals = mesh["normals"]
            edges = mesh["edges"]
            tree = HGraph()
            tree.build_single_hgraph(
                Data(x=torch.cat([vertices, normals], dim=1), edge_index=edges)
            )
            self.embds = self.model(
                torch.cat([vertices, normals], dim=1),
                tree,
                tree.depth,
                dist=None,
                only_embd=True,
            ).detach()

    def forward(self, p_vertices=None, q_vertices=None):
        assert self.embds is not None, "Please call precompute() first!"
        with torch.no_grad():
            return self.embd_decoder_func(p_vertices, q_vertices, self.embds)

    def SSAD(self, source: list):
        assert self.embds is not None, "Please call precompute() first!"
        ret = []
        with torch.no_grad():
            for src in source:
                s = torch.tensor([src]).repeat(self.embds.shape[0]).to(self.device)
                t = torch.arange(self.embds.shape[0]).to(self.device)
                ret.append(self.embd_decoder_func(s, t, self.embds))
        return ret


# a wrapper of pretrained model, so that it can be called directly from the command line
def main(config):
    # Load the latest checkpoint
    logdir = config["solver"]["logdir"]
    ckpt_dir = os.path.join(logdir, "checkpoints")  # Checkpoint directory for past runs

    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".tar")]
    ckpt_files.sort()
    ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])

    # Create a path to the first mesh file
    PATH_TO_MESH = config["data"]["preparation"]["path_to_mesh"]
    mesh_files = [f for f in os.listdir(PATH_TO_MESH) if f.endswith(".obj")]
    test_file = os.path.join(PATH_TO_MESH, mesh_files[0])

    # Output directory
    output_dir = config["model"]["output_dir"]
    output_ssad = os.path.join(output_dir, "ssad_ours.npy")
    output_mesh = os.path.join(output_dir, "our_mesh.obj")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="SSAD", help="only SSAD available for now"
    )
    parser.add_argument(
        "--test_file", type=str, default=test_file, help="path to the obj file"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=ckpt_path, help="path to the checkpoint"
    )
    parser.add_argument("--start_pts", type=int, default=0, help="an int is expected.")
    parser.add_argument(
        "--output", type=str, default=output_ssad, help="path to the output file"
    )
    args = parser.parse_args()

    if args.mode == "SSAD":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj_dic = read_mesh(args.test_file)
        # print the vertex and face number
        print(
            "Vertex number: ",
            obj_dic["vertices"].shape[0],
            "Face number: ",
            obj_dic["faces"].shape[0],
            1,
        )
        start_pts = torch.tensor(int(args.start_pts)).to(device)

        model = PretrainedGeGnn(args.ckpt_path, config["model"]).to(device)
        model.precompute(obj_dic)
        dist_pred = model.SSAD([start_pts])[0]
        np.save(args.output, dist_pred.detach().cpu().numpy())

        # save the colored mesh for visualization
        # given the vertices, faces of a mesh, save it as obj file
        def save_mesh_as_obj(vertices, faces, scalar=None, path=output_mesh):
            with open(path, "w") as f:
                f.write("# mesh\n")  # header of LittleRender
                for v in vertices:
                    f.write("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")
                for face in faces:
                    f.write(
                        "f "
                        + str(face[0] + 1)
                        + " "
                        + str(face[1] + 1)
                        + " "
                        + str(face[2] + 1)
                        + "\n"
                    )
                if scalar is not None:
                    # normalize the scalar to [0, 1]
                    scalar = (scalar - np.min(scalar)) / (
                        np.max(scalar) - np.min(scalar)
                    )
                    for c in scalar:
                        f.write("c " + str(c) + " " + str(c) + " " + str(c) + "\n")

            print("Saved mesh as obj file:", path, end="")
            if scalar is not None:
                print(" (with color) ")
            else:
                print(" (without color)")

        save_mesh_as_obj(
            obj_dic["vertices"].detach().to(device).numpy(),
            obj_dic["faces"].detach().to(device).numpy(),
            dist_pred.detach().to(device).numpy(),
        )

    else:
        print("Invalid mode! (" + args.mode + ")")


if __name__ == "__main__":
    # Load the config file
    args = parse_args("main_model/configs/config_encoder.yaml")
    config = load_config(args.config)

    # Run main function
    main(config)

    ###################################
    # visualization via polyscope starts
    # comment out the following lines if you are using ssh
    ###################################
    import polyscope as ps
    import numpy as np
    import trimesh

    # Output directory
    output_dir = config["model"]["output_dir"]
    output_ssad = os.path.join(output_dir, "ssad_ours.npy")
    output_mesh = os.path.join(output_dir, "our_mesh.obj")

    # Load mesh
    mesh = trimesh.load_mesh(output_mesh, process=False)
    vertices = mesh.vertices
    faces = mesh.faces

    # Load color numpy array
    colors = np.load(output_ssad)
    print(colors.shape)

    # Initialize polyscope
    ps.init()
    ps_cloud = ps.register_point_cloud("mesh", vertices)
    ps_cloud.add_scalar_quantity("geo_distance", colors, enabled=True)
    ps.show()
    ###################################
    # visualization via polyscope ends
    ###################################
