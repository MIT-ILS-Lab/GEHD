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

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "serif"


def show_colormap_legend_horizontal(
    cmap,
    vmin,
    vmax,
    power=1.0,
    label="Distance",
    n_ticks=5,
    filename="legend.svg",
    dpi=300,
    width_in=1.34,
    height_in=0.35,
):
    norm_vals = np.linspace(0, 1, 256)
    adjusted = norm_vals**power
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(bottom=0.7, top=0.95, left=0.08, right=0.98)
    cb = plt.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap),
        cax=ax,
        orientation="horizontal",
        ticks=np.linspace(0, 1, n_ticks),
    )
    tick_locs = np.linspace(0, 1, n_ticks)
    tick_vals = vmin + (tick_locs ** (1 / power)) * (vmax - vmin)
    cb.set_ticks(tick_locs)
    cb.set_ticklabels([f"{v:.2f}" for v in tick_vals])
    cb.set_label(label)
    cb.ax.tick_params(length=3, width=0.8)
    plt.savefig(filename, bbox_inches="tight", dpi=dpi, format="svg")
    plt.close(fig)
    print(f"Legend saved as {filename} ({width_in} in wide)")


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


def compute_geodesic_distances(vertices, faces, source_index=0):
    from pygeodesic import geodesic

    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
    source_indices = np.array([source_index])
    target_indices = None
    distances, _ = geoalg.geodesicDistances(source_indices, target_indices)
    return distances


# Custom colormaps
pink = [0.96, 0, 0.42]
cyan = [0.22, 0.827, 0.835]
blue = [0.216, 0.522, 0.882]

cmap = LinearSegmentedColormap.from_list("blue_pink", [(0.0, blue), (1.0, pink)])
cmap_diff = LinearSegmentedColormap.from_list(
    "cyan_blue_pink", [(0.0, cyan), (0.5, blue), (1.0, pink)]
)


def main(config):
    logdir = config["solver"]["logdir"]
    ckpt_dir = os.path.join(logdir, "checkpoints")
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".tar")]
    ckpt_files.sort()
    ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])

    PATH_TO_MESH = config["data"]["preparation"]["path_to_mesh"]
    mesh_files = [f for f in os.listdir(PATH_TO_MESH) if f.endswith(".obj")]
    test_file = os.path.join(PATH_TO_MESH, mesh_files[0])

    output_dir = config["model"]["output_dir"]
    output_ssad = os.path.join(output_dir, "ssad_ours.npy")
    output_mesh = os.path.join(output_dir, "our_mesh.obj")
    output_colors = os.path.join(output_dir, "our_mesh_colors.npy")

    obj_dic_temp = read_mesh(test_file)
    centroid = obj_dic_temp["vertices"].mean(dim=0)
    distances = torch.norm(obj_dic_temp["vertices"] - centroid, dim=1)
    default_start_idx = int(torch.argmin(distances).item())

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="SSAD")
    parser.add_argument("--test_file", type=str, default=test_file)
    parser.add_argument("--ckpt_path", type=str, default=ckpt_path)
    parser.add_argument(
        "--start_pts",
        type=int,
        default=default_start_idx,
        help="Start point index; default is the vertex closest to the centroid.",
    )
    parser.add_argument("--output", type=str, default=output_ssad)
    args = parser.parse_args()

    if args.mode == "SSAD":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj_dic = read_mesh(args.test_file)
        print(
            "Vertex number: ",
            obj_dic["vertices"].shape[0],
            "Face number: ",
            obj_dic["faces"].shape[0],
        )

        start_pts = torch.tensor(int(args.start_pts)).to(device)

        model = PretrainedGeGnn(args.ckpt_path, config["model"]).to(device)
        model.precompute(obj_dic)

        dist_pred = model.SSAD([start_pts])[0]
        dist_pred = torch.clamp(dist_pred, min=0.0)  # Cap at 0 to avoid negatives
        np.save(args.output, dist_pred.detach().cpu().numpy())

        def save_mesh_as_obj(vertices, faces, scalar=None, path=output_mesh):
            with open(path, "w") as f:
                f.write("# mesh\n")
                for v in vertices:
                    f.write("v " + " ".join(map(str, v)) + "\n")
                for face in faces:
                    f.write("f " + " ".join(str(i + 1) for i in face) + "\n")
                if scalar is not None:
                    scalar = (scalar - np.min(scalar)) / (
                        np.max(scalar) - np.min(scalar)
                    )
                    for c in scalar:
                        f.write("c " + " ".join([str(c)] * 3) + "\n")
            print(
                "Saved mesh as obj file:",
                path,
                "(with color)" if scalar is not None else "(without color)",
            )

        save_mesh_as_obj(
            obj_dic["vertices"].detach().cpu().numpy(),
            obj_dic["faces"].detach().cpu().numpy(),
            dist_pred.detach().cpu().numpy(),
        )

        import polyscope as ps

        mesh = trimesh.load_mesh(output_mesh, process=False)
        vertices = mesh.vertices
        faces = mesh.faces

        source_index = int(args.start_pts)
        geodist = compute_geodesic_distances(vertices, faces, source_index=source_index)

        dist_pred_np = dist_pred.detach().cpu().numpy()
        maxval = max(np.max(dist_pred_np), np.max(geodist))
        norm_pred = dist_pred_np / maxval
        norm_geodist = geodist / maxval

        power = 1.5
        adjusted_pred = norm_pred**power
        adjusted_geodist = norm_geodist**power
        pred_colors = cmap(adjusted_pred)[:, :3]
        geodist_colors = cmap(adjusted_geodist)[:, :3]

        diff = norm_pred - norm_geodist
        max_abs = np.max(np.abs(diff))
        diff_centered = (
            np.full_like(diff, 0.5)
            if max_abs < 1e-8
            else (diff + max_abs) / (2 * max_abs)
        )
        diff_adjusted = diff_centered**power
        diff_colors = cmap_diff(diff_adjusted)[:, :3]

        np.save(output_colors, pred_colors)

        ps.init()
        ps_mesh = ps.register_surface_mesh("mesh", vertices, faces)
        ps_mesh.set_material("clay")
        ps_mesh.set_edge_width(0.5)
        ps.set_ground_plane_mode("none")

        ps_mesh.add_color_quantity("Model Prediction", pred_colors, enabled=True)
        ps_mesh.add_color_quantity("True Geodesic", geodist_colors, enabled=False)
        ps_mesh.add_color_quantity("Difference (Pred-True)", diff_colors, enabled=False)

        source_point = vertices[source_index].reshape(1, 3)
        source_handle = ps.register_point_cloud("Source Node", source_point)
        source_handle.set_color([0.0, 0.0, 0.0])  # black
        source_handle.set_radius(0.02, relative=True)

        ps.show()

        show_colormap_legend_horizontal(
            cmap,
            vmin=0,
            vmax=maxval,
            power=power,
            label="Distance from source node",
            filename="legend_distance.svg",
            width_in=2,
            height_in=0.7,
            n_ticks=4,
        )

        show_colormap_legend_horizontal(
            cmap_diff,
            vmin=-max_abs,
            vmax=+max_abs,
            power=power,
            label="Difference (Predicted - True)",
            filename="legend_difference.svg",
            width_in=2,
            height_in=0.7,
            n_ticks=4,
        )
    else:
        print(f"Invalid mode! ({args.mode})")


if __name__ == "__main__":
    args = parse_args("main_model/configs/config_encoder.yaml")
    config = load_config(args.config)
    main(config)
