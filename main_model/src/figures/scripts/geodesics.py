import numpy as np
import trimesh
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "serif"

# Custom colormaps
pink = [0.96, 0, 0.42]
blue = [0.216, 0.522, 0.882]
cmap = LinearSegmentedColormap.from_list("blue_pink", [(0.0, blue), (1.0, pink)])


def show_colormap_legend_vertical(
    cmap,
    vmin,
    vmax,
    power=1.0,
    label="Distance",
    n_ticks=5,
    filename="legend.pdf",  # Default to PDF
    dpi=300,
    width_in=0.5,
    height_in=2,
):
    norm_vals = np.linspace(0, 1, 256)
    adjusted = norm_vals**power
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.5, right=0.95, top=0.98, bottom=0.08)
    cb = plt.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap),
        cax=ax,
        orientation="vertical",
        ticks=np.linspace(0, 1, n_ticks),
    )
    tick_locs = np.linspace(0, 1, n_ticks)
    tick_vals = vmin + (tick_locs ** (1 / power)) * (vmax - vmin)
    cb.set_ticks(tick_locs)
    cb.set_ticklabels([f"{v:.2f}" for v in tick_vals])
    cb.set_label(label)
    cb.ax.tick_params(length=3, width=0.8)
    plt.savefig(filename, bbox_inches="tight", dpi=dpi, format="pdf", transparent=True)
    plt.close(fig)
    print(f"Legend saved as {filename} ({height_in} in tall, PDF format)")


def compute_geodesic_distances(vertices, faces, source_index=0):
    from pygeodesic import geodesic

    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
    source_indices = np.array([source_index])
    target_indices = None
    distances, _ = geoalg.geodesicDistances(source_indices, target_indices)
    return distances


def save_mesh_as_obj(vertices, faces, scalar=None, path="geod_icosphere.obj"):
    with open(path, "w") as f:
        f.write("# mesh\n")
        for v in vertices:
            f.write("v " + " ".join(map(str, v)) + "\n")
        for face in faces:
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")
        if scalar is not None:
            scalar = (scalar - np.min(scalar)) / (np.max(scalar) - np.min(scalar))
            for c in scalar:
                f.write("c " + " ".join([str(c)] * 3) + "\n")
    print(
        "Saved mesh as obj file:",
        path,
        "(with color)" if scalar is not None else "(without color)",
    )


def main():
    # Generate mesh
    subdivisions = 2
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    vertices = mesh.vertices
    faces = mesh.faces

    # Find centroid-proximate vertex
    centroid = np.mean(vertices, axis=0)
    distances = np.linalg.norm(vertices - centroid, axis=1)
    source_index = int(np.argmin(distances))

    # Compute geodesic distances
    geodist = compute_geodesic_distances(vertices, faces, source_index=source_index)
    maxval = np.max(geodist)
    norm_geodist = geodist / maxval

    # Color mapping (with power curve)
    power = 1.5
    adjusted_geodist = norm_geodist**power
    geodist_colors = cmap(adjusted_geodist)[:, :3]

    # Visualization
    import polyscope as ps

    # ps.init()
    # ps_mesh = ps.register_surface_mesh("Icosphere", vertices, faces)
    # ps_mesh.set_material("clay")
    # ps_mesh.set_edge_width(0.5)
    # ps.set_ground_plane_mode("none")
    # ps_mesh.add_color_quantity("True Geodesic", geodist_colors, enabled=True)

    # source_point = vertices[source_index].reshape(1, 3)
    # source_handle = ps.register_point_cloud("Source Node", source_point)
    # source_handle.set_color([0.0, 0.0, 0.0])  # black
    # source_handle.set_radius(0.02, relative=True)
    # ps.show()

    # Save colorbar
    show_colormap_legend_vertical(
        cmap,
        vmin=0,
        vmax=maxval,
        power=power,
        dpi=1200,
        label="Distance from source node",
        filename="legend_distance.pdf",
        width_in=0.5,
        height_in=3.5,
        n_ticks=4,
    )


if __name__ == "__main__":
    main()
