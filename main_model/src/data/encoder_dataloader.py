import torch
import numpy as np

from gegnn.utils import ocnn

from gegnn.hgraph.hgraph import Data
from gegnn.hgraph.hgraph import HGraph

from utils.thsolver import Dataset


class Transform:

    def __init__(self, distort=False, angle=[10, 10, 10], scale=0.1, jitter=0.1):
        """
        Initialization of the Transform class. Parameters for random distortion are set here.

        Args:
          distort: Whether to apply data distortion (rotation, scaling, jitter).
          angle: List of angles for random rotation on each axis [X, Y, Z].
          scale: Scale factor for random scaling.
          jitter: Jitter factor for random translation.
        """
        # Set distortion parameters
        self.distort = distort
        self.angle = angle  # Rotation angles for each axis
        self.scale = scale  # Scaling factor for vertices
        self.jitter = jitter  # Jitter factor for translation

    def __call__(self, sample: dict, idx: int):
        """
        Applies transformations to each sample including random distortions.

        Args:
          sample: A dict containing vertices, normals, edges, dist_idx, dist_val.
          idx: Index of the sample.

        Returns:
          A dict with keys: hgraph, vertices, normals, dist, edges after applying transformations.
        """
        # Extract data from the sample dictionary
        vertices = torch.from_numpy(sample["vertices"].astype(np.float32))
        normals = torch.from_numpy(sample["normals"].astype(np.float32))
        edges = (
            torch.from_numpy(sample["edges"].astype(np.float32)).t().contiguous().long()
        )
        dist_idx = sample["dist_idx"].astype(np.float32)
        dist_val = sample["dist_val"].astype(np.float32)

        # Concatenate distances and create a tensor for them
        dist = np.concatenate([dist_idx, dist_val], -1)
        dist = torch.from_numpy(dist)

        # Randomly sample distance pairs
        size = min(len(dist), 100_000)
        rnd_idx = torch.randint(low=0, high=dist.shape[0], size=(size,))
        dist = dist[rnd_idx]

        # Normalize normals to unit vectors
        norm2 = torch.sqrt(torch.sum(normals**2, dim=1, keepdim=True))
        normals = normals / torch.clamp(norm2, min=1.0e-12)

        # Apply distortions if enabled
        if self.distort:
            vertices, normals = self.apply_distortion(vertices, normals)

        # Construct the hierarchical graph
        h_graph = HGraph()
        h_graph.build_single_hgraph(
            Data(x=torch.cat([vertices, normals], dim=1), edge_index=edges)
        )

        # Return the transformed data
        return {
            "hgraph": h_graph,
            "vertices": vertices,
            "normals": normals,
            "dist": dist,
            "edges": edges,
        }

    def apply_distortion(self, vertices: torch.Tensor, normals: torch.Tensor):
        """Applies random distortion to the vertices and normals including rotation, scaling, and jittering."""

        # Apply random rotation
        rotation_matrix = self.random_rotation_matrix(self.angle)
        vertices = torch.matmul(vertices, rotation_matrix)
        normals = torch.matmul(normals, rotation_matrix)

        # Apply random jitter (translation)
        jitter = (
            torch.rand(vertices.shape) * 2 * self.jitter - self.jitter
        )  # Center the jitter around 0
        vertices = vertices + jitter

        # Apply random scaling
        scale_factor = (
            torch.rand(1) * (2 * self.scale) - self.scale + 1.0
        )  # Shift to value between -scale and scale
        vertices = vertices * scale_factor

        return vertices, normals

    def random_rotation_matrix(self, angles: list):
        """Generates a random 3D rotation matrix."""
        rotation_matrices = []
        for i in range(3):
            angle = torch.randint(low=-angles[i], high=angles[i], size=(1,)).item()
            angle = angle * np.pi / 180.0  # Convert to radians
            cos_val = torch.cos(torch.tensor(angle))
            sin_val = torch.sin(torch.tensor(angle))

            # Construct the rotation matrix for each axis
            if i == 0:  # X-axis
                rotation_matrix = torch.tensor(
                    [[1, 0, 0], [0, cos_val, -sin_val], [0, sin_val, cos_val]],
                    dtype=torch.float32,
                )
            elif i == 1:  # Y-axis
                rotation_matrix = torch.tensor(
                    [[cos_val, 0, sin_val], [0, 1, 0], [-sin_val, 0, cos_val]],
                    dtype=torch.float32,
                )
            else:  # Z-axis
                rotation_matrix = torch.tensor(
                    [[cos_val, -sin_val, 0], [sin_val, cos_val, 0], [0, 0, 1]],
                    dtype=torch.float32,
                )
            rotation_matrices.append(rotation_matrix)

        # Multiply the rotation matrices for each axis to get the final rotation matrix
        final_rotation_matrix = rotation_matrices[0]
        for mat in rotation_matrices[1:]:
            final_rotation_matrix = torch.matmul(final_rotation_matrix, mat)

        return final_rotation_matrix


def collate_batch(batch: list):
    """
    This function is used to collate a batch of samples.

    Args:
      batch: list of single samples. Each sample is a dict with keys: edges, vertices, normals, dist

    Returns:
      outputs: a big sample as a dict with keys: edges, vertices, normals, dist, feature, hgraph
    """
    assert type(batch) == list

    outputs = {}
    for key in batch[0].keys():
        outputs[key] = [b[key] for b in batch]

    pts_num = torch.tensor([pts.shape[0] for pts in outputs["vertices"]])
    cum_sum = ocnn.utils.cumsum(pts_num, dim=0, exclusive=True)
    for i, dist in enumerate(outputs["dist"]):
        dist[:, :2] += cum_sum[i]

    outputs["dist"] = torch.cat(outputs["dist"], dim=0)

    # input feature
    vertices = torch.cat(outputs["vertices"], dim=0)
    normals = torch.cat(outputs["normals"], dim=0)
    feature = torch.cat([vertices, normals], dim=1)
    outputs["feature"] = feature

    # merge a batch of hgraphs into one super hgraph
    hgraph_super = HGraph(batch_size=len(batch))
    hgraph_super.merge_hgraph(outputs["hgraph"])
    outputs["hgraph"] = hgraph_super

    return outputs


def get_dataset(flags):
    transform = Transform(flags.distort, flags.angle, flags.scale, flags.jitter)
    dataset = Dataset(
        flags.location, flags.filelist, transform, read_file=np.load, take=flags.take
    )
    return dataset, collate_batch
