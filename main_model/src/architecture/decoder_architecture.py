import torch
import torch.nn as nn
import torch.nn.functional as F

from main_model.src.utils.general_utils import read_mesh
from main_model.src.architecture.encoder_architecture import GraphUNet
from main_model.src.utils.hgraph.hgraph import Data, HGraph


class LEHD(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = Encoder(**model_params)

        self.decoder = PointerDecoder(
            input_dim=model_params["pretrained_encoder"]["out_channels"] + 1,
            embedding_dim=model_params["embedding_dim"],
            hidden_dim=model_params["hidden_dim"],
            num_heads=model_params["head_num"],
            num_layers=model_params["decoder_layer_num"],
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoded_nodes = None

    def forward(
        self,
        solutions,
        capacities=None,
    ):
        remaining_capacity = solutions[:, 0, 3]
        encoder_out = self.encoder(solutions)

        normalized_demand = 2 * (
            (solutions[:, :, 2] - remaining_capacity[:, None]) / capacities[:, None]
        )

        encoder_out = torch.cat((encoder_out, normalized_demand.unsqueeze(-1)), dim=2)

        # Set source node capacity to 0
        encoder_out[:, 0, -1] = 0.0

        # Forward through decoder
        logits_node, logits_flag = self.decoder(encoder_out)

        return logits_node, logits_flag


class NodeEmbedder(nn.Module):
    """Separate module for embedding different node types"""

    def __init__(self, input_dim, embedding_dim):
        super(NodeEmbedder, self).__init__()
        self.embedd_source = nn.Linear(input_dim, embedding_dim, bias=True)
        self.embedding_destination = nn.Linear(input_dim, embedding_dim, bias=True)
        self.embedd_depot = nn.Linear(input_dim, embedding_dim, bias=True)
        self.embedding_candidates = nn.Linear(input_dim, embedding_dim, bias=True)

    def forward(self, solutions):
        # Extract different node types
        source = solutions[:, [0], :]
        candidates = solutions[:, 1:-2, :]
        destination = solutions[:, [-2], :]
        depot = solutions[:, [-1], :]

        # Apply specialized embeddings
        source_emb = self.embedd_source(source)
        destination_emb = self.embedding_destination(destination)
        depot_emb = self.embedd_depot(depot)
        candidates_emb = self.embedding_candidates(candidates)

        # Return all embeddings and the number of candidates
        return source_emb, candidates_emb, destination_emb, depot_emb


class PointerDecoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_heads, num_layers):
        super(PointerDecoder, self).__init__()

        # Node embedder to handle different node types
        self.node_embedder = NodeEmbedder(input_dim, embedding_dim)

        # Self-attention layers
        self.self_attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        # Feed-forward layers
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim),
                )
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(embedding_dim) for _ in range(num_layers * 2)]
        )

        # Pointer and flag classifier
        self.pointer = nn.Linear(embedding_dim, 1)
        self.flag_classifier = nn.Linear(embedding_dim, 2)

    def forward(self, solutions):
        # Get embeddings for each node type
        source_emb, candidates_emb, destination_emb, depot_emb = self.node_embedder(
            solutions
        )

        # Combine for decoder input
        x = torch.cat((source_emb, candidates_emb, destination_emb, depot_emb), dim=1)

        # Process through transformer layers
        for i in range(len(self.self_attention_layers)):
            # Self-attention with residual connection
            residual = x
            x = self.layer_norms[i * 2](x)
            x, _ = self.self_attention_layers[i](x, x, x)
            x = x + residual

            # Feed-forward with residual connection
            residual = x
            x = self.layer_norms[i * 2 + 1](x)
            x = self.ffn_layers[i](x)
            x = x + residual

        # Pointer mechanism - only generate logits for candidate nodes
        candidate_logits = self.pointer(
            x[:, 1 : 1 + candidates_emb.size(1), :]
        ).squeeze(-1)

        # Flag classification using source node representation
        flag_logits = self.flag_classifier(x[:, 0, :])

        return candidate_logits, flag_logits


class GeGnn(nn.Module):
    def __init__(self, config):
        super(GeGnn, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GraphUNet(
            config["in_channels"], config["hidden_channels"], config["out_channels"]
        ).to(self.device)
        self.embds = None
        ckpt = torch.load(config["ckp_path"], map_location=self.device)
        self.model.load_state_dict(ckpt["model_dict"])
        self.mesh_path = config["mesh_path"]

        for param in self.model.parameters():
            param.requires_grad = False

    def compute_embeddings(self):
        self.model.eval()

        mesh = read_mesh(self.mesh_path)
        # No need for with torch.no_grad() if parameters are frozen
        vertices = mesh["vertices"].to(self.device)
        normals = mesh["normals"].to(self.device)
        edges = mesh["edges"].to(self.device)
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
        )
        # Explicitly detach embeddings from computation graph
        self.embds = self.embds.detach()
        self.embds = F.normalize(self.embds)  # normalize embeddings

    def forward(self, idxs):
        assert self.embds is not None, "Please call compute_embeddings() first!"
        assert idxs.dtype == torch.int64, "Please make sure idxs has type int64"
        # No need for with torch.no_grad() if parameters are frozen
        return self.embds[idxs]


class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_dim = self.model_params["embedding_dim"]

        self.gegnn = GeGnn(self.model_params["pretrained_encoder"]).to(self.device)
        self.gegnn.compute_embeddings()

        self.transition_layer = nn.Linear(
            self.model_params["pretrained_encoder"]["out_channels"] + 1,
            embedding_dim,
            bias=True,
        )

    def prepare_embedding(self, city_indexes):
        self.gegnn.embds = self.gegnn.embds[city_indexes]

    def forward(self, solutions):
        out = self.gegnn(solutions[:, :, 0].to(torch.int64))
        return out
