import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from main_model.src.utils.general_utils import read_mesh
from main_model.src.architecture.encoder_architecture import GraphUNet
from main_model.src.utils.hgraph.hgraph import Data, HGraph


class ScalableSoftmax(nn.Module):
    def __init__(self, s=0.43, learn_scaling=True, bias=False):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(s)) if learn_scaling else s
        self.bias = nn.Parameter(torch.tensor(0.0)) if bias else None

    def forward(self, x):
        # Get sequence length (n)
        n = x.size(-1)
        # Apply scaling with log(n)
        scaled_x = x + self.s * torch.log(torch.tensor(n, device=x.device))
        if self.bias is not None:
            scaled_x = scaled_x + self.bias
        # Apply standard softmax
        return F.softmax(scaled_x, dim=-1)


class SSMaxMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, s=0.43, learn_scaling=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.ssmax = ScalableSoftmax(s=s, learn_scaling=learn_scaling)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        # Linear projections and reshape
        q = (
            self.q_proj(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        # Apply SSMax instead of standard softmax
        attn_weights = self.ssmax(scores)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and apply output projection
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class LEHD(nn.Module):
    def __init__(self, city_indices, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = Encoder(city_indices, **model_params)

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
        logits, feasibility_logits = self.decoder(encoder_out)

        return logits, feasibility_logits


class NodeEmbedder(nn.Module):
    """Separate module for embedding different node types"""

    def __init__(self, input_dim, embedding_dim):
        super(NodeEmbedder, self).__init__()
        self.embedd_source = nn.Linear(input_dim, embedding_dim, bias=True)
        self.embedd_depot = nn.Linear(input_dim, embedding_dim, bias=True)
        self.embedding_candidates = nn.Linear(input_dim, embedding_dim, bias=True)

    def forward(self, solutions):
        # Extract different node types
        source = solutions[:, [0], :]
        candidates = solutions[:, 1:-1, :]
        depot = solutions[:, [-1], :]

        # Apply specialized embeddings
        source_emb = self.embedd_source(source)
        depot_emb = self.embedd_depot(depot)
        candidates_emb = self.embedding_candidates(candidates)

        # Return all embeddings and the number of candidates
        return source_emb, candidates_emb, depot_emb


class PointerDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        num_heads,
        num_layers,
        s=0.43,
        learn_scaling=True,
    ):
        super(PointerDecoder, self).__init__()

        # Node embedder to handle different node types
        self.node_embedder = NodeEmbedder(input_dim, embedding_dim)

        self.feasibility_predictor = nn.Linear(embedding_dim, 1)

        # Replace standard MultiheadAttention with SSMax version
        self.self_attention_layers = nn.ModuleList(
            [
                SSMaxMultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_heads,
                    s=s,
                    learn_scaling=learn_scaling,
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

        # Pointer and flag classifier
        self.pointer = nn.Linear(embedding_dim, 2)
        self.flag_classifier = nn.Linear(embedding_dim, 2)

    def forward(self, solutions):
        # Get embeddings for each node type
        source_emb, candidates_emb, depot_emb = self.node_embedder(solutions)

        # Combine for decoder input
        x = torch.cat((source_emb, candidates_emb, depot_emb), dim=1)

        # Process through first transformer layer
        residual = x
        x, _ = self.self_attention_layers[0](x, x, x)
        x = x + residual

        # Extract feasibility predictions after first layer (before bias)
        feasibility_logits = self.feasibility_predictor(
            x[:, 1:-1, :]
        )  # Only for candidate nodes

        # Continue with remaining layers
        for i in range(len(self.self_attention_layers)):
            if i == 0:
                # Already processed first layer
                residual = x
                x = self.ffn_layers[0](x)
                x = x + residual
                continue

            # Self-attention with residual connection
            residual = x
            x, _ = self.self_attention_layers[i](x, x, x)
            x = x + residual

            # Feed-forward with residual connection
            residual = x
            x = self.ffn_layers[i](x)
            x = x + residual

        logits = self.pointer(x)

        candidate_logits = logits[:, 1:-1, 0]
        flag_logits = logits[:, 1:-1, 1]

        logits = torch.cat(
            [
                candidate_logits,
                flag_logits,
            ],
            dim=1,
        )

        return logits, feasibility_logits.squeeze(-1)


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
    def __init__(self, city_indices, **model_params):
        super().__init__()
        self.model_params = model_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_dim = self.model_params["embedding_dim"]

        self.prepare_embeddings(city_indices)

        self.transition_layer = nn.Linear(
            self.model_params["pretrained_encoder"]["out_channels"] + 1,
            embedding_dim,
            bias=True,
        )

    def prepare_embeddings(self, city_indexes):
        gegnn = GeGnn(self.model_params["pretrained_encoder"]).to(self.device)
        gegnn.compute_embeddings()
        self.embeddings = gegnn.embds[city_indexes]

    def forward(self, solutions):
        out = self.embeddings[solutions[:, :, 0].to(torch.int64)]
        return out
