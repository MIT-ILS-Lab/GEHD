import torch
import torch.nn as nn
import torch.nn.functional as F

from main_model.src.utils.general_utils import read_mesh
from main_model.src.architecture.encoder_architecture import GraphUNet
from main_model.src.utils.hgraph.hgraph import Data, HGraph


class SmallWorldLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rewiring_prob=0.1):
        super(SmallWorldLayer, self).__init__()
        self.regular_conn = nn.Linear(input_dim, output_dim)
        self.rewiring_prob = rewiring_prob
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create sparse weights for long-range connections
        # We'll create these dynamically in the forward pass

    def forward(self, x):
        # Apply regular connections
        regular_out = self.regular_conn(x)

        # For small-world connections, we'll add random long-range connections
        # that change with each forward pass (this is a form of stochastic rewiring)
        batch_size = x.size(0)
        seq_len = x.size(1) if len(x.shape) > 2 else 1

        # Create random connections for this forward pass
        # This is more efficient than using masks
        if len(x.shape) > 2:
            # For 3D tensors (batch_size, seq_len, features)
            x_flat = x.reshape(-1, x.size(-1))

            # Create random weights for this pass
            # We'll use a smaller number of random connections
            num_connections = int(self.input_dim * self.output_dim * self.rewiring_prob)

            # Create sparse random weights
            random_weights = torch.zeros(
                self.output_dim, self.input_dim, device=x.device
            )
            if num_connections > 0:
                # Get random indices
                row_indices = torch.randint(
                    0, self.output_dim, (num_connections,), device=x.device
                )
                col_indices = torch.randint(
                    0, self.input_dim, (num_connections,), device=x.device
                )
                values = torch.randn(num_connections, device=x.device) * 0.01

                # Populate the random weights
                for i in range(num_connections):
                    random_weights[row_indices[i], col_indices[i]] = values[i]

            # Apply random connections
            random_out = torch.matmul(x_flat, random_weights.t())
            random_out = random_out.view(batch_size, seq_len, self.output_dim)
        else:
            # For 2D tensors (batch_size, features)
            # Similar approach but without reshaping
            num_connections = int(self.input_dim * self.output_dim * self.rewiring_prob)

            random_weights = torch.zeros(
                self.output_dim, self.input_dim, device=x.device
            )
            if num_connections > 0:
                row_indices = torch.randint(
                    0, self.output_dim, (num_connections,), device=x.device
                )
                col_indices = torch.randint(
                    0, self.input_dim, (num_connections,), device=x.device
                )
                values = torch.randn(num_connections, device=x.device) * 0.01

                for i in range(num_connections):
                    random_weights[row_indices[i], col_indices[i]] = values[i]

            random_out = torch.matmul(x, random_weights.t())

        # Combine regular and random connections
        return regular_out + random_out * 0.1  # Scale random connections


class LEHD(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = SmallWorldEncoder(**model_params)

        # Use small-world decoder instead of regular pointer decoder
        self.decoder = SmallWorldPointerDecoder(
            input_dim=model_params["pretrained_encoder"]["out_channels"] + 1,
            embedding_dim=model_params["embedding_dim"],
            hidden_dim=model_params["hidden_dim"],
            num_heads=model_params["head_num"],
            num_layers=model_params["decoder_layer_num"],
            rewiring_prob=model_params.get(
                "rewiring_prob", 0.1
            ),  # Default rewiring probability
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
        logits = self.decoder(encoder_out)

        return logits


class SmallWorldNodeEmbedder(nn.Module):
    """Small-world node embedder with local clustering and long-range connections"""

    def __init__(self, input_dim, embedding_dim, rewiring_prob=0.1):
        super(SmallWorldNodeEmbedder, self).__init__()
        # Regular embeddings for local structure
        self.embedd_source = nn.Linear(embedding_dim + 1, embedding_dim, bias=True)
        self.embedding_destination = nn.Linear(
            embedding_dim + 1, embedding_dim, bias=True
        )
        self.embedd_depot = nn.Linear(embedding_dim + 1, embedding_dim, bias=True)
        self.embedding_candidates = nn.Linear(
            embedding_dim + 1, embedding_dim, bias=True
        )

        # Small-world cross-connections for global integration
        self.sw_integration = SmallWorldLayer(
            embedding_dim + 1, embedding_dim, rewiring_prob
        )
        self.integration_weight = nn.Parameter(torch.tensor(0.3))  # Learnable weight

    def forward(self, solutions):
        # Extract different node types
        source = solutions[:, [0], :]
        candidates = solutions[:, 1:-2, :]
        destination = solutions[:, [-2], :]
        depot = solutions[:, [-1], :]

        # Apply specialized embeddings (local structure)
        source_emb = self.embedd_source(source)
        destination_emb = self.embedding_destination(destination)
        depot_emb = self.embedd_depot(depot)
        candidates_emb = self.embedding_candidates(candidates)

        # Apply small-world integration (global structure)
        sw_source = self.sw_integration(source)
        sw_destination = self.sw_integration(destination)
        sw_depot = self.sw_integration(depot)
        sw_candidates = self.sw_integration(candidates)

        # Combine local and global representations with learnable weight
        alpha = torch.sigmoid(self.integration_weight)
        source_emb = (1 - alpha) * source_emb + alpha * sw_source
        destination_emb = (1 - alpha) * destination_emb + alpha * sw_destination
        depot_emb = (1 - alpha) * depot_emb + alpha * sw_depot
        candidates_emb = (1 - alpha) * candidates_emb + alpha * sw_candidates

        # Return all embeddings
        return source_emb, candidates_emb, destination_emb, depot_emb


class SmallWorldPointerDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        num_heads,
        num_layers,
        rewiring_prob=0.1,
    ):
        super(SmallWorldPointerDecoder, self).__init__()

        # Small-world node embedder
        self.node_embedder = SmallWorldNodeEmbedder(
            input_dim, embedding_dim, rewiring_prob
        )

        # Small-world transformer layers
        self.sw_layers = nn.ModuleList()

        for _ in range(num_layers):
            # Self-attention with small-world connections
            self.sw_layers.append(
                nn.ModuleDict(
                    {
                        "attention": nn.MultiheadAttention(
                            embed_dim=embedding_dim,
                            num_heads=num_heads,
                            batch_first=True,
                        ),
                        "sw_ffn": SmallWorldLayer(
                            embedding_dim, embedding_dim, rewiring_prob
                        ),
                        "layer_norm1": nn.LayerNorm(embedding_dim),
                        "layer_norm2": nn.LayerNorm(embedding_dim),
                    }
                )
            )

        # Global integration layer
        self.global_integration = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            SmallWorldLayer(embedding_dim, embedding_dim, rewiring_prob=0.2),
            nn.ReLU(),
        )

        # Pointer and flag classifier
        self.pointer = nn.Linear(embedding_dim, 2)

    def forward(self, solutions):
        # Get embeddings for each node type
        source_emb, candidates_emb, destination_emb, depot_emb = self.node_embedder(
            solutions
        )

        # Combine for decoder input
        x = torch.cat((source_emb, candidates_emb, destination_emb, depot_emb), dim=1)

        # Process through small-world transformer layers
        for layer in self.sw_layers:
            # Self-attention with residual connection
            residual = x
            x_norm = layer["layer_norm1"](x)
            x_attn, _ = layer["attention"](x_norm, x_norm, x_norm)
            x = x_attn + residual

            # Small-world FFN with residual connection
            residual = x
            x_norm = layer["layer_norm2"](x)
            x_sw = layer["sw_ffn"](x_norm)
            x = x_sw + residual

        # Global integration for improved information flow
        x = self.global_integration(x)

        logits = self.pointer(x)

        candidate_logits = logits[:, 1:-2, 0]
        flag_logits = logits[:, 1:-2, 1]

        return torch.cat([candidate_logits, flag_logits], dim=1)


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


class SmallWorldEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_dim = self.model_params["embedding_dim"]

        self.gegnn = GeGnn(self.model_params["pretrained_encoder"]).to(self.device)
        self.gegnn.compute_embeddings()

        # Replace standard transition layer with small-world layer
        self.transition_layer = SmallWorldLayer(
            self.model_params["pretrained_encoder"]["out_channels"],
            embedding_dim,
            rewiring_prob=model_params.get("rewiring_prob", 0.1),
        )

        # Add global integration layer
        self.global_integration = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

    def prepare_embedding(self, city_indexes):
        self.gegnn.embds = self.gegnn.embds[city_indexes]

    def forward(self, solutions):
        out = self.gegnn(solutions[:, :, 0].to(torch.int64))

        # Apply small-world transition and global integration
        out = self.transition_layer(out)
        out = self.global_integration(out)

        return out
