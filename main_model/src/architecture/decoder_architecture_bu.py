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

        self.decoder = HierarchicalCapacityAwareDecoder(
            input_dim=model_params["pretrained_encoder"]["out_channels"] + 1,
            embedding_dim=model_params["embedding_dim"],
            hidden_dim=model_params["hidden_dim"],
            num_heads=model_params["head_num"],
            num_layers=model_params["decoder_layer_num"],
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, solutions, capacities=None):
        remaining_capacity = solutions[:, 0, 3]
        encoder_out = self.encoder(solutions)

        normalized_demand = 2 * (
            (solutions[:, :, 2] - remaining_capacity[:, None]) / capacities[:, None]
        )

        encoder_out = torch.cat((encoder_out, normalized_demand.unsqueeze(-1)), dim=2)
        encoder_out[:, 0, -1] = 0.0

        logits_node, logits_flag = self.decoder(encoder_out)

        return logits_node, logits_flag


class HierarchicalCapacityAwareDecoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_heads, num_layers):
        super(HierarchicalCapacityAwareDecoder, self).__init__()

        # Node embedder
        self.node_embedder = NodeEmbedder(input_dim, embedding_dim)

        # Capacity-aware embeddings
        self.capacity_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
        )

        # Demand-aware embeddings
        self.demand_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
        )

        # Global planning module
        self.global_planner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
            ),
            num_layers=num_layers // 3,
        )

        # Flag prediction with capacity awareness
        self.flag_predictor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim * 3,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
            ),
            num_layers=num_layers // 3,
        )

        self.flag_classifier = nn.Linear(embedding_dim * 3, 2)

        # Local routing module with flag information
        self.local_router = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
            ),
            num_layers=num_layers // 3,
        )

        # Pointer for node selection
        self.pointer = nn.Linear(embedding_dim, 1)

        self.linear_memory = nn.Linear(embedding_dim + 2, embedding_dim)

    def forward(self, solutions):
        # Get embeddings
        source_emb, candidates_emb, destination_emb, depot_emb = self.node_embedder(
            solutions
        )

        # Extract capacity and demand information
        remaining_capacity = solutions[:, 0, 3].unsqueeze(-1)
        candidate_demands = solutions[:, 1:-2, 2].unsqueeze(-1)

        # Encode capacity and demands
        capacity_emb = self.capacity_encoder(remaining_capacity)
        demand_embs = self.demand_encoder(candidate_demands)

        # Calculate feasibility scores (capacity >= demand)
        remaining_capacity = remaining_capacity.unsqueeze(1).expand(
            -1, candidate_demands.size(1), -1
        )
        feasibility = (remaining_capacity >= candidate_demands).float()

        # Enhance candidate embeddings with demand information
        enhanced_candidates = candidates_emb + demand_embs * feasibility

        # Global planning: understand the overall route structure
        x_global = torch.cat(
            (source_emb, enhanced_candidates, destination_emb, depot_emb), dim=1
        )
        global_features = self.global_planner(x_global)

        # Extract global context
        global_context = global_features.mean(dim=1, keepdim=True)

        # Flag prediction with capacity awareness
        capacity_context = torch.cat([source_emb, capacity_emb.unsqueeze(1)], dim=2)
        flag_input = torch.cat(
            [capacity_context, global_context.expand(-1, capacity_context.size(1), -1)],
            dim=2,
        )
        flag_features = self.flag_predictor(flag_input)
        flag_logits = self.flag_classifier(flag_features[:, 0, :])

        # Get flag probabilities
        flag_probs = F.softmax(flag_logits, dim=1)

        # Create memory for decoder based on flag decision
        # Expand flag_probs to match the sequence length of enhanced_candidates
        flag_memory = flag_probs.unsqueeze(1).expand(
            -1, enhanced_candidates.size(1), -1
        )

        # Expand global_context to match the sequence length of enhanced_candidates
        global_context_expanded = global_context.expand(
            -1, enhanced_candidates.size(1), -1
        )

        # Concatenate global_context and flag_probs along the last dimension
        memory_input = torch.cat([global_context_expanded, flag_memory], dim=2)

        memory_input = self.linear_memory(memory_input)

        # Local routing with flag information as memory
        x_local = torch.cat(
            (source_emb, enhanced_candidates, destination_emb, depot_emb), dim=1
        )
        local_features = self.local_router(x_local, memory_input)

        # Generate pointer logits
        num_candidates = enhanced_candidates.size(1)
        candidate_logits = self.pointer(
            local_features[:, 1 : 1 + num_candidates, :]
        ).squeeze(-1)

        return candidate_logits, flag_logits


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

        # Node embedder
        self.node_embedder = NodeEmbedder(input_dim, embedding_dim)

        # Flag prediction
        self.flag_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
            ),
            num_layers=num_layers // 2,
        )

        self.flag_classifier = nn.Linear(embedding_dim, 2)

        # Memory key generator
        self.memory_key_generator = nn.Linear(embedding_dim + 2, embedding_dim)

        # Node selection
        self.node_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
            ),
            num_layers=num_layers // 2,
        )

        # Pointer for node selection
        self.pointer = nn.Linear(embedding_dim, 1)

    def forward(self, solutions):
        # Get embeddings
        source_emb, candidates_emb, destination_emb, depot_emb = self.node_embedder(
            solutions
        )

        # First step: predict flag
        x = torch.cat((source_emb, candidates_emb, destination_emb, depot_emb), dim=1)
        flag_features = self.flag_transformer(x)
        flag_logits = self.flag_classifier(flag_features[:, 0, :])

        # Get flag probabilities
        flag_probs = F.softmax(flag_logits, dim=1)

        # Generate memory keys based on flag probabilities
        batch_size = candidates_emb.size(0)
        num_candidates = candidates_emb.size(1)

        # Create memory keys for each candidate based on flag information
        flag_probs_expanded = flag_probs.unsqueeze(1).expand(-1, num_candidates, -1)
        memory_input = torch.cat([candidates_emb, flag_probs_expanded], dim=2)
        memory_keys = self.memory_key_generator(memory_input)

        # Process through transformer with memory-enhanced candidates
        x_with_memory = torch.cat(
            (source_emb, memory_keys, destination_emb, depot_emb), dim=1
        )
        node_features = self.node_transformer(x_with_memory)

        # Generate pointer logits
        candidate_logits = self.pointer(
            node_features[:, 1 : 1 + num_candidates, :]
        ).squeeze(-1)

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
