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
        self.decoder = DecoderCity(**model_params)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoded_nodes = None

    def forward(
        self,
        solutions,
        capacities=None,
    ):
        # solution's shape : [B, V]
        # TODO: This only works if all capacities are the same I guess
        self.capacity = capacities.ravel()[0].item()
        memory = self.encoder.gegnn.embds  # TODO: Quick fix

        remaining_capacity = solutions[:, 0, 3]

        encoder_out = self.encoder(solutions)

        encoder_out = torch.cat(
            (
                encoder_out,
                solutions[:, :, 2].unsqueeze(-1) / self.capacity,
            ),
            dim=2,
        )

        # add remaining capacity for the source node TODO: Does it make sense to just leave this as is without this step?
        encoder_out[:, 0, -1] = remaining_capacity / self.capacity

        # TODO: Check this and how to handle memory
        logits_node, logits_flag = self.decoder(encoder_out, memory)

        return logits_node, logits_flag


class DecoderCityHelp(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        num_layers,
        output_dim=10000,
        decision_dim=2,
    ):
        super(DecoderCityHelp, self).__init__()

        # Linear projection layer to map input to correct dimension
        self.input_projection = nn.Linear(input_dim, input_dim)

        # Transformer Decoder Layers (self-attention-based)
        self.self_attention = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(self.self_attention, num_layers=num_layers)

        # Fully connected layers for outputs
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.decision_layer = nn.Linear(input_dim, decision_dim)

    def forward(self, x, memory):
        """
        x: (batch_size, seq_len, input_dim) - The decoder input.
        memory: (batch_size, seq_len, input_dim) - The encoder output.
        """
        # Project input into correct dimensional space
        x = self.input_projection(x)

        # expand the memory to the batch size of x
        memory = memory.expand(x.shape[0], -1, -1)

        decoded = self.decoder(x, memory)

        # take last entry of the layer output
        pooled = decoded[:, -1, :]

        # Aggregate information using global mean pooling
        # pooled = decoded.mean(dim=1)  # Reduces seq_len dimension

        # TODO: This the best way to do this?
        output_tensor = self.output_layer(pooled)  # (batch_size, city_size)
        decision_tensor = self.decision_layer(pooled)  # (batch_size, 2)

        return output_tensor, decision_tensor


class DecoderCity(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        embedding_dim = 32

        self.embedd_source = nn.Linear(embedding_dim + 1, embedding_dim, bias=True)
        self.embedding_destination = nn.Linear(
            embedding_dim + 1, embedding_dim, bias=True
        )
        self.embedd_depot = nn.Linear(embedding_dim + 1, embedding_dim, bias=True)
        self.embedding_candidates = nn.Linear(
            embedding_dim + 1, embedding_dim, bias=True
        )

        self.decoder = DecoderCityHelp(
            input_dim=embedding_dim,
            hidden_dim=model_params["hidden_dim"],
            num_heads=model_params["head_num"],
            num_layers=model_params["decoder_layer_num"],
            output_dim=model_params["city_size"],
            decision_dim=2,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, encoded_problems_reindexed, memory):
        source = encoded_problems_reindexed[:, [0], :]
        candidates = encoded_problems_reindexed[:, 1:-2, :]
        destination = encoded_problems_reindexed[:, [-2], :]
        depot = encoded_problems_reindexed[:, [-1], :]

        # perform linear transformation on the embeddings
        source = self.embedd_source(source)
        destination = self.embedding_destination(destination)
        depot = self.embedd_depot(depot)
        candidates = self.embedding_candidates(candidates)

        # TODO: think about having depot, source and destination in the memory (or at least the depot)
        decoder_input = torch.cat((source, candidates, destination, depot), dim=1)
        logits_student, logits_flag = self.decoder(decoder_input, memory)

        return logits_student, logits_flag


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

    def forward(self, problems):
        out = self.gegnn(problems[:, :, 0].to(torch.int64))
        return out
