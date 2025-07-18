"""
This file contains the GeGnnTrainer class, which is a subclass of the Solver class.
The code is adapted from the GeGnn paper titled "Learning the Geodesic Embedding with Graph Neural Networks" by Pang Bo et al (https://doi.org/10.1145/3618317).
Their GitHub repository can be found at https://github.com/IntelligentGeometry/GeGnn.
"""

import torch
import logging

from main_model.src.trainer.base_trainer import Solver
from main_model.src.architecture.encoder_architecture import GraphUNet
from main_model.src.data.encoder_dataloader import get_dataset

logger = logging.getLogger(__name__)


def get_parameter_number(model):
    """Log the number of parameters in a model on terminal."""
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nTotal Parameters: {total_num}, trainable: {trainable_num}")
    return {"Total": total_num, "Trainable": trainable_num}


class GeGnnTrainer(Solver):
    def __init__(self, config, is_master=True):
        super().__init__(config, is_master)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model(self, config):
        if config["name"].lower() == "unet":
            model = GraphUNet(
                config["in_channels"], config["hidden_channels"], config["out_channels"]
            )
        else:
            raise ValueError("Unknown model name")

        # Move model to the correct device
        model.to(self.device)

        # Log the number of parameters
        get_parameter_number(model)
        return model

    def get_dataset(self, config):
        return get_dataset(config)

    def model_forward(self, batch):
        """Equivalent to `self.get_embd` + `self.embd_decoder_func`"""
        data = batch["feature"].to(self.device)
        hgraph = batch["hgraph"]
        dist = batch["dist"].to(self.device)

        pred = self.model(data, hgraph, hgraph.depth, dist)
        return pred

    def get_embd(self, batch):
        """Only used in visualization!"""
        data = batch["feature"].to(self.device)
        hgraph = batch["hgraph"]
        dist = batch["dist"].to(self.device)

        embedding = self.model(data, hgraph, hgraph.depth, dist, only_embd=True)
        return embedding

    def embd_decoder_func(self, i, j, embedding):
        """Only used in visualization!"""
        i = i.long()
        j = j.long()
        embd_i = embedding[i].squeeze(-1)
        embd_j = embedding[j].squeeze(-1)
        embd = (embd_i - embd_j) ** 2
        pred = self.model.embedding_decoder_mlp(embd)
        pred = pred.squeeze(-1)
        return pred

    def train_step(self, batch):
        pred = self.model_forward(batch)
        loss = self.loss_function(batch, pred)
        return {"train/loss": loss}

    def test_step(self, batch):
        pred = self.model_forward(batch)
        loss = self.loss_function(batch, pred)
        return {"test/loss": loss}

    def loss_function(self, batch, pred):
        dist = batch["dist"].to(self.device)
        gt = dist[:, 2]

        # option 1: Mean Absolute Error, MAE
        # loss = torch.abs(pred - gt).mean()

        # option 2: relative MAE
        loss = (torch.abs(pred - gt) / (gt + 1e-3)).mean()

        # option 3: Mean Squared Error, MSE
        # loss = torch.square(pred - gt).mean()

        # option 4: relative MSE
        # loss = torch.square((pred - gt) / (gt + 1e-3)).mean()

        # option 5: root mean squared error, RMSE
        # loss = torch.sqrt(torch.square(pred - gt).mean())

        return loss

    def get_visualization_data(self):
        """helper function of visualization.
            return a dict, containing the following components:
              'filename': the name of the obj file to load
              'dist_func': a function: (i, j, embd) => int
              'embedding': embedding of vertices
            the dict will be used in interactive.py.
        Returns:
            a dict
        """
        # helper function, a utility for the visualization
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()

        self.model.eval()

        for it in tqdm(range(1), ncols=80, leave=False):
            batch = self.test_iter.__next__()
            with torch.no_grad():
                embedding = self.get_embd(batch)

        if self.config["model"]["get_test_stat"]:
            self.test_epoch(499)

        mesh_file = self.config["model"]["mesh_file"]
        if mesh_file == None:
            filename = batch["filename"][0]
        else:
            filename = mesh_file

        # specially designed for geodesic dist task. may not operate correctly on other tasks
        return {
            "filename": filename,
            "dist_func": self.embd_decoder_func,
            "embedding": embedding,
        }
