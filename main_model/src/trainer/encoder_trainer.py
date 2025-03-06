import torch

from hgraph.models.graph_unet import GraphUNet
from gegnn.utils.thsolver import default_settings, Solver
from gegnn.utils.thsolver.config import parse_args

# from gegnn.dataset_ps import get_dataset

from main_model.src.data.encoder_dataloader import get_dataset


def get_parameter_number(model):
    """Print the number of parameters in a model on terminal."""
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_num}, trainable: {trainable_num}")
    return {"Total": total_num, "Trainable": trainable_num}


class GeGnnSolver(Solver):
    def __init__(self, FLAGS, is_master=True):
        super().__init__(FLAGS, is_master)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model(self, flags):
        if flags.name.lower() == "unet":
            model = GraphUNet(
                flags.in_channels, flags.hidden_channels, flags.out_channels
            )
        else:
            raise ValueError("Unknown model name")

        # Move model to the correct device
        model.to(self.device)

        # Print the number of parameters
        get_parameter_number(model)
        return model

    def get_dataset(self, flags):
        return get_dataset(flags)

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

        loss = (torch.abs(pred - gt) / (gt + 1e-3)).mean()
        loss = torch.clamp(loss, -10, 10)

        return loss


if __name__ == "__main__":
    # Initialize global settings
    default_settings._init()
    FLAGS = parse_args(config_path="main_model/config.yaml")
    default_settings.set_global_values(FLAGS)

    solver = GeGnnSolver(FLAGS)
    solver.run()
