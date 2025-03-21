import torch
import torch.nn as nn
from typing import Dict

from main_model.src.utils.hgraph.hgraph import HGraph
from main_model.src.utils.hgraph.modules.resblocks import GraphResBlocks, GraphResBlock2
from main_model.src.utils.hgraph.modules import modules


class GraphUNet(nn.Module):
    """
    A U-Net like network with graph neural network, utilizing HGraph (hierarchical graph) as
    the data structure.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(GraphUNet, self).__init__()

        # The number of input, hidden, and output channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # The number of blocks in each stage of the encoder and decoder
        # self.encoder_blocks = [2, 3, 3, 3, 2]
        # self.decoder_blocks = [2, 3, 3, 3, 2]
        self.encoder_blocks = [3, 4, 4, 4, 3]
        self.decoder_blocks = [3, 4, 4, 4, 3]
        self.encoder_stages = len(self.encoder_blocks)
        self.decoder_stages = len(self.decoder_blocks)

        self.bottleneck = 1
        self.resblk = GraphResBlock2

        # Encoder
        self.conv1 = modules.GraphConvBnRelu(in_channels, self.hidden_channels)

        self.downsample = nn.ModuleList(
            [modules.PoolingGraph() for i in range(self.encoder_stages)]
        )
        self.encoder = nn.ModuleList(
            [
                GraphResBlocks(
                    self.hidden_channels,
                    self.hidden_channels,
                    resblk_num=self.encoder_blocks[i],
                    resblk=self.resblk,
                )
                for i in range(self.encoder_stages)
            ]
        )

        # Decoder
        channel = [
            self.hidden_channels + self.hidden_channels
            for i in range(self.decoder_stages)
        ]
        self.upsample = nn.ModuleList(
            [modules.UnpoolingGraph() for i in range(self.decoder_stages)]
        )
        self.decoder = nn.ModuleList(
            [
                GraphResBlocks(
                    channel[i],
                    self.hidden_channels,
                    resblk_num=self.decoder_blocks[i],
                    resblk=self.resblk,
                    bottleneck=self.bottleneck,
                )
                for i in range(self.decoder_stages)
            ]
        )

        # header
        self.header = nn.Sequential(
            modules.Conv1x1BnRelu(self.hidden_channels, self.hidden_channels),
            modules.Conv1x1(self.hidden_channels, self.out_channels, use_bias=True),
        )

        # an embedding decoder function
        self.embedding_decoder_mlp = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels, bias=True),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels, bias=True),
            nn.ReLU(),
            nn.Linear(self.out_channels, 1, bias=True),
        )

    def unet_encoder(self, data: torch.Tensor, hgraph: HGraph, depth: int):
        """
        The encoder of the U-Net.
        """

        convd = dict()
        convd[depth] = self.conv1(data, hgraph, depth)
        for i in range(self.encoder_stages):
            d = depth - i
            conv = self.downsample[i](convd[d], hgraph, i + 1)
            convd[d - 1] = self.encoder[i](conv, hgraph, d - 1)
        return convd

    def unet_decoder(self, convd: Dict[int, torch.Tensor], hgraph: HGraph, depth: int):
        """
        The decoder of the U-Net.
        """

        deconv = convd[depth]
        for i in range(self.decoder_stages):
            d = depth + i
            deconv = self.upsample[i](deconv, hgraph, self.decoder_stages - i)
            deconv = torch.cat([convd[d + 1], deconv], dim=1)  # skip connections
            deconv = self.decoder[i](deconv, hgraph, d + 1)
        return deconv

    def forward(
        self,
        data: torch.Tensor,
        hgraph: HGraph,
        depth: int,
        dist: torch.Tensor,
        only_embd=False,
    ):
        """
        Forward pass of the network. This function is used for training and testing. The network
        will return the predicted distance between the input points.

        Args:
          data: torch.Tensor, the input features.
          hgraph: HGraph, the hierarchical graph.
          depth: int, the depth of the hierarchical graph.
          dist: torch.Tensor, the ground truth distance between the input points.
          only_embd: bool, if True, only the embedding will be returned.

        Returns:
          torch.Tensor, the predicted distance between the input points
        """
        convd = self.unet_encoder(data, hgraph, depth)
        deconv = self.unet_decoder(convd, hgraph, depth - self.encoder_stages)

        embedding = self.header(deconv)

        if dist == None and only_embd:
            return embedding

        # calculate the distance
        i, j = dist[:, 0].long(), dist[:, 1].long()

        embd_i = embedding[i].squeeze(-1)
        embd_j = embedding[j].squeeze(-1)

        embd = (embd_i - embd_j) ** 2

        pred = self.embedding_decoder_mlp(embd)
        pred = pred.squeeze(-1)

        if only_embd:
            return embedding
        else:
            return pred
