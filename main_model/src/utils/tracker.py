# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import time
import torch
import torch.distributed
from typing import Dict
import logging


logger = logging.getLogger(__name__)


class AverageTracker:

    def __init__(self):
        self.value = None
        self.num = 0.0
        self.max_len = 76
        self.start_time = time.time()

    def update(self, value: Dict[str, torch.Tensor]):
        if not value:
            return  # empty input, return

        value = {key: val.detach() for key, val in value.items()}
        if self.value is None:
            self.value = value
        else:
            for key, val in value.items():
                self.value[key] += val
        self.num += 1

    def average(self):
        return {key: val.item() / self.num for key, val in self.value.items()}

    @torch.no_grad()
    def average_all_gather(self):
        for key, tensor in self.value.items():
            if not tensor.is_cuda:
                continue
            tensors_gather = [
                torch.ones_like(tensor)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
            tensors = torch.stack(tensors_gather, dim=0)
            self.value[key] = torch.mean(tensors)
