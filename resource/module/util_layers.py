# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class KMaxPooling(nn.Module):
    def __init__(self, top_k):
        super(KMaxPooling, self).__init__()
        self.top_k = top_k

    def forward(self, tensor):
        value, index = tensor.topk(self.top_k, dim=-1)
        return value, index
