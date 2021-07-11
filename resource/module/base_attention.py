# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self,
                 query_size,
                 memory_size=None,
                 hidden_size=None,
                 mode="mlp"):
        super(Attention, self).__init__()
        assert mode in ["dot", "general", "mlp"]

        self.query_size = query_size
        self.memory_size = memory_size or query_size
        self.hidden_size = hidden_size or query_size
        self.mode = mode

        if mode == "general":
            self.linear_query = nn.Linear(self.query_size, self.memory_size, bias=False)
        elif mode == "mlp":
            self.linear_query = nn.Linear(self.query_size, self.hidden_size, bias=True)
            self.linear_memory = nn.Linear(self.memory_size, self.hidden_size, bias=False)
            self.v = nn.Linear(self.hidden_size, 1, bias=False)
            self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, memory, key=None, mask=None, not_softmax=False, return_weight_only=False):
        key = key if key is not None else memory
        masked_value = -1e24  # 0.0

        if self.mode == "dot":
            assert query.size(-1) == key.size(-1)
            attn = torch.bmm(query, key.transpose(1, 2))
        elif self.mode == "general":
            assert self.memory_size == key.size(-1)
            key_ = self.linear_query(query)
            attn = torch.bmm(key_, key.transpose(1, 2))
        else:
            hidden = self.linear_query(query).unsqueeze(2) + self.linear_memory(key).unsqueeze(1)
            key_ = self.tanh(hidden)
            attn = self.v(key_).squeeze(-1)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, query.size(1), -1)
            attn = attn.masked_fill(mask, masked_value)  # 多用点masked_fill吧，别用masked_fill_

        if not not_softmax:
            weights = self.softmax(attn)
        else:
            weights = attn

        if return_weight_only:
            return weights

        weighted_memory = torch.bmm(weights, memory)
        return weighted_memory, weights
