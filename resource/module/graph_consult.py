# -*- coding: utf-8 -*-

import torch.nn as nn
from resource.module.base_attention import Attention


class GraphAttn(nn.Module):
    def __init__(self, node_embed_dim, hidden_dim):
        super(GraphAttn, self).__init__()
        self.node_embed_dim = node_embed_dim
        self.hidden_dim = hidden_dim
        # attention read context
        self.attn = Attention(self.hidden_dim, self.node_embed_dim)

    def forward(self, node_embedding, node_efficient, head_flag_bit_matrix, h_c_t):
        efficient_mask = (1 - node_efficient)  # B, N
        efficient_mask = efficient_mask | head_flag_bit_matrix
        node_context, _ = self.attn.forward(h_c_t,
                                            node_embedding,
                                            mask=efficient_mask.bool())  # B, 1, E
        return node_context


class GraphCopy(nn.Module):
    def __init__(self, node_embed_dim, hidden_dim):
        super(GraphCopy, self).__init__()
        self.node_embed_dim = node_embed_dim
        self.hidden_dim = hidden_dim
        self.copy_attn = Attention(self.hidden_dim, self.node_embed_dim)

    def forward(self, node_embedding, node_efficient, head_flag_bit_matrix, h_c_t):
        efficient_mask = (1 - node_efficient)  # B, N
        efficient_mask = efficient_mask | head_flag_bit_matrix
        node_logits = self.copy_attn.forward(h_c_t,
                                             node_embedding,
                                             mask=efficient_mask.bool(),
                                             not_softmax=True,
                                             return_weight_only=True)  # B, 1, N
        return node_logits
