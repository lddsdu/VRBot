# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class MyGAT(nn.Module):
    def __init__(self,
                 embed_dim, edge_embed_dim, flag_embed_dim,
                 node_num, edge_num, flag_num):
        super(MyGAT, self).__init__()
        self.embed_dim = embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.flag_embed_dim = flag_embed_dim
        self.node_num = node_num
        self.edge_num = edge_num
        self.flag_num = flag_num

        # networks
        self.node_embedding = nn.Embedding(self.node_num, self.embed_dim)
        self.edge_embedding = nn.Embedding(self.edge_num, self.edge_embed_dim)
        self.flag_embedding = nn.Embedding(self.flag_num, self.flag_embed_dim)

        self.in_embed_dim = self.embed_dim * 2 + self.edge_embed_dim + \
                            self.flag_embed_dim * 2
        self.w1 = nn.Linear(self.in_embed_dim, 1)
        self.t1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w2 = nn.Linear(self.in_embed_dim, 1)
        self.t2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, adjacent_matrix, head_nodes, head_flag_bit_matrix, edge_type_matrix):
        n = adjacent_matrix.size(1)

        # 1st propagate
        tail_embedding_matrix = head_embedding_matrix = self.t1(self.node_embedding(head_nodes))  # B, N, En
        head_tail_embedding_matrix = torch.cat([head_embedding_matrix.unsqueeze(2).expand(-1, -1, n, -1),
                                                tail_embedding_matrix.unsqueeze(1).expand(-1, n, -1, -1)],
                                               -1)  # B, N, N, En + En + Ee

        tail_flag_bit_embedding = head_flag_bit_embedding = self.flag_embedding(head_flag_bit_matrix)  # B, N, Ef
        edge_embedding_matrix = self.edge_embedding(edge_type_matrix)  # B, N, N, Ee
        flag_embedding_matrix = torch.cat([head_flag_bit_embedding.unsqueeze(2).expand(-1, -1, n, -1),
                                           tail_flag_bit_embedding.unsqueeze(1).expand(-1, n, -1, -1),
                                           edge_embedding_matrix], -1)

        first_layer_logit = self.w1(torch.cat([head_tail_embedding_matrix, flag_embedding_matrix], -1))
        first_layer_logit = self.leaky_relu(first_layer_logit)
        first_layer_logit = first_layer_logit.masked_fill((1 - adjacent_matrix.unsqueeze(-1)).bool(), -1e12)
        first_layer_weight = torch.softmax(first_layer_logit, 1).squeeze(-1)  # B, N(soft-max), N
        tail_embedding_matrix = torch.bmm(first_layer_weight.permute(0, 2, 1),
                                          head_embedding_matrix)  # B, N, En
        tail_embedding_matrix = torch.sigmoid(tail_embedding_matrix)

        head_embedding_matrix = tail_embedding_matrix = self.t2(tail_embedding_matrix)
        head_tail_embedding_matrix = torch.cat([head_embedding_matrix.unsqueeze(2).expand(-1, -1, n, -1),
                                                tail_embedding_matrix.unsqueeze(1).expand(-1, n, -1, -1)],
                                               -1)
        second_layer_logit = self.w2(torch.cat([head_tail_embedding_matrix, flag_embedding_matrix], -1))
        second_layer_logit = self.leaky_relu(second_layer_logit)
        second_layer_logit = second_layer_logit.masked_fill((1 - adjacent_matrix.unsqueeze(-1)).bool(), -1e12)
        second_layer_weight = torch.softmax(second_layer_logit, 1).squeeze(-1)
        tail_embedding_matrix = torch.bmm(second_layer_weight.permute(0, 2, 1), head_embedding_matrix)
        return tail_embedding_matrix
