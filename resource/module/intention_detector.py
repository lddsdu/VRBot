# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from resource.option.train_option import TrainOption as TO
from resource.util.misc import reverse_sequence_mask
from resource.module.base_attention import Attention


class IntentionDetector(nn.Module):
    def __init__(self, know_vocab_size, intention_cate=4, hidden_dim=512, embed_dim=300, graph_dim=128, is_prior=True):
        super(IntentionDetector, self).__init__()
        self.know_vocab_size = know_vocab_size
        self.intention_cate = intention_cate
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.is_prior = is_prior
        self.graph_dim = graph_dim if self.is_prior else 0

        if not self.is_prior:
            self.res_attn = Attention(self.hidden_dim, self.hidden_dim)
        self.his_attn = Attention(self.hidden_dim, self.hidden_dim)
        self.state_attn = Attention(self.hidden_dim, self.embed_dim)

        intention_mlp_input_size = self.hidden_dim + self.hidden_dim + self.embed_dim

        if not self.is_prior:
            intention_mlp_input_size += self.hidden_dim

        if self.is_prior:  # prior policy network, use the graph embedding
            intention_mlp_input_size += self.graph_dim

        self.intention_mlp = nn.Sequential(nn.Linear(intention_mlp_input_size, self.hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_dim, self.intention_cate),
                                           nn.Softmax())

    def forward(self, state_emb, hidden, pv_r_u_enc, pv_r_u_len=None, r_enc=None, graph_context=None):
        intention_input = []

        # r
        if not self.is_prior:
            r_inp, _ = self.res_attn.forward(hidden.permute(1, 0, 2), r_enc)
            intention_input.append(r_inp)

        # pv_r_u
        h_inp, _ = self.his_attn.forward(hidden.permute(1, 0, 2),
                                         pv_r_u_enc,
                                         mask=reverse_sequence_mask(pv_r_u_len, max_len=pv_r_u_enc.size(1)))
        intention_input.append(h_inp)

        # state
        s_inp, _ = self.state_attn.forward(hidden.permute(1, 0, 2), state_emb)
        intention_input.append(s_inp)

        # question_hidden
        intention_input.append(hidden.permute(1, 0, 2))
        intention_input = torch.cat(intention_input, dim=-1).squeeze(1)  # B, E + H + H [+ H]

        if graph_context is not None and self.is_prior:
            intention_input = torch.cat([intention_input, graph_context.squeeze(1)], dim=-1)

        # intention project
        intention = self.intention_mlp.forward(intention_input)  # B, I
        return intention
