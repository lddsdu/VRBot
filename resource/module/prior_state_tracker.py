# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from resource.module.base_attention import Attention
from resource.module.basic_state_tracker import BasicStateTracker
from resource.util.loc_glo_trans import LocGloInterpreter
from resource.option.dataset_option import DatasetOption as DO
from resource.option.train_option import TrainOption as TO


class PriorStateTracker(nn.Module):
    def __init__(self,
                 state_num,
                 hidden_dim,
                 know_vocab_size,
                 embed_dim,
                 embedder,
                 lg_interpreter: LocGloInterpreter,
                 gen_strategy,
                 know2word_tensor,
                 with_copy=True):
        super(PriorStateTracker, self).__init__()
        self.state_num = state_num
        self.hidden_dim = hidden_dim
        self.know_vocab_size = know_vocab_size
        self.embed_dim = embed_dim
        self.embedder = embedder
        self.lg_interpreter = lg_interpreter
        self.gen_strategy = gen_strategy
        self.know2word_tensor = know2word_tensor
        self.with_copy = with_copy

        self.embed_attn = Attention(embed_dim, hidden_dim)
        self.prior_basic_state_tracker = BasicStateTracker(self.know2word_tensor,
                                                           self.state_num,
                                                           self.hidden_dim,
                                                           self.know_vocab_size,
                                                           self.embed_dim,
                                                           self.embedder,
                                                           self.lg_interpreter,
                                                           self.gen_strategy,
                                                           self.with_copy)

        self.rnn_cell = nn.GRU(input_size=self.embed_dim,
                               hidden_size=self.embed_dim,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=False)

    def forward(self, hidden, pv_state, pv_r_u, pv_r_u_enc, gth_state=None, supervised=False):
        # B, S, E
        pv_state_emb = self.know_prob_embed(pv_state)
        tmp = []
        s_hidden = None
        for i in range(pv_state_emb.size(1)):
            _, s_hidden = self.rnn_cell.forward(pv_state_emb[:, i: i+1, :], s_hidden)  # [1, B, H]
            tmp.append(s_hidden.permute(1, 0, 2))  # [B, 1, E] * S
        pv_state_emb = torch.cat(tmp, 1)  # B, S, E

        # B, 1, E
        pv_state_emb_mean = pv_state_emb.mean(1).unsqueeze(1)
        # B, 1, T
        weight = self.embed_attn.forward(pv_state_emb_mean,
                                         pv_r_u_enc,
                                         mask=pv_r_u <= DO.PreventWord.RESERVED_MAX_INDEX,
                                         not_softmax=True,
                                         return_weight_only=True)
        # B, 1, T
        weight = torch.softmax(weight, -1)
        # B, 1, H ==> B, H
        hidden = hidden.squeeze(0) + torch.bmm(weight, pv_r_u_enc).squeeze(1)

        states, gumbel_states = self.prior_basic_state_tracker.forward(hidden,
                                                                       pv_state,
                                                                       pv_state_emb,
                                                                       pv_r_u,
                                                                       pv_r_u_enc,
                                                                       gth_state=gth_state,
                                                                       supervised=supervised)
        return states, gumbel_states

    def know_prob_embed(self, state_gumbel_prob):
        B, S, K = state_gumbel_prob.shape
        # K, E
        know_embedding = self.embedder(self.know2word_tensor)
        state_gumbel_embed = torch.bmm(state_gumbel_prob.reshape(B * S, 1, K),
                                       know_embedding.unsqueeze(0).expand(B * S, K, self.embed_dim))
        state_gumbel_embed = state_gumbel_embed.reshape(B, S, self.embed_dim)
        return state_gumbel_embed
