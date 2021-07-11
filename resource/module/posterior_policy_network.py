# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from resource.module.base_attention import Attention
from resource.module.basic_policy_network import BasicPolicyNetwork
from resource.module.intention_detector import IntentionDetector
from resource.module.db_querier import DBQuerier
from resource.module.rnn_encoder import RNNEncoder


class PosteriorPolicyNetwork(nn.Module):
    def __init__(self, action_num, hidden_dim, know_vocab_size,
                 embed_dim, embedder, lg_interpreter, gen_strategy,
                 with_copy, know2word_tensor, know_encoder: RNNEncoder = None):
        super(PosteriorPolicyNetwork, self).__init__()
        self.action_num = action_num
        self.hidden_dim = hidden_dim
        self.know_vocab_size = know_vocab_size
        self.embed_dim = embed_dim
        self.embedder = embedder
        self.lg_interpreter = lg_interpreter
        self.gen_strategy = gen_strategy
        self.with_copy = with_copy
        self.know2word_tensor = know2word_tensor
        self.embed_attn = Attention(hidden_dim, embed_dim)
        self.hidden_attn = Attention(hidden_dim, hidden_dim)

        if know_encoder is not None:
            self.know_encoder = know_encoder
        else:
            self.know_encoder = RNNEncoder(self.embed_dim,
                                           self.hidden_dim,
                                           self.hidden_dim // 2,
                                           num_layers=1,
                                           bidirectional=True,
                                           dropout=0,
                                           embedder=self.embedder)

        self.embed2hidden_linear = nn.Linear(self.embed_dim, self.hidden_dim)
        self.intention_detector = IntentionDetector(know_vocab_size,
                                                    hidden_dim=self.hidden_dim,
                                                    embed_dim=self.embed_dim,
                                                    is_prior=False)
        self.basic_policy_network = BasicPolicyNetwork(self.action_num,
                                                       self.hidden_dim,
                                                       self.embed_dim,
                                                       self.know_vocab_size,
                                                       self.embedder,
                                                       self.lg_interpreter,
                                                       self.know2word_tensor,
                                                       self.gen_strategy,
                                                       self.with_copy)

        self.rnn_cell = nn.GRU(input_size=self.embed_dim,
                               hidden_size=self.embed_dim,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=False)

    def forward(self, hidden, state, pv_r_u_enc, pv_r_u_len, r, r_enc,
                mask_gen=False, gth_action=None, supervised=False):
        state_word = self.lg_interpreter.loc2glo(state)
        state_embed = self.embedder(state_word)
        tmp = []
        s_hidden = None
        for i in range(state_embed.size(1)):
            _, s_hidden = self.rnn_cell.forward(state_embed[:, i: i+1, :], s_hidden)  # [1, B, H]
            tmp.append(s_hidden.permute(1, 0, 2))  # [B, 1, E] * S
        state_embed = torch.cat(tmp, 1)  # B, S, E

        intention = self.intention_detector.forward(state_embed,
                                                    hidden,
                                                    pv_r_u_enc,
                                                    pv_r_u_len,
                                                    r_enc=r_enc)

        action, gumbel_action = self.basic_policy_network.forward(hidden.squeeze(0),
                                                                  state_embed,
                                                                  state,
                                                                  pv_r_u_enc,
                                                                  pv_r_u_len,
                                                                  r=r, r_enc=r_enc,
                                                                  mask_gen=mask_gen,
                                                                  gth_action=gth_action,
                                                                  supervised=supervised)

        return intention, action, gumbel_action
