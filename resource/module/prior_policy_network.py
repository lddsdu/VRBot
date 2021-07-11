# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from resource.module.base_attention import Attention
from resource.module.basic_policy_network import BasicPolicyNetwork
from resource.module.mygat import MyGAT
from resource.module.graph_consult import GraphAttn
from resource.module.intention_detector import IntentionDetector
from resource.option.train_option import TrainOption as TO
from resource.option.vrbot_option import VRBotOption as VO
from resource.model.vrbot_train_state import vrbot_train_stage
from resource.util.misc import one_hot_scatter
from resource.input.graph_db import GraphDB


class PriorPolicyNetwork(nn.Module):
    def __init__(self,
                 action_num,
                 hidden_dim,
                 know_vocab_size,
                 embed_dim,
                 embedder,
                 lg_interpreter,
                 gen_strategy,
                 with_copy,
                 graph_db: GraphDB,
                 know2word_tensor,
                 intention_gumbel_softmax):
        super(PriorPolicyNetwork, self).__init__()
        self.action_num = action_num
        self.hidden_dim = hidden_dim
        self.know_vocab_size = know_vocab_size
        self.embed_dim = embed_dim
        self.embedder = embedder
        self.lg_interpreter = lg_interpreter
        self.gen_strategy = gen_strategy
        self.with_copy = with_copy
        self.graph_db = graph_db
        self.intention_gumbel_softmax = intention_gumbel_softmax
        self.know2word_tensor = know2word_tensor
        self.embed_attn = Attention(hidden_dim, embed_dim)
        self.hidden_attn = Attention(hidden_dim, hidden_dim)
        self.hidden_type_linear = nn.Linear(hidden_dim + 4, hidden_dim)

        self.embed2hidden_linear = nn.Linear(self.embed_dim, self.hidden_dim)
        self.intention_detector = IntentionDetector(know_vocab_size,
                                                    hidden_dim=self.hidden_dim,
                                                    embed_dim=self.embed_dim,
                                                    graph_dim=VO.GATConfig.embed_dim,
                                                    is_prior=True)

        self.basic_policy_network = BasicPolicyNetwork(self.action_num,
                                                       self.hidden_dim,
                                                       self.embed_dim,
                                                       self.know_vocab_size,
                                                       self.embedder,
                                                       self.lg_interpreter,
                                                       self.know2word_tensor,
                                                       self.gen_strategy,
                                                       self.with_copy)

        self.graph_attn = GraphAttn(VO.node_embed_dim, VO.hidden_dim)
        self.rnn_enc_cell = nn.GRU(input_size=self.embed_dim,
                                   hidden_size=self.embed_dim,
                                   num_layers=1,
                                   batch_first=True,
                                   bidirectional=False)

        self.r_gat = None
        gatc = VO.GATConfig
        self.r_gat = MyGAT(gatc.embed_dim, gatc.edge_embed_dim, gatc.flag_embed_dim,
                           gatc.node_num, gatc.edge_num, gatc.flag_num)

    def forward(self, hidden, state, gth_intention, pv_r_u_enc,
                pv_r_u_len=None, gth_action=None, supervised=False):
        state_word = self.lg_interpreter.loc2glo(state)
        state_embed = self.embedder(state_word)
        tmp = []
        s_hidden = None
        for i in range(state_embed.size(1)):
            _, s_hidden = self.rnn_enc_cell.forward(state_embed[:, i: i + 1, :], s_hidden)  # [1, B, H]
            tmp.append(s_hidden.permute(1, 0, 2))  # [B, 1, E] * S
        state_embed = torch.cat(tmp, 1)  # B, S, E

        node_embedding = None
        head_nodes = None
        node_efficient = None
        head_flag_bit = None
        graph_context = None

        rets = self.graph_db.graph_construct(state.cpu().numpy().tolist())

        adjacent_matrix, head_nodes, node_efficient, head_flag_bit, edge_type_matrix = [
            torch.tensor(r).to(TO.device) for r in rets]
        node_embedding = self.r_gat.forward(adjacent_matrix, head_nodes, head_flag_bit, edge_type_matrix)
        graph_context = self.graph_attn.forward(node_embedding, node_efficient, head_flag_bit,
                                                hidden.permute(1, 0, 2))

        intention = self.intention_detector.forward(state_embed,
                                                    hidden,
                                                    pv_r_u_enc,
                                                    pv_r_u_len,
                                                    r_enc=None,
                                                    graph_context=graph_context)

        gumbel_intention = self.intention_gumbel_softmax.forward(intention, vrbot_train_stage.a_tau)

        if (gth_intention is None) or (not self.training):
            last_dim = gumbel_intention.size(-1)
            gth_intention = one_hot_scatter(intention.argmax(-1), last_dim, dtype=torch.float)

        hidden = self.hidden_type_linear.forward(torch.cat([hidden.squeeze(0), gth_intention], dim=-1))

        action, gumbel_action = self.basic_policy_network.forward(hidden,
                                                                  state_embed,
                                                                  state,
                                                                  pv_r_u_enc,
                                                                  pv_r_u_len,
                                                                  r=None,
                                                                  r_enc=None,
                                                                  gth_action=gth_action,
                                                                  supervised=supervised,
                                                                  node_embedding=node_embedding,  # B, N, E
                                                                  head_nodes=head_nodes,  # B, N
                                                                  node_efficient=node_efficient,  # B, N
                                                                  head_flag_bit=head_flag_bit)  # B, N

        return intention, action, gumbel_action
