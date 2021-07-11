# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from resource.module.base_attention import Attention
from resource.util.loc_glo_trans import LocGloInterpreter
from resource.option.train_option import TrainOption as TO
from resource.option.vrbot_option import VRBotOption as VO
from resource.model.vrbot_train_state import vrbot_train_stage
from resource.util.misc import one_hot_scatter
from resource.util.misc import reverse_sequence_mask
from resource.module.gumbel_softmax import GumbelSoftmax
from resource.module.graph_consult import GraphCopy


class Reshape(nn.Module):
    def __init__(self, axis, new_dim):
        super(Reshape, self).__init__()
        self.axis = axis
        self.new_dim = new_dim

    def forward(self, tensor):
        origin_shape = list(tensor.shape)
        new_shape = origin_shape[:self.axis] + self.new_dim + origin_shape[self.axis + 1:]
        reshaped_tensor = tensor.reshape(new_shape)
        return reshaped_tensor


class BasicPolicyNetwork(nn.Module):
    def __init__(self,
                 action_num,
                 hidden_dim,
                 embed_dim,
                 know_vocab_size,
                 embedder,
                 kw_interpreter: LocGloInterpreter,
                 know2word_tensor,
                 gen_strategy=None,
                 with_copy=True):
        super(BasicPolicyNetwork, self).__init__()
        self.action_num = action_num
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.know_vocab_size = know_vocab_size
        self.embedder = embedder
        self.kw_interpreter = kw_interpreter
        self.gen_strategy = gen_strategy
        self.with_copy = with_copy
        self.know2word_tensor = know2word_tensor
        self.gumbel_softmax = GumbelSoftmax(normed=True)
        self.graph_copy = GraphCopy(VO.node_embed_dim, VO.hidden_dim)
        assert self.gen_strategy in ("gru", "mlp")

        if self.with_copy:
            self.embed_copy_attn = Attention(self.hidden_dim, self.hidden_dim)
            self.hidden_copy_attn = Attention(self.hidden_dim, self.hidden_dim)

        if self.gen_strategy == "gru":
            self.embed_attn = Attention(self.hidden_dim, self.embed_dim)
            self.hidden_attn = Attention(self.hidden_dim, self.hidden_dim)
            self.embed2hidden_linear = nn.Linear(self.embed_dim, self.hidden_dim)
            self.gru = nn.GRU(input_size=self.embed_dim + self.embed_dim + self.hidden_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=1,
                              dropout=0.0,
                              batch_first=True)

        elif self.gen_strategy == "mlp":
            self.eh_linear = nn.Sequential(
                nn.Linear(self.hidden_dim + self.embed_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.trans_attn = Attention(self.hidden_dim, self.hidden_dim)  # for interaction between state
            self.action_attn = Attention(self.hidden_dim, self.hidden_dim)  # for attentively read state
            self.pred_proj = nn.Sequential(
                nn.Linear(self.hidden_dim + self.hidden_dim + self.embed_dim, self.hidden_dim * self.action_num),
                Reshape(1, [self.action_num, self.hidden_dim]))

        else:
            raise NotImplementedError

        self.gen_linear = nn.Linear(self.hidden_dim, self.know_vocab_size)
        self.hidden_projection = nn.Linear(self.hidden_dim, self.know_vocab_size)
        self.word_softmax = nn.Softmax(-1)
        self.hidden2word_projection = nn.Sequential(
            self.hidden_projection,
            self.word_softmax
        )

    def forward(self, hidden, state_emb, state, pv_r_u_enc, pv_r_u_len,
                r=None, r_enc=None, r_mask=None, mask_gen=False,
                gth_action=None, supervised=False,
                node_embedding=None,
                head_nodes=None, node_efficient=None, head_flag_bit=None):

        if self.gen_strategy == "gru":
            return self.forward_gru(hidden, state_emb,
                                    pv_r_u_enc, pv_r_u_len,
                                    r=r, r_enc=r_enc,
                                    r_mask=r_mask, gth_action=gth_action,
                                    mask_gen=mask_gen, supervised=supervised,
                                    node_embedding=node_embedding,  # B, N, E
                                    head_nodes=head_nodes,  # B, N
                                    node_efficient=node_efficient,  # B, N
                                    head_flag_bit=head_flag_bit)
        elif self.gen_strategy == "mlp":
            return self.forward_mlp(hidden, state_emb, r=r, r_enc=r_enc,
                                    r_mask=r_mask, mask_gen=mask_gen), None
        else:
            raise NotImplementedError

    def forward_gru(self, hidden, state_emb,
                    pv_r_u_enc, pv_r_u_len,
                    r=None, r_enc=None,
                    r_mask=None, gth_action=None,
                    mask_gen=False, supervised=False,
                    node_embedding=None,  # B, N, E
                    head_nodes=None,  # B, N
                    node_efficient=None,  # B, N
                    head_flag_bit=None):
        batch_size = hidden.size(0)

        # B, 1, E
        state_context, _ = self.embed_attn.forward(hidden.unsqueeze(1), state_emb)
        hidden = hidden + self.embed2hidden_linear(state_context).squeeze(1)
        hidden = hidden.unsqueeze(0)  # 1, B, H

        # init input
        step_input = torch.zeros(batch_size, 1, self.know_vocab_size, dtype=torch.float, device=TO.device)
        step_input[:, :, 0] = 1.0  # B, 1, K
        step_input_embed = self.know_prob_embed(step_input)  # B, 1, E

        actions = []
        gumbel_actions = []

        for i in range(self.action_num):
            # B, 1, E + H
            state_context, _ = self.embed_attn.forward(hidden.permute(1, 0, 2), state_emb)
            pv_r_u_mask = reverse_sequence_mask(pv_r_u_len, pv_r_u_enc.size(1))
            post_context, _ = self.hidden_attn.forward(hidden.permute(1, 0, 2), pv_r_u_enc,
                                                       mask=pv_r_u_mask)
            pv_s_input = torch.cat([step_input_embed, state_context, post_context], dim=-1)

            next_action_hidden, hidden = self.gru.forward(pv_s_input, hidden)

            probs = self.action_pred(batch_size, next_action_hidden,
                                     r=r, r_enc=r_enc, r_mask=r_mask, mask_gen=mask_gen,
                                     node_embedding=node_embedding, head_nodes=head_nodes,
                                     node_efficient=node_efficient, head_flag_bit=head_flag_bit)
            actions.append(probs)

            if self.training and TO.auto_regressive and (gth_action is not None) and supervised and (
                    not TO.no_action_super):
                gth_step_input = gth_action[:, i: i + 1]
                gth_step_input = one_hot_scatter(gth_step_input, self.know_vocab_size, dtype=torch.float)
                step_input_embed = self.know_prob_embed(gth_step_input)
            else:
                if TO.auto_regressive:
                    if self.training:
                        gumbel_probs = self.action_gumbel_softmax_sampling(probs)
                    else:
                        max_indices = probs.argmax(-1)
                        gumbel_probs = one_hot_scatter(max_indices, probs.size(2), dtype=torch.float)

                    step_input_embed = self.know_prob_embed(gumbel_probs)
                    gumbel_actions.append(gumbel_probs)
                else:
                    step_input_embed = self.know_prob_embed(probs)

        actions = torch.cat(actions, dim=1)
        if len(gumbel_actions) == 0:
            return actions, None

        gumbel_actions = torch.cat(gumbel_actions, dim=1)
        return actions, gumbel_actions

    def forward_mlp(self, hidden, state_emb,
                    r=None, r_enc=None, r_mask=None, mask_gen=False):
        batch_size, state_num = state_emb.size(0), state_emb.size(1)
        # B, S, H
        expanded_hidden = hidden.unsqueeze(0).permute(1, 0, 2).expand(batch_size, state_num, self.hidden_dim)
        # B, S, H
        deep_input = self.eh_linear(torch.cat([expanded_hidden, state_emb], dim=2))
        # B, S, H
        deep_inner, _ = self.trans_attn.forward(deep_input, deep_input)
        # B, 1, H
        deep_output, _ = self.action_attn.forward(hidden.unsqueeze(1), deep_inner)
        deep_output = deep_output.squeeze(1)
        # B, H + H + E
        proj_input = torch.cat([deep_output, hidden, state_emb.sum(1)], dim=-1)
        # B, A, H (for generation or for copy from )
        pred_action_hidden = self.pred_proj.forward(proj_input)

        return self.action_pred(batch_size, pred_action_hidden,
                                r, r_enc, r_mask, mask_gen=mask_gen)

    def action_pred(self, batch_size, pred_action_hidden,
                    r=None, r_enc=None,
                    r_mask=None, mask_gen=False,
                    node_embedding=None,  # B, N, E
                    head_nodes=None,  # B, N
                    node_efficient=None,  # B, N
                    head_flag_bit=None):  # B, N
        logits = []
        indexs = None

        action_num = pred_action_hidden.size(1)
        action_gen_logits = self.gen_linear(pred_action_hidden)
        logits.append(action_gen_logits)

        if r is not None:
            r_know_index = self.kw_interpreter.glo2loc(r)
            r_inner_mask = (r_know_index == 0)
            if r_mask is not None:
                r_mask = r_inner_mask | r_mask
            else:
                r_mask = r_inner_mask

            r_logits = self.hidden_copy_attn.forward(pred_action_hidden,
                                                     r_enc,
                                                     mask=r_mask,
                                                     not_softmax=True,
                                                     return_weight_only=True)
            logits.append(r_logits)
            indexs = r_know_index  # B, Tr

        if node_embedding is not None:
            node_copy_logits = self.graph_copy.forward(node_embedding, node_efficient,
                                                       head_flag_bit, pred_action_hidden)
            logits.append(node_copy_logits)
            indexs = head_nodes  # B, N

        if (r is None) or (not VO.ppn_dq):
            if len(logits) >= 1:
                logits = torch.cat(logits, -1)
            else:
                raise RuntimeError

            probs = self.word_softmax(logits)

            if indexs is None:
                return probs

            if not mask_gen:
                gen_probs = probs[:, :, :self.know_vocab_size]
                copy_probs = probs[:, :, self.know_vocab_size:]
            else:
                gen_probs = 0.0
                copy_probs = probs
        else:
            g_lambda = 0.05
            gen_probs = self.word_softmax(logits[0]) * (1 - g_lambda)
            copy_probs = self.word_softmax(logits[1]) * g_lambda

        copy_probs_placeholder = torch.zeros(batch_size, action_num,
                                             self.know_vocab_size, device=TO.device)
        if indexs is not None:
            expand_indexs = indexs.unsqueeze(1).expand(-1, action_num, -1)
            copy_probs = copy_probs_placeholder.scatter_add(2, expand_indexs, copy_probs)  # B, A, K

        action_probs = gen_probs + copy_probs  # B, A, K
        return action_probs

    def action_gumbel_softmax_sampling(self, probs):
        gumbel_probs = self.gumbel_softmax.forward(probs, vrbot_train_stage.a_tau)
        return gumbel_probs

    def know_prob_embed(self, action_prob):
        B, A, K = action_prob.shape
        # K, E
        know_embedding = self.embedder(self.know2word_tensor)
        action_embed = torch.bmm(action_prob.reshape(B * A, 1, K),
                                 know_embedding.unsqueeze(0).expand(B * A, K, self.embed_dim))
        action_embed = action_embed.reshape(B, A, self.embed_dim)
        return action_embed
