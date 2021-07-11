# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from resource.option.train_option import TrainOption as TO
from resource.option.dataset_option import DatasetOption as DO
from resource.module.base_attention import Attention
from resource.util.loc_glo_trans import LocGloInterpreter
from resource.util.misc import one_hot_scatter
from resource.module.gumbel_softmax import GumbelSoftmax
from resource.model.vrbot_train_state import vrbot_train_stage


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


class BasicStateTracker(nn.Module):
    def __init__(self, know2word_tensor, state_num, hidden_dim, know_vocab_size, embed_dim, embedder,
                 lg_interpreter: LocGloInterpreter,
                 gen_strategy="gru", with_copy=True):
        super(BasicStateTracker, self).__init__()
        self.know2word_tensor = know2word_tensor
        self.state_num = state_num
        self.hidden_dim = hidden_dim
        self.know_vocab_size = know_vocab_size
        self.embed_dim = embed_dim
        self.embedder = embedder
        self.lg_interpreter = lg_interpreter
        self.gen_strategy = gen_strategy
        self.with_copy = with_copy
        self.gumbel_softmax = GumbelSoftmax(normed=True)
        assert self.gen_strategy in ("gru", "mlp") or self.gen_strategy is None

        self.embed_attn = Attention(self.hidden_dim, self.embed_dim)
        self.hidden_attn = Attention(self.hidden_dim, self.hidden_dim)

        if self.with_copy:
            self.embed_copy_attn = Attention(self.hidden_dim, self.embed_dim)
            self.hidden_copy_attn = Attention(self.hidden_dim, self.hidden_dim)

        if self.gen_strategy == "gru":
            self.gru = nn.GRU(input_size=self.embed_dim + self.hidden_dim + self.embed_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=1,
                              dropout=0.0,
                              batch_first=True)

        elif self.gen_strategy == "mlp":
            self.step_linear = nn.Sequential(
                nn.Linear(self.hidden_dim, self.state_num * self.hidden_dim),
                Reshape(1, [self.state_num, self.hidden_dim]),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim))
        else:
            raise NotImplementedError

        self.hidden_projection = nn.Linear(self.hidden_dim, self.know_vocab_size)
        self.word_softmax = nn.Softmax(-1)

        self.hidden2word_projection = nn.Sequential(
            self.hidden_projection,
            self.word_softmax
        )

    def state_gumbel_softmax_sampling(self, probs):
        gumbel_probs = self.gumbel_softmax.forward(probs, vrbot_train_stage.s_tau)
        return gumbel_probs

    def forward(self, hidden, pv_state, pv_state_emb, pv_r_u, pv_r_u_enc,
                gth_state=None, supervised=False):
        batch_size = pv_state_emb.size(0)
        states = []
        gumbel_states = []

        multi_hiddens = None
        step_input_embed = None
        if self.gen_strategy == "gru":
            hidden = hidden.unsqueeze(0)  # 1, B, H
            step_input = torch.zeros(batch_size, 1, self.know_vocab_size, dtype=torch.float, device=TO.device)
            step_input[:, :, 0] = 1.0  # B, 1, K
            step_input_embed = self.know_prob_embed(step_input)  # B, 1, E
        elif self.gen_strategy == "mlp":
            multi_hiddens = self.step_linear(hidden)  # B, S, H
        else:
            raise NotImplementedError

        for i in range(self.state_num):
            if self.gen_strategy == "gru":
                pv_s_context, _ = self.embed_attn.forward(hidden.permute(1, 0, 2), pv_state_emb)
                pv_r_u_context, _ = self.hidden_attn.forward(hidden.permute(1, 0, 2), pv_r_u_enc)
                pv_s_input = torch.cat([pv_s_context, pv_r_u_context, step_input_embed], dim=-1)  # B, 1, E + H + E
                next_state_hidden, hidden = self.gru.forward(pv_s_input, hidden)  # B, 1, H | 1, B, H
            elif self.gen_strategy == "mlp":
                next_state_hidden = multi_hiddens[:, i:i + 1, :]  # B, 1, H
            else:
                raise NotImplementedError

            next_state = self.hidden_projection(next_state_hidden)
            logits = [next_state]
            indexs = []

            if self.with_copy:
                pv_state_weight = self.embed_copy_attn.forward(next_state_hidden,
                                                               pv_state_emb,
                                                               mask=None,
                                                               not_softmax=True,
                                                               return_weight_only=True)
                logits.append(pv_state_weight)

                pv_r_u_know_index = self.lg_interpreter.glo2loc(pv_r_u)
                pv_r_u_mask = (pv_r_u_know_index == 0)
                pv_r_u_weight = self.hidden_copy_attn.forward(next_state_hidden,
                                                              pv_r_u_enc,
                                                              mask=pv_r_u_mask,
                                                              not_softmax=True,
                                                              return_weight_only=True)
                logits.append(pv_r_u_weight)
                indexs.append(pv_r_u_know_index)

                logits = torch.cat(logits, -1)
                indexs = torch.cat(indexs, -1).unsqueeze(1)

            probs = self.word_softmax(logits)

            if self.with_copy:
                gen_probs = probs[:, :, :self.know_vocab_size]

                pv_state_copy_probs = probs[:, :, self.know_vocab_size: self.know_vocab_size + DO.state_num]
                pv_state_copy_probs = torch.bmm(pv_state_copy_probs, pv_state)

                copy_probs = probs[:, :, self.know_vocab_size + DO.state_num:]
                copy_probs_placeholder = torch.zeros(batch_size, 1, self.know_vocab_size, device=TO.device)
                copy_probs = copy_probs_placeholder.scatter_add(2, indexs, copy_probs)

                probs = gen_probs + pv_state_copy_probs + copy_probs

            states.append(probs)

            if self.training and TO.auto_regressive and (gth_state is not None) and supervised:
                gth_step_input = gth_state[:, i:i + 1]  # B, 1
                gth_step_input = one_hot_scatter(gth_step_input, self.know_vocab_size, dtype=torch.float)
                step_input_embed = self.know_prob_embed(gth_step_input)
            else:
                if TO.auto_regressive:
                    if self.training:
                        gumbel_probs = self.state_gumbel_softmax_sampling(probs)  # gumbel-softmax
                    else:
                        max_indices = probs.argmax(-1)  # B, S
                        gumbel_probs = one_hot_scatter(max_indices, probs.size(2), dtype=torch.float)

                    step_input_embed = self.know_prob_embed(gumbel_probs)
                    gumbel_states.append(gumbel_probs)
                else:
                    step_input_embed = self.know_prob_embed(probs)  # B, 1, E

        states = torch.cat(states, dim=1)
        if len(gumbel_states) == 0:
            return states, None
        gumbel_states = torch.cat(gumbel_states, dim=1)
        return states, gumbel_states

    def know_prob_embed(self, state_prob):
        B, S, K = state_prob.shape
        know_embedding = self.embedder(self.know2word_tensor)
        state_embed = torch.bmm(state_prob.reshape(B * S, 1, K),
                                know_embedding.unsqueeze(0).expand(B * S, K, self.embed_dim))
        state_embed = state_embed.reshape(B, S, self.embed_dim)
        return state_embed
