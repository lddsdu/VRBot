# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
from resource.option.dataset_option import DatasetOption as DO
from resource.option.vrbot_option import VRBotOption as VO
from resource.module.base_attention import Attention
from resource.module.util_layers import KMaxPooling
from resource.util.misc import reverse_sequence_mask
from resource.option.train_option import TrainOption
from resource.model.vrbot_train_state import vrbot_train_stage


class RNNDecoder(nn.Module):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 target_max_len,
                 embedder=None,
                 num_layers=1,
                 dropout=0.0,
                 attention_mode="mlp",
                 max_pooling_k=10,
                 attn_history_sentence=True,
                 with_state_know=True,
                 with_action_know=True,
                 with_copy=True):
        super(RNNDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.target_max_len = target_max_len
        self.embedder = embedder
        self.max_pooling_k = max_pooling_k
        self.num_layers = num_layers
        self.dropout = dropout if (self.num_layers > 1) else 0.0

        self.attn_history_sentence = attn_history_sentence

        self.with_state_know = False
        self.with_action_know = False
        self.with_copy = with_copy and (with_state_know or with_action_know)

        self.rnn_input_size = self.embed_size  # r_{t-1} input
        self.rnn_input_size += (
                self.hidden_size + self.embed_size + self.embed_size)  # history attn, state attn, action attn
        self.out_input_size = self.hidden_size

        if self.attn_history_sentence:
            # SENTENCE & WORD LEVEL ATTENTION
            self.history_attn_word = Attention(query_size=self.hidden_size, mode=attention_mode)

        if self.with_state_know:
            self.state_know_attn = Attention(query_size=self.hidden_size, mode=attention_mode)
            self.rnn_input_size += self.hidden_size

        if self.with_action_know:
            self.action_know_attn = Attention(query_size=self.hidden_size, mode=attention_mode)
            self.rnn_input_size += self.hidden_size

        if self.with_copy:
            # copy from knowledge
            self.copy_attn = Attention(query_size=self.hidden_size, mode="dot")
            self.sparse_copy_attn = Attention(query_size=self.hidden_size,
                                              memory_size=self.embed_size + 1,
                                              hidden_size=self.hidden_size)
            self.hm_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
            # select the most relative items
            self.k_max_pooling = KMaxPooling(self.max_pooling_k)

        # DECODER CELL
        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout,
                          batch_first=True)

        self.init_s_attn = Attention(self.hidden_size, self.embed_size)
        self.init_a_attn = Attention(self.hidden_size, self.embed_size)
        self.s_linear = nn.Linear(self.embed_size, self.hidden_size)
        self.a_linear = nn.Linear(self.embed_size, self.hidden_size)

        if not self.with_copy:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.vocab_size),
                nn.LogSoftmax(dim=-1))
        else:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.vocab_size))

            self.softmax = nn.Softmax(dim=-1)

    def decode(self, inp, hidden, hs_vectors,
               state_gumbel_prob, state_gumbel_embed,
               action_gumbel_prob, action_gumbel_embed,
               history_lens=None, history_word_indices=None,
               mask_action_prob=False, mask_state_prob=False,
               state_detach=False):
        if self.embedder is not None:
            inp = self.embedder(inp)
        # B, 1, E
        rnn_input = inp

        if self.attn_history_sentence:
            word_mask = None
            B, L_sent, H = hs_vectors.shape
            if history_lens is not None:
                # B, L_sent
                word_mask = reverse_sequence_mask(history_lens, max_len=L_sent).reshape(B, -1)
            # B, 1, H
            c_his_word, _ = self.history_attn_word.forward(hidden.transpose(0, 1),
                                                           hs_vectors,
                                                           mask=word_mask)
            rnn_input = torch.cat([rnn_input, c_his_word], dim=-1)

        # attn state and action embed representation
        c_state = self.init_consult(hidden, state_gumbel_embed, self.init_s_attn)
        c_action = self.init_consult(hidden, action_gumbel_embed, self.init_a_attn)
        rnn_input = torch.cat([rnn_input, c_state, c_action], dim=-1)

        rnn_output, new_hidden = self.rnn(rnn_input, hidden)

        gene_output = self.output_layer(rnn_output)

        if not self.with_copy:
            return gene_output, new_hidden, None

        copy_query = new_hidden.permute(1, 0, 2)
        copy_logits = []
        copy_word_index = []

        # copy from post
        hs_logits = self.copy_attn.forward(copy_query, hs_vectors, not_softmax=True,
                                           return_weight_only=True)  # B, 1, L_sentence
        copy_logits.append(hs_logits)
        copy_word_index.append(history_word_indices.unsqueeze(1))

        s_tag = torch.zeros(list(state_gumbel_prob.shape)[:-1] + [1], device=state_gumbel_prob.device,
                            dtype=torch.float)
        a_tag = torch.ones(list(action_gumbel_prob.shape)[:-1] + [1], device=action_gumbel_prob.device,
                           dtype=torch.float)
        # COPY FROM PV_STATE & ACTION
        # B, 1, S
        state_copy_weight = self.sparse_copy(copy_query, torch.cat([state_gumbel_embed, s_tag], dim=-1))
        # B, 1, A
        action_copy_weight = self.sparse_copy(copy_query, torch.cat([action_gumbel_embed, a_tag], dim=-1))

        if mask_state_prob:
            state_copy_weight.masked_fill_(mask=torch.ones_like(state_copy_weight).bool(), value=-1e24)
        if mask_action_prob:
            action_copy_weight.masked_fill_(mask=torch.ones_like(action_copy_weight).bool(), value=-1e24)

        copy_logits.append(state_copy_weight)
        copy_logits.append(action_copy_weight)

        # B, V [+ (k * L_triple) [+ (k * L_triple)]] + S + A
        word_proba = self.softmax(torch.cat([gene_output] + copy_logits, dim=-1)).squeeze(1)
        # B, V
        gene_proba = word_proba[:, :self.vocab_size]

        state_gumbel_weight = word_proba[:, -(DO.state_num + DO.action_num):-DO.action_num].unsqueeze(1)  # B, 1, S
        action_gumbel_weight = word_proba[:, -DO.action_num:].unsqueeze(1)  # B, 1, A
        # B, V
        state_copy_prob = torch.bmm(state_gumbel_weight, state_gumbel_prob).squeeze(1)
        if state_detach:
            state_copy_prob = state_copy_prob.detach()

        action_copy_prob = torch.bmm(action_gumbel_weight, action_gumbel_prob).squeeze(1)

        l_s = vrbot_train_stage.s_copy_lambda
        l_a = vrbot_train_stage.a_copy_lambda if VO.train_stage == "action" else vrbot_train_stage.a_copy_lambda_mini
        gene_proba = gene_proba + l_s * state_copy_prob + l_a * action_copy_prob

        if len(copy_word_index) == 0:
            output = gene_proba
        else:
            if len(copy_word_index) > 1:
                copy_word_index = torch.cat(copy_word_index, dim=-1)
            else:
                copy_word_index = copy_word_index[0]
            # B, V
            B = word_proba.size(0)
            copy_proba = torch.zeros(B, self.vocab_size, device=word_proba.device)
            copy_proba = copy_proba.scatter_add(1, copy_word_index.squeeze(1),
                                                word_proba[:, self.vocab_size: -(DO.state_num + DO.action_num)])
            output = gene_proba + copy_proba

        output = torch.log(output.unsqueeze(1))
        return output, new_hidden

    def copy(self, query, memory_word_enc, memory_word_index):
        B, t_num, l_trip, H = memory_word_enc.shape
        flatten_memory_word_enc = memory_word_enc.reshape(B, t_num * l_trip, H)
        flatten_selected_word_index = memory_word_index.reshape(B, -1)
        flatten_selected_mask = flatten_selected_word_index <= DO.PreventWord.RESERVED_MAX_INDEX
        match_logits = self.copy_attn.forward(query, flatten_memory_word_enc, not_softmax=True)[1].squeeze(1)
        match_logits = match_logits.masked_fill(flatten_selected_mask, -1e24)
        return match_logits, flatten_selected_word_index

    def sparse_copy(self, query, word_enc):
        # B, 1, S
        copy_weight = self.sparse_copy_attn.forward(query, word_enc, not_softmax=True, return_weight_only=True)
        return copy_weight

    @staticmethod
    def init_consult(hidden, state_emb, attn, state_linear=None):
        state_attn_emb, _ = attn.forward(hidden.permute(1, 0, 2), state_emb)  # B, 1, E
        if state_linear is not None:
            state_attn_emb = state_linear(state_attn_emb)  # B, 1, H
            return state_attn_emb.permute(1, 0, 2)  # 1, B, H
        else:
            return state_attn_emb

    def forward(self, hidden, inputs, hs_vectors,
                state_gumbel_prob, state_gumbel_embed,
                action_gumbel_prob, action_gumbel_embed,
                history_lens=None, history_word_indices=None,
                mask_action_prob=False, mask_state_prob=False, state_detach=False):
        if self.training:
            assert inputs is not None, "In training stage, inputs should not be None"
            batch_size, s_max_len = inputs.shape
        else:
            batch_size = hidden.size(1)

        rnn_input = hidden.new_ones(batch_size, 1, dtype=torch.long) * DO.PreventWord.SOS_ID  # SOS

        # consult the state and action
        hidden = self.init_consult_state_action(action_gumbel_embed, hidden,
                                                mask_action_prob, state_gumbel_embed)

        if self.training:
            valid_lengths = s_max_len - (inputs == DO.PreventWord.EOS_ID).long().cumsum(1).sum(1)
            valid_lengths, indices = valid_lengths.sort(descending=True)
            lengths_tag = torch.arange(0, s_max_len, device=TrainOption.device).unsqueeze(0).expand(batch_size, -1)

            # max_len
            batch_num_valid = (valid_lengths.unsqueeze(1).expand(-1, s_max_len) > lengths_tag).long().sum(0)
            batch_num_valid = batch_num_valid.cpu().numpy().tolist()
            # B, max_len, vocab_size
            output_placeholder = torch.zeros(batch_size, self.target_max_len, self.vocab_size,
                                             dtype=torch.float, device=TrainOption.device)

            # input Tensor index_select
            hidden = index_select_if_not_none(hidden, indices, 1)
            inputs = index_select_if_not_none(inputs, indices, 0)
            hs_vectors = index_select_if_not_none(hs_vectors, indices, 0)

            state_gumbel_embed = index_select_if_not_none(state_gumbel_embed, indices, 0)
            state_gumbel_prob = index_select_if_not_none(state_gumbel_prob, indices, 0)
            action_gumbel_embed = index_select_if_not_none(action_gumbel_embed, indices, 0)
            action_gumbel_prob = index_select_if_not_none(action_gumbel_prob, indices, 0)
            history_word_indices = index_select_if_not_none(history_word_indices, indices, 0)
            history_lens = index_select_if_not_none(history_lens, indices, 0)

            for dec_step, vb in enumerate(batch_num_valid):
                if vb <= 0:
                    break

                if inputs is not None and self.training:  # teacher forcing in training stage
                    # B, 1
                    rnn_input = inputs[:, dec_step].unsqueeze(1)

                rnn_dec = self.decode(rnn_input[:vb, :],
                                      hidden[:, :vb, :],
                                      hs_vectors[:vb, ...] if hs_vectors is not None else None,
                                      state_gumbel_prob[:vb, ...] if state_gumbel_prob is not None else None,
                                      state_gumbel_embed[:vb, ...] if state_gumbel_embed is not None else None,
                                      action_gumbel_prob[:vb, ...] if action_gumbel_prob is not None else None,
                                      action_gumbel_embed[:vb, ...] if action_gumbel_embed is not None else None,
                                      history_lens[:vb] if history_lens is not None else None,
                                      history_word_indices[:vb] if history_word_indices is not None else None,
                                      mask_action_prob,
                                      mask_state_prob,
                                      state_detach)

                word_output, hidden = rnn_dec
                # B, max_len, vocab_size
                output_placeholder[:vb, dec_step: dec_step + 1, :] = word_output

                if not (inputs is not None and self.training):  # inference in testing or valid stage
                    rnn_input = word_output.argmax(dim=-1)

            _, rev_indices = indices.sort()
            output_placeholder = output_placeholder.index_select(0, rev_indices)
            return output_placeholder
        else:
            global_indices = torch.arange(0, batch_size, dtype=torch.long, device=TrainOption.device)

            output_placeholder = torch.zeros(batch_size,
                                             self.target_max_len,
                                             self.vocab_size,
                                             dtype=torch.float,
                                             device=TrainOption.device)

            # decode
            for i in range(self.target_max_len):
                word_output, hidden = self.decode(rnn_input,
                                                  hidden,
                                                  hs_vectors,
                                                  history_lens)
                # B, 1
                next_step_input = word_output.argmax(dim=-1)
                # B,
                continue_tag = (next_step_input.squeeze(1) != DO.PreventWord.EOS_ID).long()
                _, local_indices = continue_tag.sort(descending=True)
                output_placeholder[global_indices, i: i + 1, :] = word_output
                b_case = continue_tag.sum().item()
                if b_case <= 0:
                    break

                # B',
                local_indices = local_indices[:b_case]
                global_indices = global_indices.index_select(0, local_indices)
                rnn_input = next_step_input

                # input next time
                rnn_input = rnn_input.index_select(0, local_indices)
                hidden = hidden.index_select(1, local_indices)

                # history_info & memory
                hs_vectors = index_select_if_not_none(hs_vectors, local_indices, 0)
                state_know_key = index_select_if_not_none(state_know_key, local_indices, 0)
                state_know_value = index_select_if_not_none(state_know_value, local_indices, 0)
                state_know_word_enc = index_select_if_not_none(state_know_word_enc, local_indices, 0)
                state_know_word = index_select_if_not_none(state_know_word, local_indices, 0)
                state_know_len = index_select_if_not_none(state_know_len, local_indices, 0)
                action_know_key = index_select_if_not_none(action_know_key, local_indices, 0)
                action_know_value = index_select_if_not_none(action_know_value, local_indices, 0)
                action_know_word_enc = index_select_if_not_none(action_know_word_enc, local_indices, 0)
                action_know_word = index_select_if_not_none(action_know_word, local_indices, 0)
                action_know_len = index_select_if_not_none(action_know_len, local_indices, 0)
                history_lens = index_select_if_not_none(history_lens, local_indices, 0)

            return output_placeholder

    def init_consult_state_action(self, action_gumbel_embed, hidden, mask_action_prob, state_gumbel_embed):
        if mask_action_prob:
            s_emb = self.init_consult(hidden, state_gumbel_embed, self.init_s_attn, self.s_linear)
            hidden = hidden + s_emb
        else:
            s_emb = self.init_consult(hidden, state_gumbel_embed, self.init_s_attn, self.s_linear)
            a_emb = self.init_consult(hidden, action_gumbel_embed, self.init_a_attn, self.a_linear)
            hidden = hidden + (s_emb + a_emb) / 2.0
        return hidden


def index_select_if_not_none(tensor, indices, dim):
    """index select tensor, if tensor is not None"""
    if tensor is None:
        return None

    tensor = tensor.index_select(dim, indices)
    return tensor
