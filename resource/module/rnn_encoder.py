# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class RNNEncoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_units,
                 rnn_hidden_units,
                 num_layers,
                 bidirectional=True,
                 dropout=0.2,
                 embedder=None):

        super(RNNEncoder, self).__init__()
        assert rnn_hidden_units * (2 if bidirectional else 1) == hidden_units

        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.rnn_hidden_units = rnn_hidden_units
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout if (self.num_layers > 1) else 0.0
        self.embedder = embedder

        self.rnn_cell = nn.GRU(input_size=self.embed_dim,
                               hidden_size=self.rnn_hidden_units,
                               num_layers=self.num_layers,
                               batch_first=True,
                               bidirectional=self.bidirectional,
                               dropout=self.dropout)

    def forward(self,
                inputs,
                lengths=None,
                hidden=None,
                do_embedding=True):
        rnn_inputs = inputs
        if self.embedder is not None and do_embedding:
            rnn_inputs = self.embedder(inputs)

        if lengths is not None:
            T = inputs.size(1)
            values, indices = lengths.sort(descending=True)
            batch_size = values.size(0)
            non_zero_batch_size = (values > 0).long().sum().item()
            non_zero_values = values[:non_zero_batch_size]
            non_zero_indices = indices[:non_zero_batch_size]

            if non_zero_batch_size <= 0:
                device = inputs.device
                return torch.zeros(batch_size, T, self.hidden_units, dtype=torch.float, device=device), \
                       torch.zeros(1, batch_size, self.hidden_units, dtype=torch.float, device=device)

            rnn_inputs = rnn_inputs.index_select(0, non_zero_indices)
            rnn_inputs = pack_padded_sequence(rnn_inputs, non_zero_values.tolist(), batch_first=True)
            if hidden is not None:
                hidden = hidden.index_select(1, non_zero_indices)

        # FORWARD
        outputs, hidden = self.rnn_cell(rnn_inputs, hidden)
        if self.bidirectional:
            hidden = self._bridge_bidirectional_hidden(hidden)  # 1', B', H

        if lengths is not None:
            rnn_outputs = pad_packed_sequence(outputs, batch_first=True)[0]  # B', T', H
            zero_batch_size = batch_size - non_zero_batch_size
            ro_device = rnn_outputs.device
            T_ = rnn_outputs.size(1)

            # PAD BATCH
            if zero_batch_size > 0:
                # [B', T', H] + [(B - B'), T', H] ==> [B, T', H]
                rnn_outputs = torch.cat(
                    [rnn_outputs, torch.zeros(zero_batch_size, T_, self.hidden_units, device=ro_device)], dim=0)
                # [1, B', H] + [1, (B - B'), H] ==> [1, B, H]
                hidden = torch.cat([hidden,
                                    torch.zeros(1, zero_batch_size, self.hidden_units, device=ro_device)], dim=1)

            # PAD LENGTH
            if T_ < T:
                # [B, T', H] + [B, (T - T'), H] ==> [B, T, H]
                rnn_outputs = torch.cat(
                    [rnn_outputs, torch.zeros(batch_size, T - T_, self.hidden_units, device=ro_device)], dim=1)

            _, inv_indices = indices.sort()
            rnn_outputs = rnn_outputs.index_select(0, inv_indices)
            hidden = hidden.index_select(1, inv_indices)
        else:
            rnn_outputs = outputs

        # rnn_outputs: B, T, hidden_size
        # hidden: 1, B, hidden_size
        return rnn_outputs, hidden

    def _bridge_bidirectional_hidden(self, hidden):
        num_directions = 2 if self.bidirectional else 1
        batch_size = hidden.size(1)
        rnn_hidden_size = hidden.size(2)

        hidden = hidden.view(self.num_layers, num_directions, batch_size, rnn_hidden_size).transpose(1, 2).contiguous()
        hidden = hidden.view(self.num_layers, batch_size, -1)

        return hidden


class HierarchicalEncoder(nn.Module):
    def __init__(self,
                 encoder_hidden_size,
                 decoder_hidden_size):
        super(HierarchicalEncoder, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.rnn_cell = nn.GRU(input_size=self.encoder_hidden_size,
                               hidden_size=self.decoder_hidden_size,
                               batch_first=True)

    def forward(self, sentence_vector, hidden=None):
        outputs, last_hidden = self.rnn_cell.forward(sentence_vector, hidden)
        return outputs, last_hidden

