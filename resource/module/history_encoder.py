# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class HistoryEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_units, bidirectional=True, dropout=0.2, embedder=None):
        super(HistoryEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.embedder = embedder
        self.num_layers = 1
        self.rnn_cell = nn.GRU(input_size=self.embed_dim,
                               hidden_size=self.hidden_units // 2,
                               num_layers=self.num_layers,
                               batch_first=True,
                               bidirectional=self.bidirectional,
                               dropout=self.dropout)

    def forward(self, pv_r_u, pv_r_u_len=None, hidden=None):
        if self.embedder is not None:
            pv_r_u = self.embedder(pv_r_u)

        if pv_r_u_len is not None:
            input_len = pv_r_u.size(1)
            values, indices = pv_r_u_len.sort(descending=True)
            batch_size = values.size(0)
            non_zero_batch_size = (values > 0).long().sum().item()
            non_zero_values = values[:non_zero_batch_size]
            non_zero_indices = indices[:non_zero_batch_size]

            if non_zero_batch_size <= 0:
                device = pv_r_u.device
                return torch.zeros(batch_size, input_len, self.hidden_units, dtype=torch.float, device=device), \
                       torch.zeros(1, batch_size, self.hidden_units, dtype=torch.float, device=device)

            # B', T, H
            rnn_inputs = pv_r_u.index_select(0, non_zero_indices)
            rnn_inputs = pack_padded_sequence(rnn_inputs, non_zero_values.tolist(), batch_first=True)
            if hidden is not None:
                # 1, B', H
                hidden = hidden.index_select(1, non_zero_indices)

        # FORWARD
        outputs, hidden = self.rnn_cell(rnn_inputs, hidden)

        if self.bidirectional:
            hidden = self._bridge_bidirectional_hidden(hidden)  # 1', B', H

        if pv_r_u_len is not None:
            rnn_outputs = pad_packed_sequence(outputs, batch_first=True)[0]  # B', T', H
            zero_batch_size = batch_size - non_zero_batch_size
            ro_device = rnn_outputs.device
            output_len = rnn_outputs.size(1)

            # PAD BATCH
            if zero_batch_size > 0:
                rnn_outputs = torch.cat(
                    [rnn_outputs, torch.zeros(zero_batch_size, output_len, self.hidden_units, device=ro_device)], dim=0)

                hidden = torch.cat([hidden,
                                    torch.zeros(1, zero_batch_size, self.hidden_units, device=ro_device)], dim=1)

            # PAD LENGTH
            if output_len < input_len:
                # PAD LENGTH
                # [B, T', H] + [B, (T - T'), H] ==> [B, T, H]
                rnn_outputs = torch.cat(
                    [rnn_outputs, torch.zeros(batch_size, input_len - output_len, self.hidden_units, device=ro_device)],
                    dim=1)

            _, inv_indices = indices.sort()
            rnn_outputs = rnn_outputs.index_select(0, inv_indices)
            hidden = hidden.index_select(1, inv_indices)
        else:
            rnn_outputs = outputs

        return rnn_outputs, hidden

    def _bridge_bidirectional_hidden(self, hidden):
        num_directions = 2 if self.bidirectional else 1
        batch_size = hidden.size(1)
        rnn_hidden_size = hidden.size(2)

        hidden = hidden.view(self.num_layers, num_directions, batch_size, rnn_hidden_size).transpose(1, 2).contiguous()
        hidden = hidden.view(self.num_layers, batch_size, -1)

        return hidden
