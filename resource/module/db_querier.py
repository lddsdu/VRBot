# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from resource.option.vrbot_option import VRBotOption as VO
from resource.option.dataset_option import DatasetOption as DO


class DBQuerier(nn.Module):
    def __init__(self, knowledge_np, knowledge_len_np, device):
        super(DBQuerier, self).__init__()
        self.knowledge_np = knowledge_np
        self.knowledge_len_np = knowledge_len_np
        self.device = device

    def forward(self, query_index):
        B, S = query_index.shape
        query_index = query_index.cpu().detach().numpy()

        batch_tensors = []
        batch_len_tensors = []

        for b in range(B):
            batch_tensor = []
            batch_len_tensor = []

            for s in range(S):
                batch_tensor.append(self.knowledge_np[int(query_index[b][s])])
                batch_len_tensor.append(self.knowledge_len_np[int(query_index[b][s])])

            batch_tensor = np.concatenate(batch_tensor, axis=0)
            batch_len_tensor = np.concatenate(batch_len_tensor, axis=0)

            batch_tensor = batch_tensor[:S * VO.triple_num_per_graph, :]
            batch_len_tensor = batch_len_tensor[:S * VO.triple_num_per_graph]

            pad_num = S * VO.triple_num_per_graph - (batch_tensor.shape[0])
            if pad_num > 0:
                pad_batch_tensor = [[DO.PreventWord.PAD_ID] * DO.triple_len] * pad_num
                pad_batch_tensor = np.asarray(pad_batch_tensor)
                batch_tensor = np.concatenate([batch_tensor, pad_batch_tensor], axis=0)

            pad_len_num = S * VO.triple_num_per_graph - (len(batch_len_tensor))
            if pad_len_num > 0:
                pad_batch_len_tensor = [0] * pad_len_num
                pad_batch_len_tensor = np.asarray(pad_batch_len_tensor)
                batch_len_tensor = np.concatenate([batch_len_tensor, pad_batch_len_tensor], axis=0)

            batch_tensors.append(batch_tensor)
            batch_len_tensors.append(batch_len_tensor)

        batch_tensors = torch.tensor(batch_tensors, dtype=torch.long, device=self.device)
        batch_len_tensors = torch.tensor(batch_len_tensors, dtype=torch.long, device=self.device)

        return batch_tensors, batch_len_tensors
