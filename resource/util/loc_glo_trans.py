# -*- coding: utf-8 -*-
# @Time    : 2020-06-27 20:45
# @Author  : lddsdu
# @File    : loc_glo_trans.py


import torch


def loc_idx2glo_idx(loc2glo_tensor, loc_idx):
    loc2glo_tensor = loc2glo_tensor.unsqueeze(0).expand(loc_idx.size(0), loc2glo_tensor.size(0))
    glo_index = torch.gather(loc2glo_tensor, 1, loc_idx)

    return glo_index


def glo_idx2loc_idx(glo2loc_tensor, glo_idx):
    glo2loc_tensor = glo2loc_tensor.unsqueeze(0).expand(glo_idx.size(0), glo2loc_tensor.size(0))
    loc_index = torch.gather(glo2loc_tensor, 1, glo_idx)

    return loc_index


class LocGloInterpreter:
    def __init__(self, loc2glo, glo2loc):
        self.loc2glo_ = loc2glo
        self.glo2loc_ = glo2loc

    def loc2glo(self, loc_idx):
        return loc_idx2glo_idx(self.loc2glo_, loc_idx)

    def glo2loc(self, glo_idx):
        return glo_idx2loc_idx(self.glo2loc_, glo_idx)
