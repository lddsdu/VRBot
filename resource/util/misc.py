# -*- coding: utf-8 -*-

import os
import time
import torch
import numpy as np
import argparse
import functools
from resource.option.dataset_option import DatasetOption as DO


def mkdir_if_necessary(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def clip_pad_sentence(sentence,
                      max_len,
                      sos=DO.PreventWord.SOS,
                      eos=DO.PreventWord.EOS,
                      pad=DO.PreventWord.PAD,
                      save_prefix=True,
                      pad_suffix=True,
                      return_length=False):
    if sos is not None:
        sentence = [sos] + sentence

    ml = max_len
    if eos is not None:
        ml = ml - 1
    if save_prefix:
        sentence = sentence[:ml]
    else:
        sentence = sentence[-ml:]
    if eos is not None:
        sentence = sentence + [eos]

    length = None
    if return_length:
        length = len(sentence)

    if pad_suffix:
        sentence += [pad] * (max_len - len(sentence))
    else:
        sentence = [pad] * (max_len - len(sentence)) + sentence

    if not return_length:
        return sentence
    return sentence, length


def sequence_mask(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long, device=lengths.device).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.lt(lengths.unsqueeze(-1))
    return mask


def reverse_sequence_mask(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long, device=lengths.device).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.ge(lengths.unsqueeze(-1))
    return mask


def max_lens(X):
    if not isinstance(X[0], list):
        return [len(X)]
    elif not isinstance(X[0][0], list):
        return [len(X), max(len(x) for x in X)]
    elif not isinstance(X[0][0][0], list):
        return [len(X), max(len(x) for x in X),
                max(len(x) for xs in X for x in xs)]
    else:
        raise ValueError(
            "Data list whose dim is greater than 3 is not supported!")


def list2tensor(X):
    size = max_lens(X)
    if len(size) == 1:
        tensor = torch.tensor(X)
        return tensor

    tensor = torch.zeros(size, dtype=torch.long)
    lengths = torch.zeros(size[:-1], dtype=torch.long)

    if len(size) == 2:
        for i, x in enumerate(X):
            l = len(x)
            tensor[i, :l] = torch.tensor(x)
            lengths[i] = l
    else:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                l = len(x)
                tensor[i, j, :l] = torch.tensor(x)
                lengths[i, j] = l

    return tensor, lengths


def one_hot_scatter(indice, num_classes, dtype=torch.long):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder


def one_hot_sign(indice, num_classes):
    one_hot_tag = one_hot_scatter(indice, num_classes + 1).long()
    sign = torch.ones_like(one_hot_tag) - torch.cumsum(one_hot_tag, 1)
    sign = sign[:, :-1]
    return sign


def one_hot(indice, num_classes):
    I = torch.eye(num_classes).to(indice.device)
    T = I[indice]
    return T


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def index_select(condidate_ids, needed_id):
    matching_tag = (condidate_ids == needed_id).long()
    select_batch_size = matching_tag.sum()
    if select_batch_size < 1:
        return None
    _, index = matching_tag.sort()
    select_index = index[- select_batch_size:]
    return select_index


def nested_index_select(origin_data, select_index):
    origin_data_shape = list(origin_data.shape)
    select_index_shape = list(select_index.shape)
    work_axes = len(select_index_shape) - 1
    grad_v = functools.reduce(lambda x, y: x * y, origin_data_shape[:work_axes])
    new_dim = select_index_shape[-1]
    grad = torch.arange(0, grad_v, dtype=torch.long, device=origin_data.device).unsqueeze(-1)
    grad = grad.expand(-1, new_dim)
    grad = grad.reshape(-1)
    grad = grad * origin_data_shape[work_axes]
    select_index = select_index.reshape(-1) + grad
    reshaped_data = origin_data.reshape(grad_v * origin_data_shape[work_axes], -1)
    selected_data = reshaped_data.index_select(0, select_index)
    origin_data_shape[work_axes] = new_dim
    selected_data = selected_data.reshape(origin_data_shape)
    return selected_data


def select_value(tensor_list, selected_index):
    return [h.index_select(0, selected_index) if h is not None else h for h in tensor_list]


def strftime():
    return time.strftime("%Y-%m-%d-%H-%M-%S")


def adjust_learning_rate(optimizer,
                         rate_decay,
                         mini_lr):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        param_group['lr'] = max(lr * rate_decay, mini_lr)


def partition_arg_topK(matrix, K, axis=0, select_high=True):
    if select_high:
        matrix = matrix * -1

    a_part = np.argpartition(matrix, K, axis=axis)

    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

