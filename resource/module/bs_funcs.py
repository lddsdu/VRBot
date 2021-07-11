# -*- coding: utf-8 -*-

import heapq


def expand_if_not_none(tensor, dim, beam_width):
    if tensor is None:
        return None
    tensor_shape = list(tensor.shape)
    tensor = tensor.unsqueeze(dim + 1)
    expand_dims = [-1] * (len(tensor_shape) + 1)
    expand_dims[dim + 1] = beam_width
    tensor = tensor.expand(*expand_dims)
    tensor_shape[dim] = tensor_shape[dim] * beam_width
    tensor = tensor.reshape(*tensor_shape)
    return tensor.contiguous()


def repeat_if_not_none(tensor, dim, beam_width):
    if tensor is None:
        return None
    tensor_shape = list(tensor.shape)
    tensor = tensor.unsqueeze(dim + 1)
    expand_dims = [1] * (len(tensor_shape) + 1)
    expand_dims[dim + 1] = beam_width
    tensor = tensor.repeat(*expand_dims)
    tensor_shape[dim] = tensor_shape[dim] * beam_width
    tensor = tensor.reshape(*tensor_shape)
    return tensor


class Branch:
    def __init__(self, score, tensor, length, alpha=1.0, log_act=True):
        self.score = Branch.normal_score(score, length, alpha, log_act)
        self.tensor = tensor

    def __lt__(self, other):
        return self.score <= other.score

    def __eq__(self, other):
        return self.score == other.score

    def __gt__(self, other):
        return self.score >= other.score

    @staticmethod
    def normal_score(score, length, alpha=1.0, log_act=True):
        assert alpha >= 0.0, "alpha should >= 0.0"
        assert alpha <= 1.0, "alpha should <= 1.0"

        if log_act:
            score = score / (length ** alpha)
        else:
            score = score ** (1 / (length ** alpha))

        return score

    def get_tensor(self):
        return self.tensor


class MatureBucket:
    def __init__(self, bucket_size):
        self.bucket_size = bucket_size
        self.bucket = []

    def push(self, item: Branch):
        if len(self.bucket) < self.bucket_size:
            heapq.heappush(self.bucket, item)
        else:
            if item.score > self.bucket[0].score:
                heapq.heappushpop(self.bucket, item)

    def get_max(self):
        self.bucket = sorted(self.bucket, reverse=True)
        return self.bucket[0].get_tensor()
