# -*- coding: utf-8 -*-
# @Time    : 2020-03-04 17:07
# @Author  : lddsdu
# @File    : eval_distinct.py


from tqdm import tqdm
from argparse import ArgumentParser


def line_fn_origin(line):
    return line


def get_line_fn_split(split_tag="<==>", index=1):
    def line_fn(line):
        return line.split(split_tag)[index]

    return line_fn


class DistinctEval:
    def __init__(self, grams=[1]):
        assert type(grams) == list, "grams should be list, but {}".format(type(grams))
        assert len(grams) > 0, "grams's length should be larger than 0"
        self.grams = grams

    def _get_n_grams(self, lines):
        n_gram_set_list = [set() for _ in range(len(self.grams))]
        n_gram_num = [0 for _ in range(len(self.grams))]

        for line in tqdm(lines):
            words = line.split()

            for gram_idx, n_gram in enumerate(self.grams):
                exist_n_gram_num = max(0, len(words) - n_gram + 1)
                n_gram_num[gram_idx] += exist_n_gram_num

                for i in range(0, exist_n_gram_num):
                    n_gram_set_list[gram_idx].add(" ".join(words[i: i + n_gram]))

        n_gram_set_size = [len(x) for x in n_gram_set_list]
        distinct_ns = [float(ele) / float(deno) for ele, deno in zip(n_gram_set_size, n_gram_num)]

        return distinct_ns

    def distinct_score(self, hypothesis):
        hypothesis = [h.strip().split(" ") for h in hypothesis]
        n_gram_set_list = [set() for _ in range(len(self.grams))]
        n_gram_num = [0 for _ in range(len(self.grams))]

        for words in tqdm(hypothesis):
            for gram_idx, n_gram in enumerate(self.grams):
                exist_n_gram_num = max(0, len(words) - n_gram + 1)
                n_gram_num[gram_idx] += exist_n_gram_num

                for i in range(0, exist_n_gram_num):
                    n_gram_set_list[gram_idx].add(" ".join(words[i: i + n_gram]))

        n_gram_set_size = [len(x) for x in n_gram_set_list]
        distinct_ns = [float(ele) / float(deno) for ele, deno in zip(n_gram_set_size, n_gram_num)]
        return distinct_ns

    def distinct_score2str(self, distinct_ns):
        return " ".join("Dist@{}: {:.6f}".format(n, d) for n, d in zip(self.grams, distinct_ns))
