# -*- coding: utf-8 -*-
# @Time    : 2020-03-24 10:55
# @Author  : lddsdu
# @File    : eval_rouge.py

from rouge import Rouge
from tqdm import tqdm


class RougeEval:
    def __init__(self):
        self.rouge = Rouge()

    def rouge_score(self, hypothesis, targets, ret_matrix=False):
        rouge_hyps = []
        rouge_tgts = []
        wrong_indices = []
        r1s = []
        r2s = []

        pbar = tqdm(enumerate(zip(hypothesis, targets)))
        for idx, (hyp, tgt) in pbar:
            pbar.set_description("test rouge case-{}".format(idx))
            try:
                single_rouge = self.rouge.get_scores(hyp or "<pad>", tgt)  # test it this pair is correct.
                r1s.append(single_rouge[0]["rouge-1"]["r"])
                r2s.append(single_rouge[0]["rouge-2"]["r"])
                rouge_hyps.append(hyp)
                rouge_tgts.append(tgt)
            except (Exception,) as e:
                wrong_indices.append(idx)
        print("rouge wrong num : {}".format(len(wrong_indices)))
        pad_rouge_hyps = [x or "<pad>" for x in rouge_hyps]
        rs = self.rouge.get_scores(pad_rouge_hyps, rouge_tgts, avg=True, ignore_empty=True)
        rouge_1 = rs["rouge-1"]["r"]
        rouge_2 = rs["rouge-2"]["r"]

        if not ret_matrix:
            return [rouge_1, rouge_2]
        else:
            return [rouge_1, rouge_2, r1s, r2s]

