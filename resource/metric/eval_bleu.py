# -*- coding: utf-8 -*-

import os
from nlgeval import NLGEval

os.environ["NLGEVAL_DATA"] = "/home/lddsdu/hard_disk/nlgeval"
all_metrics = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4",
               "ROUGE_L", "CIDEr", "SkipThoughtCS",
               "EmbeddingAverageCosineSimilairty",
               "VectorExtremaCosineSimilarity",
               "GreedyMatchingScore"]

bleu_nlgeval = NLGEval(metrics_to_omit=["METEOR", "CIDEr", "ROUGE_L", "SkipThoughtCS",
                                        "EmbeddingAverageCosineSimilairty",
                                        "VectorExtremaCosineSimilarity",
                                        "GreedyMatchingScore"])


def eval_bleu(ref_list, hyp_list, do_print=False):
    scores = bleu_nlgeval.compute_metrics(ref_list, hyp_list)

    if do_print:
        for key, value in scores.items():
            print("{} : {}".format(key, value))

    scores = [scores[x] for x in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]]
    return scores
