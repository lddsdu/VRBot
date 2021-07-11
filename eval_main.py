# -*- coding: utf-8 -*-
# @Time    : 2020-03-15 22:22
# @Author  : lddsdu
# @File    : eval_main.py

import json
import pandas
import argparse
import numpy as np
import scipy.stats as stats
from nlgeval import NLGEval
from collections import OrderedDict
from collections import Counter
from tqdm import tqdm
from resource.util.eval_rouge import RougeEval
from resource.util.eval_distinct import DistinctEval


def t_test(a_score_dict: OrderedDict, b_score_dict: OrderedDict):
    metric_names = list(a_score_dict.keys())
    efficient_metric_names = [x for x in metric_names if isinstance(a_score_dict[x], list)]
    a_scores = [a_score_dict[name] for name in efficient_metric_names]
    b_scores = [b_score_dict[name] for name in efficient_metric_names]
    p_values = [float(stats.ttest_ind(a, b).pvalue) for (a, b) in tqdm(zip(a_scores, b_scores))]
    name2p = dict(zip(efficient_metric_names, p_values))
    return dict([(name, name2p[name]) if name in name2p else (name, float("nan")) for name in metric_names])


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ef", "--eval_filename", type=str, required=True,
                        help="the first file which wanted eval")
    parser.add_argument("-ef2", "--eval_filename2", type=str,
                        help="the second file which wanted eval")
    parser.add_argument("-vf", "--vocab_filename", type=str)
    parser.add_argument("-af", "--alias2scientific_filename", type=str)
    args = parser.parse_args()
    return args


def cosine_sim(a, b, eps=1e-24):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similarity = np.dot(a, b.T) / (a_norm * b_norm + eps)  # cosine similarity
    return similarity


def eval(eval_filename, vocab_filename, alias2scientific_filename):
    def remove_stopwords(sent, stop_word_set):
        items = sent.split()
        items = [ite for ite in items if ite not in stop_word_set]
        return " ".join(items)

    with open("data/stopwords.txt") as f:
        stopwords = f.read().strip().split()
        stopwords = set(stopwords)

    bleu_nlgeval = NLGEval(
        metrics_to_omit=["METEOR", "CIDEr", "ROUGE_L", "SkipThoughtCS", "EmbeddingAverageCosineSimilairty",
                         "VectorExtremaCosineSimilarity", "GreedyMatchingScore"])
    rouge_eval = RougeEval()
    disease2x = pandas.read_csv(vocab_filename)
    disease2x = disease2x[disease2x["Is_know"] > 0]
    disease2x = dict(zip(list(disease2x["Word"]), list(disease2x["Is_know"])))
    distinct_eval = DistinctEval(grams=[1, 2])

    with open(eval_filename) as f:
        sessions = json.load(f)

    gths = [[episode["gth"] for episode in session["session"]] for session in sessions]
    hyps = [[episode["hyp"] for episode in session["session"]] for session in sessions]
    entity_gths = [[" ".join([i for i in x.split(" ") if i in disease2x]) for x in y] for y in gths]
    entity_hyps = [[" ".join([i for i in x.split(" ") if i in disease2x]) for x in y] for y in hyps]

    def flat(lists):
        tmp = []
        for items in lists:
            tmp += items
        return tmp

    gths = flat(gths)
    hyps = flat(hyps)
    entity_gths = flat(entity_gths)
    entity_hyps = flat(entity_hyps)

    gths = [remove_stopwords(gth, stopwords) for gth in gths]
    hyps = [remove_stopwords(hyp, stopwords) for hyp in hyps]

    ret_metrics = OrderedDict()
    ret_metric = OrderedDict()

    bleu_score_matrix = [bleu_nlgeval.compute_individual_metrics([gth], hyp) for gth, hyp in zip(gths, hyps)]
    b2s = [b["Bleu_2"] for b in bleu_score_matrix]
    ret_metrics["B@2"] = b2s
    bleu_score = bleu_nlgeval.compute_metrics([gths], hyps)
    b2 = bleu_score["Bleu_2"]
    ret_metric["B@2"] = b2
    rouge1, rouge2, r1s, r2s = rouge_eval.rouge_score(hyps, gths, ret_matrix=True)
    ret_metrics["R@2"] = r2s
    ret_metric["R@2"] = rouge2
    dist_scores = distinct_eval.distinct_score(hyps)
    ret_metric["D@1"] = dist_scores[0]
    ret_metric["D@2"] = dist_scores[1]
    ret_metrics["D@1"] = float("nan")
    ret_metrics["D@2"] = float("nan")
    eps = 1e-24

    def compute_f1(p, r):
        return 2 * p * r / (p + r + eps)

    overlapped_entity = [[i for i in x.split() if i in y.split()] for x, y in zip(entity_hyps, entity_gths)]
    overlapped_entity = [list(set(x)) for x in overlapped_entity]
    hyp_entity = [set(y.split()) for y in entity_hyps]
    gth_entity = [set(y.split()) for y in entity_gths]
    entity2prf = OrderedDict()
    for oe, he, ge in zip(overlapped_entity, hyp_entity, gth_entity):
        for e in oe:
            if e not in entity2prf:
                entity2prf[e] = {"FN": 0, "FP": 0, "TP": 0}
            entity2prf[e]["TP"] += 1

        for e in he:
            if e not in entity2prf:
                entity2prf[e] = {"FN": 0, "FP": 0, "TP": 0}
            if e not in oe:
                entity2prf[e]["FP"] += 1

        for e in ge:
            if e not in entity2prf:
                entity2prf[e] = {"FN": 0, "FP": 0, "TP": 0}
            if e not in oe:
                entity2prf[e]["FN"] += 1

    counter = Counter()
    for gth in gth_entity:
        counter.update(gth)
    need_entity_ind = [x[0] for x in counter.most_common() if x[1] > 5]
    print("len(need_entity_ind) = {}".format(len(need_entity_ind)))
    ret_metrics["ma-P"] = [entity2prf[e]["TP"] / (entity2prf[e]["TP"] + entity2prf[e]["FP"] + eps) for e in
                           need_entity_ind]
    ret_metrics["ma-R"] = [entity2prf[e]["TP"] / (entity2prf[e]["TP"] + entity2prf[e]["FN"] + eps) for e in
                           need_entity_ind]
    ret_metrics["ma-F1"] = [compute_f1(p, r) for (p, r) in zip(ret_metrics["ma-P"], ret_metrics["ma-R"])]
    ret_metric["ma-P"] = float(np.mean(ret_metrics["ma-P"]))
    ret_metric["ma-R"] = float(np.mean(ret_metrics["ma-R"]))
    ret_metric["ma-F1"] = compute_f1(ret_metric["ma-P"], ret_metric["ma-R"])
    mi_precision = [len(x) / (len(y) + 1e-14)
                    for x, y in zip(overlapped_entity, [set(y.split()) for y in entity_hyps])]
    mi_recall = [len(x) / (len(y) + 1e-14)
                 for x, y in zip(overlapped_entity, [set(y.split()) for y in entity_gths])]
    gth_n = [len(set(ws.split())) for ws in entity_gths]
    hyp_n = [len(set(ws.split())) for ws in entity_hyps]
    ret_metric["mi-P"] = np.sum([p * w for (p, w) in zip(mi_precision, hyp_n)]) / np.sum(hyp_n)
    ret_metric["mi-R"] = np.sum([r * w for (r, w) in zip(mi_recall, gth_n)]) / np.sum(gth_n)
    ret_metric["mi-F1"] = compute_f1(ret_metric["mi-P"], ret_metric["mi-R"])
    ret_metrics["mi-P"] = mi_precision
    ret_metrics["mi-R"] = mi_recall
    ret_metrics["mi-F1"] = [compute_f1(p, r) for (p, r) in zip(mi_precision, mi_recall)]
    with open("data/word2embedding.txt") as f:
        content = f.read().strip()
    single_word2embedding = {}
    for line in content.split("\n"):
        item = line.split()
        word = item[0]
        embedding = np.asarray([float(x) for x in item[1:]])
        single_word2embedding[word] = embedding
    alias2scientific = json.load(open(alias2scientific_filename))
    padding_embed = np.zeros(768)

    hyp_emb_avg = [
        np.asarray(
            [np.asarray([single_word2embedding.get(w, padding_embed) for w in alias2scientific.get(e, e)]).mean(0) for e
             in entity_hyp.split()]).mean(0)
        if len(entity_hyp.split()) > 0 else padding_embed for entity_hyp in entity_hyps]
    gth_emb_avg = [
        np.asarray(
            [np.asarray([single_word2embedding.get(w, padding_embed) for w in alias2scientific.get(e, e)]).mean(0) for e
             in entity_gth.split()]).mean(0)
        if len(entity_gth.split()) > 0 else padding_embed for entity_gth in entity_gths]
    eas = [cosine_sim(h, g) for h, g in zip(hyp_emb_avg, gth_emb_avg)]
    ea = float(np.mean(eas))
    ret_metrics["EA"] = eas
    ret_metric["EA"] = ea

    hyp_emb_means = [
        [np.asarray([single_word2embedding.get(w, padding_embed) for w in alias2scientific.get(e, e)]).mean(0) for e in
         entity_hyp.split()]
        if len(entity_hyp.split()) > 0 else [padding_embed] for entity_hyp in entity_hyps]
    gth_emb_means = [
        [np.asarray([single_word2embedding.get(w, padding_embed) for w in alias2scientific.get(e, e)]).mean(0) for e in
         entity_gth.split()]
        if len(entity_gth.split()) > 0 else [padding_embed] for entity_gth in entity_gths]

    def eval_embed_greedy(a, b):
        scores = []

        for j in b:
            score = []
            for i in a:
                s = cosine_sim(i, j)
                score.append(s)
            scores.append(score)

        if len(b) == 1 and b[0].sum() == 0.0:
            return None
        else:
            scores = np.asarray(scores)
            score1 = scores.max(0).mean()
            score2 = scores.max(1).mean()
            return (float(score1) + float(score2)) / 2.0

    eg_scores = [x for x in [eval_embed_greedy(a, b) for (a, b) in zip(hyp_emb_means, gth_emb_means)] if x is not None]
    eg_score = np.asarray(eg_scores).mean()
    ret_metrics["EG"] = eg_scores
    ret_metric["EG"] = eg_score

    return ret_metrics, ret_metric


def main():
    cfg = config()

    scores1, score1 = eval(cfg.eval_filename,
                           cfg.vocab_filename,
                           cfg.alias2scientific_filename)

    scores2, score2, p_values = None, None, None
    if cfg.eval_filename2 is not None:
        scores2, score2 = eval(cfg.eval_filename2,
                               cfg.vocab_filename,
                               cfg.alias2scientific_filename)

        p_values = t_test(scores1, scores2)

    metric_name = list(scores1.keys())
    print("\t".join(["Me"] + metric_name))
    print("\t".join(["S1"] + ["{:.2f}".format(score1[k] * 100) for k in metric_name]))

    if cfg.eval_filename2 is not None:
        print("\t".join(["S2"] + ["{:.2f}".format(score2[k] * 100) for k in metric_name]))
        print("\t".join(["PV"] + ["{:.2f}".format(p_values[k]) for k in metric_name]))


if __name__ == '__main__':
    main()
