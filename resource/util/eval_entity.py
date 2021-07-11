# -*- coding: utf-8 -*-
# @Time    : 2020-03-04 17:30
# @Author  : lddsdu
# @File    : eval_entity.py

import json
import numpy as np


class EntityEval:
    def __init__(self, disease2x):
        super(EntityEval, self).__init__()
        if isinstance(disease2x, dict):
            self.disease2x = set(list(disease2x.keys()))
        elif isinstance(disease2x, list):
            self.disease2x = list(disease2x)
        elif isinstance(disease2x, set):
            self.disease2x = disease2x
        elif isinstance(disease2x, str):
            self.disease2x = self.load_diseases_from_file(diseases_filename=disease2x)
        else:
            raise NotImplementedError

    @staticmethod
    def load_diseases_from_file(diseases_filename):
        diseases = json.load(open(diseases_filename))
        disease_set = set(diseases)
        return disease_set

    def compute_entity_rate(self, hypothesis):
        if len(hypothesis) == 0:
            return 0.0

        hypothesis0 = []
        hypothesis1 = []
        for h in hypothesis:
            word_set = list(set(h.strip().split(" ")))
            hypothesis0 += word_set
            hypothesis1.append(word_set)

        hypothesis_len = len(hypothesis0)
        entities = []

        for word_set in hypothesis1:
            entities.append([x for x in word_set if x in self.disease2x])

        entities_rate = np.mean([len(x) for x in entities])
        tmp = []
        for e in entities:
            tmp += e
        entities = tmp
        entities_category_num = len(set(entities))

        return entities_rate, entities_category_num

    def compute_entity_rate_str(self, hypothesis):
        rate, cat_num = self.compute_entity_rate(hypothesis)
        return "{:.4f}\t{}".format(rate, cat_num)
