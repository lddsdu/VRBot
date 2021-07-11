# -*- coding: utf-8 -*-


import json
import pandas as pd
import numpy as np
from resource.input.vocab import Vocab
from resource.option.dataset_option import DatasetOption
from resource.option.train_option import TrainOption as TO


class GraphLoader:
    def __init__(self, alias2scientific_filename, entity2type_filename, joint_graph_filename, know_vocab: Vocab):
        self.alias2scientific_filename = alias2scientific_filename
        self.entity2type_filename = entity2type_filename
        self.joint_graph_filename = joint_graph_filename
        self.know_vocab = know_vocab

        self.alias2scientific = json.load(open(self.alias2scientific_filename))
        self.entity2type = json.load(open(self.entity2type_filename))
        self.joint_graph = json.load(open(self.joint_graph_filename))["graph"]

        scientific2id = dict(zip([DatasetOption.PreventWord.PAD] + list(self.entity2type.keys()),
                                 list(range(len(self.entity2type) + 1))))

        # alias_index2scientific_index \in \mathcal{R}^{K_a}, np.long
        alias_index2scientific_index = []
        for know_word_index, know_word in enumerate(know_vocab.word_list):
            tmp = scientific2id.get(know_word if TO.task == "meddg" else self.alias2scientific.get(know_word, know_word))
            alias_index2scientific_index.append(tmp)
        self.alias_index2scientific_index = np.asarray(alias_index2scientific_index)

        # scientific_index2alias_index \in \mathcal{R}^{K_s \times K_a}, np.float
        scientific_index2alias_index = []
        for scientific, scientific_id in sorted(scientific2id.items(), key=lambda x: x[1]):
            signal = np.asarray(self.alias_index2scientific_index == scientific_id, dtype=np.float)
            scientific_index2alias_index.append(signal / (signal.sum() + 1e-24))
        self.scientific_index2alias_index = np.asarray(scientific_index2alias_index)

        # entity2type_index \in \mathcal{R}^{K_s}, np.float
        entity2type_list = [self.entity2type.get(x[0], "None") for x in sorted(scientific2id.items(), key=lambda x: x[1])]
        self.entity_type2type_index = {"药物": 0, "疾病": 1, "诊疗": 2, "症状": 3, "None": 4}
        self.scientific_entity2type_index = [self.entity_type2type_index[entity_type] for entity_type in
                                             entity2type_list]

        # type_matrix \in \mathcal{R}^{K_a \times T_knowledge_word_type}
        type_matrix = []
        for know_word_index, know_word in enumerate(know_vocab.word_list):
            type_str = self.entity2type.get(self.alias2scientific.get(know_word, know_word), "None")
            tmp = [0, 0, 0, 0]
            type_index = self.entity_type2type_index[type_str]
            if type_index < 4:
                tmp[type_index] = 1.0
            type_matrix.append(tmp)
        self.type_matrix = np.asarray(type_matrix, dtype=np.float)

        # relation2type
        self.relation_type2type_index = {"contains_symptom": 0, "symptom_of": 1,
                                         "caused_by": 0, "lead_to": 1,
                                         "to_treat": 2, "treated_by": 3,
                                         "checked_for": 2, "check_item": 3}

        # adjacent_matrix \in \mathcal{R}^{4 \times K_s \times K_s}
        self.adjacent_matrix = np.zeros(shape=(4, len(scientific2id), len(scientific2id)), dtype=np.long)
        for head, relation, tail in self.joint_graph:
            head_id, tail_id = scientific2id[head], scientific2id[tail]
            if relation not in self.relation_type2type_index:
                continue
            relation_id = self.relation_type2type_index[relation]
            self.adjacent_matrix[relation_id, head_id, tail_id] = 1

        print("GraphLoader load finished")


if __name__ == '__main__':
    vocab_df = pd.read_csv(DatasetOption.vocab_csv_filename.format(dataset="chunyu"))
    know_vocab_items = [DatasetOption.PreventWord.PAD] + [str(x) for x in
                                                          list(vocab_df[vocab_df["Is_know"] > 0]["Word"])]
    know_vocab = Vocab(know_vocab_items, DatasetOption.know_vocab_size)

    gl = GraphLoader(DatasetOption.alias2scientific_filename,
                     DatasetOption.entity2type_filename,
                     DatasetOption.joint_graph_filename,
                     know_vocab)
