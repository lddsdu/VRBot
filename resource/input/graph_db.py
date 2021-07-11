# -*- coding: utf-8 -*-

import json
import numpy as np
from resource.input.vocab import Vocab


class TripleLoader:
    def __init__(self, joint_graph_filename, vocab: Vocab):
        self.joint_graph_filename = joint_graph_filename
        self.vocab = vocab
        self.joint_graph = json.load(open(self.joint_graph_filename))["graph"]
        self.joint_graph = [triple for triple in self.joint_graph
                            if vocab.item_in(triple[0]) and vocab.item_in(triple[2])]
        edge_types = sorted(list(set(triple[1] for triple in self.joint_graph)))
        edge_types = ["self_loop", "session_connect"] + edge_types  # self_loop, session_connect, to_treat, ...
        self.edge_type2id = dict(zip(edge_types, list(range(len(edge_types)))))

    def load_triples(self):
        head_relation_tail = [(self.vocab.word2index(h), self.edge_type2id[r], self.vocab.word2index(t))
                              for (h, r, t) in self.joint_graph]
        head_relation_tail_np = np.asarray(head_relation_tail)

        head2index = {}
        tail2index = {}

        for idx, (h, _, t) in enumerate(head_relation_tail):
            if h not in head2index:
                head2index[h] = []
            head2index[h].append(idx)

            if t not in tail2index:
                tail2index[t] = []
            tail2index[t].append(idx)

        return head_relation_tail_np, head2index, tail2index


class GraphDB:
    def __init__(self, head_relation_tail_np, head2index, tail2index,
                 hop, max_node_num, single_node_max_triple1,
                 single_node_max_triple2=None):
        self.head_relation_tail_np = head_relation_tail_np
        self.head2indices = head2index
        self.tail2indices = tail2index
        self.hop = hop
        self.max_node_num = max_node_num
        self.single_node_max_triple1 = single_node_max_triple1
        self.single_node_max_triple2 = single_node_max_triple2

    def retrieve_triples(self, batch_state_index):
        batch_triples = []
        triple_nums = []

        pad_triple = np.asarray([0, 0, 0])
        for b_idx in range(len(batch_state_index)):
            triples = []

            # 1-hop
            for s_idx in range(len(batch_state_index[b_idx])):
                main_node = batch_state_index[b_idx][s_idx]
                if main_node in self.head2indices:
                    triples1 = [self.head_relation_tail_np[idx] for idx in
                                self.head2indices[main_node][:self.single_node_max_triple1]]
                    triples += triples1

                if main_node in self.tail2indices:
                    triples2 = [self.head_relation_tail_np[idx] for idx in
                                self.tail2indices[main_node][:self.single_node_max_triple1]]
                    triples += triples2

            triples = triples[:self.max_node_num * 2]
            triple_nums.append(len(triples))

            if len(triples) < self.max_node_num * 2:
                triples = triples + [pad_triple] * (self.max_node_num * 2 - len(triples))
            batch_triples.append(triples)

        # B， N_triple, 3 | B
        return np.asarray(batch_triples, dtype=np.long), np.asarray(triple_nums, dtype=np.long)

    def graph_construct(self, batch_state_index):
        batch_adjacent_matrix = []
        batch_head_nodes = []
        batch_node_efficient = []
        batch_head_flag_bit_matrix = []
        batch_edge_type_matrix = []

        for b_idx in range(len(batch_state_index)):
            triples = []
            nodeset = set()
            seed_node_set = set()
            one_hop_node_set = set()

            for s_idx in range(len(batch_state_index[0])):
                main_node = batch_state_index[b_idx][s_idx]
                seed_node_set.add(main_node)

                # 1-hop
                triples1 = []
                if main_node in self.head2indices:
                    triples1 = [self.head_relation_tail_np[idx] for idx in
                                self.head2indices[main_node][:self.single_node_max_triple1]]
                    triples += triples1
                if main_node in self.tail2indices:
                    triples2 = [self.head_relation_tail_np[idx] for idx in
                                self.tail2indices[main_node][:self.single_node_max_triple1]]
                    triples += triples2

                nodeset.add(main_node)
                for _, _, tail_node in triples1:
                    nodeset.add(tail_node)
                    one_hop_node_set.add(tail_node)

            # 2-hop
            if self.hop == 2:
                for main_node in one_hop_node_set:
                    triples1 = []
                    if main_node in self.head2indices:
                        triples1 = [self.head_relation_tail_np[idx] for idx in
                                    self.head2indices[main_node][:self.single_node_max_triple2]]
                        triples += triples1

                    if main_node in self.tail2indices:
                        triples2 = [self.head_relation_tail_np[idx] for idx in
                                    self.tail2indices[main_node][:self.single_node_max_triple2]]
                        triples += triples2

                    for _, _, tail_node in triples1:
                        nodeset.add(tail_node)

            # session inner connect
            state_node_index = set(list(batch_state_index[b_idx]))
            state_node_list = list(state_node_index)

            for s_idx in range(len(state_node_list)):
                for end_node in state_node_list[s_idx + 1:]:
                    start_node = state_node_list[s_idx]
                    triples.append([start_node, 1, end_node])
                    triples.append([end_node, 1, start_node])

            triples = triples[:self.max_node_num * 2]
            head_nodes = list(nodeset)[:self.max_node_num]
            node2idx = dict(zip(head_nodes, list(range(len(head_nodes)))))
            head_flag_bit_matrix = [1 if n in state_node_index else 0 for n in head_nodes]
            adjacent_matrix = np.zeros((self.max_node_num, self.max_node_num), dtype=np.long)
            edge_type_matrix = np.zeros((self.max_node_num, self.max_node_num), dtype=np.long)

            for h, r, t in triples:
                if h in node2idx and t in node2idx:
                    h_idx, t_idx = node2idx[h], node2idx[t]
                    adjacent_matrix[h_idx, t_idx] = 1
                    if r > edge_type_matrix[h_idx, t_idx]:
                        edge_type_matrix[h_idx, t_idx] = r

            node_efficient = [1] * len(head_nodes) + [0] * (self.max_node_num - len(head_nodes))
            head_nodes = head_nodes + [0] * (self.max_node_num - len(head_nodes))
            head_flag_bit_matrix = head_flag_bit_matrix + [0] * (self.max_node_num - len(head_flag_bit_matrix))

            batch_adjacent_matrix.append(adjacent_matrix)
            batch_head_nodes.append(head_nodes)
            batch_node_efficient.append(node_efficient)
            batch_head_flag_bit_matrix.append(head_flag_bit_matrix)
            batch_edge_type_matrix.append(edge_type_matrix)

        return (np.asarray(batch_adjacent_matrix),
                np.asarray(batch_head_nodes),
                np.asarray(batch_node_efficient),
                np.asarray(batch_head_flag_bit_matrix),
                np.asarray(batch_edge_type_matrix))
