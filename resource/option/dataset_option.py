# -*- coding: utf-8 -*-


import os


class DatasetOption:
    class PreventWord:
        SOS = "<sos>"
        EOS = "<eos>"
        PAD = "<pad>"
        UNK = "<unk>"
        SENTENCE_SPLITER = "<sent>"

        SOS_ID = 0
        EOS_ID = 1
        PAD_ID = 2
        UNK_ID = 3
        SENTENCE_SPLITER_ID = 4
        RESERVED_MAX_INDEX = SENTENCE_SPLITER_ID

    tgt_hyp_spliter = " <==> "
    test_filename_template = "data/cache/{model}/{uuid}/{epoch}-{global_step}-{mode}-{metric}.txt"
    ckpt_filename_template = "data/ckpt/{model}/{uuid}/{epoch}-{global_step}-{metric}.model.ckpt"
    session_zip_filename = "data/{dataset}.zip"
    session_zip_temp = "data/{}.zip"
    knowledge_zip_filename = "data/{dataset}_tokenized_knowledge_entities.zip"
    vocab_csv_filename = "data/vocabs/{dataset}_vocab.csv"
    diseases_name_filename = "data/filter_data/{dataset}_preserve_disease.json"
    reserved_relation_file = "data/config/reserved_relations.csv"
    vocab_filename = "data/vocab.txt"
    del_link_set = {"ICD-10", "UMLS"}

    @staticmethod
    def update_joint_graph(task):
        DatasetOption.joint_graph_path = DatasetOption.joint_graph_path.format(task)
        DatasetOption.alias2scientific_filename = os.path.join(DatasetOption.joint_graph_path, "alias2scientific.json")
        DatasetOption.entity2type_filename = os.path.join(DatasetOption.joint_graph_path, "entity2type.json")
        DatasetOption.entity2id_filename = os.path.join(DatasetOption.joint_graph_path, "entity2id.json")
        DatasetOption.relation2id_filename = os.path.join(DatasetOption.joint_graph_path, "relation2id.json")
        DatasetOption.joint_graph_filename = os.path.join(DatasetOption.joint_graph_path, "graph.json")

    joint_graph_path = "data/{}_joint_graph"
    alias2scientific_filename = os.path.join(joint_graph_path, "alias2scientific.json")
    entity2type_filename = os.path.join(joint_graph_path, "entity2type.json")
    entity2id_filename = os.path.join(joint_graph_path, "entity2id.json")
    relation2id_filename = os.path.join(joint_graph_path, "relation2id.json")
    joint_graph_filename = os.path.join(joint_graph_path, "graph.json")

    test_num = 3000
    valid_num = 3000
    vocab_size = 30000
    know_vocab_size = 10000
    embed_dim = 300
    trans_embed_dim = 512

    pv_r_u_max_len = 200
    r_max_len = 100
    knowledge_key_relation_len = 5
    triple_len = 50

    action_num = 3
    state_num = 10

    max_knowledge_per_session = 10
    removable_relation_filename = "data/filter_data/removable_relations.txt"
