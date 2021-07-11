# -*- coding: utf-8 -*-


class VRBotOption:
    state_num = 10
    action_num = 3
    triple_num_per_graph = 10
    know_vocab_size = None
    vocab_size = None
    response_max_len = 100

    embed_dim = 300
    hidden_units = 512
    hidden_dim = 512
    num_layers = 1
    bidirectional = True
    rnn_hidden_size = hidden_units if not bidirectional else (hidden_units // 2)
    dropout = 0.2
    attention_type = "mlp"

    # ablation study
    train_stage = "state"  # state, action, natural
    ppn_dq = False
    max_context_turn = 10
    max_sentence_word = 100
    max_target_word = 100

    # for state copy
    s_copy_lambda = 1.0
    a_copy_lambda = 1.0
    copy_lambda_decay_interval = 10000
    copy_lambda_decay_value = 1.0
    state_action_copy_lambda_mini = 1.0
    init_tau = 1.0
    tau_mini = 0.1
    tau_decay_interval = 5000
    tau_decay_rate = 0.5
    mask_state_prob = False
    node_embed_dim = 128

    # configuration
    with_weak_action = False
    training_sampling_times = 5

    attn_history_sentence = True
    with_state_know = True
    with_action_know = True
    with_copy = True

    class GATConfig:
        embed_dim = 128
        edge_embed_dim = 16
        flag_embed_dim = 16
        action_embed_dim = 16
        node_num = 6000
        edge_num = 20
        flag_num = 10
        action_num = 10

    max_node_num1 = 100
    max_node_num2 = 50
    single_node_max_triple1 = 10
    single_node_max_triple2 = 5
