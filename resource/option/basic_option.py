# -*- coding: utf-8 -*-


class BasicOption:
    hidden_size = 512
    vocab_size = 30000
    embed_dim = 300
    rnn_hidden_dim = hidden_size // 2
    bidirectional = True
    target_max_len = 100
