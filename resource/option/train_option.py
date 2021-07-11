# -*- coding: utf-8 -*-

import os
import uuid
import torch


class TrainOption:
    lr = 1e-4
    cache_turn = 0
    mini_lr = 1e-5
    decay_rate = 0.9
    decay_interval = 5000
    epoch = 20
    epoch_num = 100
    train_batch_size = 64
    efficient_train_batch_size = 64
    test_batch_size = 128
    valid_batch_size = 128
    worker_num = 3
    trunc_interval = 2
    input_buffer_size = 100
    output_buffer_size = 100
    valid_eval_interval = 10000
    test_eval_interval = 10000
    log_loss_interval = 200
    device = torch.device("cpu:0")
    task_uuid = str(uuid.uuid4())[:8]
    early_stop_count = 3
    debug = False
    joint_dist = False
    no_action_super = False

    without_t_know = False
    without_p_know = False
    without_copy = False
    attn_history_sentence = True
    attn_history_memory = True

    with_t_memory = True
    with_p_memory = True
    with_copy = True
    history_hop = 5
    consider_context_len = 10
    candidate_num = 200
    beam_decode_max_step = 50
    curtail_train = False
    beam_width = 5
    super_rate = 0.0
    gradient_stack = 1
    add_posterior = False
    episode_len = 2
    task = None
    lambda_0 = 1.0
    lambda_1 = 1.0
    lambda_2 = 0.01
    force_ckpt_dump = False
    discourse_type = [i for i in "asdmt"]
    model = None
    auto_regressive = False
    only_supervised = False
    intention_dict = {"c": 0,
                      "a": 0,
                      "s": 1,
                      "d": 2,
                      "m": 3,
                      "t": 3}
    k = 0.0
    K = 5.0
    focal = False

    fullname2abbr = {
        "chitchat": "c",
        "apprentice_response": "a",
        "prescribe_medicine": "m",
        "ask_symptoms": "s",
        "diagnosis_disease": "d",
        "diagnosis_treatment": "t"
    }

    loss_weight_floor = 0.2
    loss_weight_ceiling = 5
    with_gate = True
    copy_history = False
    aux_device = False
    no_classify_weighting = False

    @staticmethod
    def update_device(device_id):
        if device_id < 0:
            return
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(device_id)
            TrainOption.device = torch.device("cuda:{}".format(0))

    @staticmethod
    def update_curtail_train(ct):
        TrainOption.curtail_train = ct

    @staticmethod
    def update_lr(lr):
        TrainOption.lr = lr

    @staticmethod
    def update(attr, value):
        setattr(TrainOption, attr, value)
