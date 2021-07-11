# -*- coding: utf-8 -*-
import argparse
from resource.option.train_option import TrainOption as TO
from resource.option.dataset_option import DatasetOption as DO
from resource.option.vrbot_option import VRBotOption as VO


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["kamed", "meddialog", "meddg", "meddg_dev"], required=True)
    parser.add_argument("--super_rate", type=float, choices=[0.0, 0.1, 0.25, 0.5, 1.0], default=0.0)
    parser.add_argument("--only_supervised", action="store_true")
    parser.add_argument("--auto_regressive", action="store_true")
    model_choices = ["vrbot"]
    parser.add_argument("--joint_dist", action="store_true")
    parser.add_argument("--no_action_super", action="store_true")
    parser.add_argument("--model", choices=model_choices, required=True)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--worker_num", type=int, default=3)
    parser.add_argument("--valid_eval_interval", type=int, default=10000)
    parser.add_argument("--test_eval_interval", type=int, default=10000)
    parser.add_argument("--cache_turn", type=int, default=0)
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--ppn_dq", action="store_true")
    parser.add_argument("--gen_strategy", choices=["mlp", "gru"], default="gru")
    parser.add_argument("--mem_depth", type=int, default=3)
    parser.add_argument("--state_action_copy_lambda", type=float, default=1.0)
    parser.add_argument("--copy_lambda_decay_interval", type=int, default=10000)
    parser.add_argument("--tau_decay_interval", type=int, default=5000)
    parser.add_argument("--copy_lambda_decay_value", type=float, default=1.0)
    parser.add_argument("--state_action_copy_lambda_mini", type=float, default=1.0)
    parser.add_argument("--init_tau", type=float, default=1.0)
    parser.add_argument("--force_ckpt_dump", action="store_true")
    parser.add_argument("--mask_state_prob", action="store_true")
    parser.add_argument("--no_classify_weighting", action="store_true")
    parser.add_argument("--gradient_stack", type=int, default=1)
    parser.add_argument("--inference_set", choices=["test", "valid"], default="test")
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--train_stage", choices=["action", "natural"], default="natural")
    parser.add_argument("--state_num", type=int, default=10)
    parser.add_argument("--action_num", type=int, default=3)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--mini_lr", type=float, default=1e-5)
    parser.add_argument("--decay_rate", type=float, default=0.9)
    parser.add_argument("--decay_interval", type=int, default=5000)

    args = parser.parse_args()
    TO.joint_dist = args.joint_dist
    TO.model = args.model
    TO.no_action_super = args.no_action_super
    TO.cache_turn = args.cache_turn
    TO.auto_regressive = args.auto_regressive and (args.gen_strategy == "gru")
    TO.only_supervised = args.only_supervised
    TO.update_device(args.device)
    TO.super_rate = args.super_rate
    TO.beam_width = args.beam_width
    VO.ppn_dq = args.ppn_dq
    VO.train_stage = args.train_stage
    TO.no_classify_weighting = args.no_classify_weighting

    TO.update("mem_depth", args.mem_depth)
    TO.update("worker_num", args.worker_num)
    TO.update("lr", args.init_lr)
    TO.update("mini_lr", args.mini_lr)
    TO.update("decay_rate", args.decay_rate)
    TO.update("decay_interval", args.decay_interval)
    TO.update("debug", args.debug)
    TO.update("train_batch_size", args.train_batch_size)
    TO.update("test_batch_size", args.test_batch_size)
    TO.update("task", args.task)
    TO.update("valid_eval_interval", args.valid_eval_interval)
    TO.update("test_eval_interval", args.test_eval_interval)
    TO.update("force_ckpt_dump", args.force_ckpt_dump)
    DO.update_joint_graph(TO.task)

    VO.state_num = args.state_num
    DO.state_num = args.state_num
    VO.action_num = args.action_num
    DO.action_num = args.action_num
    VO.s_copy_lambda = max(args.state_action_copy_lambda, args.state_action_copy_lambda_mini)
    VO.a_copy_lambda = max(args.state_action_copy_lambda, args.state_action_copy_lambda_mini)
    VO.copy_lambda_decay_interval = args.copy_lambda_decay_interval
    VO.copy_lambda_decay_value = args.copy_lambda_decay_value
    VO.state_action_copy_lambda_mini = args.state_action_copy_lambda_mini
    VO.init_tau = args.init_tau
    VO.tau_decay_interval = args.tau_decay_interval
    VO.mask_state_prob = args.mask_state_prob

    return args
