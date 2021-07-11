# -*- coding: utf-8 -*-

import torch
import random
from resource.option.config import config
from resource.util.get_logger import get_logger
from resource.option.train_option import TrainOption as TO
from resource.option.dataset_option import DatasetOption as DO
from resource.option.vrbot_option import VRBotOption as VO

main_logger = get_logger("main", "data/log/{}.log".format(TO.task_uuid))
main_logger.info("TASK ID {}".format(TO.task_uuid))

from resource.model.vrbot import VRBot
from resource.vrbot_engine import VRBotEngine
from resource.model.vrbot_train_state import vrbot_train_stage
from resource.input.graph_db import GraphDB, TripleLoader
from resource.input.data_processor import DataProcessor
from resource.input.session_dataset import SessionDataset
from resource.input.session_dataset import MixedSessionDataset
from resource.input.session_dataset import SessionProcessor
from resource.util.loc_glo_trans import LocGloInterpreter


def prepare_data(args):
    main_logger.info("preparing sessions")
    data_processor = DataProcessor(args.task)
    train_sessions, test_sessions, valid_sessions = data_processor.get_session()

    main_logger.info("preparing vocab")
    word_vocab, know_vocab, glo2loc, loc2glo, vocab_size, inner_vocab_size = data_processor.get_vocab()
    glo2loc = torch.tensor(glo2loc, device=TO.device)
    loc2glo = torch.tensor(loc2glo, device=TO.device)

    return [train_sessions, test_sessions, valid_sessions,
            word_vocab, know_vocab, glo2loc, loc2glo, vocab_size, inner_vocab_size]


def cfg2str(option):
    cfg_str = ["\n======= {} START =======".format(option.__name__)]
    for key, value in option.__dict__.items():
        if key.startswith("_"):
            continue
        cfg_str.append("{} : {}".format(key, value))
    cfg_str += ["======= {} END =======\n".format(option.__name__)]
    return "\n".join(cfg_str)


def main():
    seed = 123
    random.seed(seed)
    main_logger.info("PARAMETER PARSING")

    args = config()
    vrbot_train_stage.update_relay()
    main_logger.info(cfg2str(VO))

    main_logger.info("PREPARE DATA")
    train_sessions, test_sessions, valid_sessions, word_vocab, inner_vocab, \
        glo2loc, loc2glo, vocab_size, inner_vocab_size = prepare_data(args)
    sp = SessionProcessor(word_vocab, inner_vocab, DO.pv_r_u_max_len, DO.r_max_len)

    if 1. > args.super_rate > .0:
        super_num = int(len(train_sessions) * args.super_rate)
        random.shuffle(train_sessions)
        super_train_sessions, unsuper_train_sessions = train_sessions[:super_num], train_sessions[super_num:]
        train_dataset = MixedSessionDataset(sp, args.train_batch_size, super_train_sessions,
                                            unsuper_train_sessions, args.super_rate)
    else:
        train_dataset = SessionDataset(sp, "train", args.train_batch_size, train_sessions,
                                       supervised=True if args.super_rate == 1. else False)

    valid_dataset = SessionDataset(sp, "valid", args.test_batch_size, valid_sessions,
                                   supervised=True if args.task.startswith("meddg") else False)
    test_dataset = SessionDataset(sp, "test", args.test_batch_size, test_sessions,
                                  supervised=True if args.task.startswith("meddg") else False)

    lg_interpreter = LocGloInterpreter(loc2glo, glo2loc)
    triple_loader = TripleLoader(DO.joint_graph_filename, inner_vocab)
    head_relation_tail_np, head2index, tail2index = triple_loader.load_triples()
    graph_db = GraphDB(head_relation_tail_np, head2index, tail2index,
                       args.hop, VO.max_node_num1 if args.hop == 1 else VO.max_node_num2,
                       VO.single_node_max_triple1, VO.single_node_max_triple2)

    model = VRBot(loc2glo, VO.state_num, VO.action_num, VO.hidden_dim,
                  inner_vocab_size, vocab_size, VO.response_max_len, VO.embed_dim,
                  lg_interpreter, gen_strategy=args.gen_strategy,
                  with_copy=True, graph_db=graph_db, beam_width=TO.beam_width)

    if args.device >= 0:
        model = model.to(TO.device)

    engine = VRBotEngine(model, train_dataset, valid_dataset, test_dataset, word_vocab, inner_vocab)

    epoch = 0
    if args.ckpt is not None:
        main_logger.info("LOAD CHECKPOINT FROM {}".format(args.ckpt))
        epoch, global_step, origin_task_uuid = engine.load_model(args.ckpt)
        engine.global_step = global_step
    elif (args.ckpt is None) and args.test:
        main_logger.warn("NO CHECKPOINT PROVIDED, INITIAL MODEL RANDOMLY")

    if not args.test:
        engine.train(start_epoch=epoch)
    else:
        dataset = engine.test_dataset if args.inference_set == "test" else engine.valid_dataset
        mode = "test" if args.inference_set == "test" else "valid"
        model_name = model.__class__.__name__.upper()
        engine.test_with_log(dataset, epoch, model_name, mode)


if __name__ == '__main__':
    main()
