# -*- coding: utf-8 -*-

import logging
import torch
import numpy as np
from resource.util.writer4json import JsonWriter
from resource.input.session_dataset import SessionDataset
from resource.option.train_option import TrainOption as TO
from resource.option.dataset_option import DatasetOption as DO
from resource.util.misc import mkdir_if_necessary

engine_logger = logging.getLogger("main.base_engine")


class BaseEngine:
    def __init__(self, model: torch.nn.Module, train_dataset: SessionDataset, valid_dataset: SessionDataset,
                 test_dataset: SessionDataset, lr, **kwargs):
        super(BaseEngine, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.global_step = 0
        self.json_writer = JsonWriter()
        self.bst_metrics = [0.0 for _ in range(4)]

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        # return targets, outputs, metrics, session_cropper
        raise NotImplementedError

    def test_with_log(self, dataset, epoch, model_name, mode):
        targets, outputs, metrics, session_cropper = self.test(dataset, mode=mode)

        metric_str = "(" + "-".join(["{:.4f}".format(x) for x in metrics]) + ")"
        # "data/cache/{model}/{uuid}/{epoch}-{global_step}-{mode}-{metric}.txt"
        valid_output_filename = DO.test_filename_template.format(model=model_name,
                                                                 uuid=TO.task_uuid,
                                                                 epoch=epoch,
                                                                 global_step=self.global_step,
                                                                 mode=mode,
                                                                 metric=metric_str)
        mkdir_if_necessary(valid_output_filename)
        engine_logger.info("WRITE {} OUTPUT TO FILE {}".format(mode, valid_output_filename))
        self.json_writer.write2file(valid_output_filename, session_cropper.to_dict())

        if mode == "valid":
            if sum(metrics) > sum(self.bst_metrics):
                self.bst_metrics = metrics
                engine_logger.info("MODEL REACH THE BEST RESULT IN EPOCH {} STEP {}".format(epoch, self.global_step))
                # "data/ckpt/{model}/{uuid}/{epoch}-{global_step}-{metric}.model.ckpt"
                ckpt_filename = DO.ckpt_filename_template.format(model=model_name,
                                                                 uuid=TO.task_uuid,
                                                                 epoch=epoch,
                                                                 global_step=self.global_step,
                                                                 metric=metric_str)
                mkdir_if_necessary(ckpt_filename)
                self.dump_model(epoch, self.global_step, ckpt_filename)
            else:
                engine_logger.info("METRICS NOT IMPROVED IN EPOCH {} STEP {}".format(epoch, self.global_step))

    def dump_model(self, epoch, step, ckpt_filename):
        dump_dict = {
            "epoch": epoch,
            "step": step,
            "ckpt": self.model.state_dict(),
            "task_uuid": TO.task_uuid,
        }
        engine_logger.info("DUMPING CKPT TO FILE {}".format(ckpt_filename))
        torch.save(dump_dict, ckpt_filename)
        engine_logger.info("DUMPING CKPT DONE")

    def load_model(self, ckpt_filename):
        engine_logger.info("LOAD CKPT FROM {}".format(ckpt_filename))
        dump_dict = torch.load(ckpt_filename)
        epoch = dump_dict["epoch"]
        step = dump_dict["step"]
        task_uuid = dump_dict["task_uuid"]
        ckpt = dump_dict["ckpt"]
        self.model.load_state_dict(ckpt, strict=False)
        return epoch, step, task_uuid

    @staticmethod
    def adjust_learning_rate(optimizer, rate_decay, mini_lr):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            param_group['lr'] = max(lr * rate_decay, mini_lr)

    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        param_cnt = sum([np.prod(p.size()) for p in module_parameters])
        print('total trainable params: %d' % param_cnt)


