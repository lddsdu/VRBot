# -*- coding: utf-8 -*-

import torch
import logging
from tqdm import tqdm
from resource.input.vocab import Vocab
from resource.model.vrbot import TRAIN
from resource.model.vrbot import VRBot
from resource.metric.eval_bleu import eval_bleu
from resource.input.tensor2nl import TensorNLInterpreter
from resource.input.session_dataset import SessionDataset
from resource.option.train_option import TrainOption as TO
from resource.option.vrbot_option import VRBotOption as VO
from resource.option.dataset_option import DatasetOption as DO
from resource.util.misc import mkdir_if_necessary
from resource.input.session import SessionCropper
from resource.util.misc import one_hot_scatter
from resource.base_engine import BaseEngine
from resource.model.vrbot_train_state import vrbot_train_stage

engine_logger = logging.getLogger("main.engine")


class VRBotEngine(BaseEngine):
    def __init__(self, model: VRBot,
                 train_dataset: SessionDataset,
                 valid_dataset: SessionDataset,
                 test_dataset: SessionDataset,
                 vocab: Vocab,
                 inner_vocab: Vocab,
                 lr=None):
        super(VRBotEngine, self).__init__(model, train_dataset, valid_dataset, test_dataset, lr or TO.lr)
        self.iw_interpreter = TensorNLInterpreter(vocab)
        self.ii_interpreter = TensorNLInterpreter(inner_vocab)
        self.cache = {"super": {"pv_hidden": None, "pv_state": None},
                      "unsuper": {"pv_hidden": None, "pv_state": None}}

    def train(self, dataset=None, start_epoch=0, end_epoch=50, state_pretrain_epoch=20):
        dataset = dataset or self.train_dataset
        epoch_point = start_epoch
        VO.train_stage = "action"

        # STATE-TRAINING
        if VO.train_stage == "state" or VO.train_stage == "natural":
            for e in range(start_epoch, state_pretrain_epoch):
                engine_logger.info("[EPOCH {:0>2}] START - INFERENCE TRAINING (state pre-train)".format(e + 1))
                sig = self.stage_train(dataset, e, state_train=True)
                engine_logger.info("[EPOCH {:0>2}] FINISHED - INFERENCE TRAINING (state pre-train)".format(e + 1))
                epoch_point = e + 1
                if sig is not None:  # early quit state-training
                    engine_logger.info("state training early stop at {} epoch, {} step".format(e, self.global_step))
                    break

        # ACTION-TRAINING
        for e in range(epoch_point, end_epoch):
            engine_logger.info("[EPOCH {:0>2}] START - INFERENCE TRAINING".format(e + 1))
            self.stage_train(dataset, e, state_train=False)
            engine_logger.info("[EPOCH {:0>2}] FINISHED - INFERENCE TRAINING".format(e + 1))

    def stage_train(self, dataset: SessionDataset, epoch, state_train=False):
        all_stage = "[{}]".format("STATE" if state_train else "ACTION")
        model_name = self.model.__class__.__name__
        dataset_bs = dataset.batch_size
        pad_loss = torch.tensor(0.0, dtype=torch.float, device=TO.device)

        # INITIAL HIDDEN & STATE
        # 1, B, H
        _init_pv_hidden = torch.zeros((1, dataset_bs, self.model.hidden_dim), dtype=torch.float, device=TO.device)
        # B, S, Vi
        _init_pv_state = torch.zeros((dataset_bs, self.model.state_num, self.model.inner_vocab_size),
                                     dtype=torch.float, device=TO.device)
        _init_pv_state[:, :, 0] = 1.0

        self.cache["super"]["pv_hidden"] = _init_pv_hidden
        self.cache["super"]["pv_state"] = _init_pv_state
        self.cache["unsuper"]["pv_hidden"] = _init_pv_hidden.clone()
        self.cache["unsuper"]["pv_state"] = _init_pv_state.clone()

        # TRAINING PROCESS
        train_bar = tqdm(dataset.load_data())  # input tensors iterator

        for input_tensors, inherited, materialistic in train_bar:
            self.global_step += 1
            loss_log_tmp = "[TASK-UUID {} EPOCH {:0>2} STEP {:0>6} TRAIN-STAGE {}] - {}"
            loss_log_template = loss_log_tmp.format(TO.task_uuid, epoch, self.global_step, all_stage, "{loss_info}")

            if len(input_tensors) == 5:
                pv_r_u, pv_r_u_len, r, r_len, gth_i = input_tensors
                gth_i = gth_i.float()
                gth_a, gth_s = None, None
                batch_supervised = False
            elif len(input_tensors) == 7:
                pv_r_u, pv_r_u_len, r, r_len, gth_s, gth_i, gth_a = input_tensors
                gth_i = gth_i.float()
                batch_supervised = True
            else:
                raise RuntimeError("error input tensor numbers {}".format(len(input_tensors)))

            key1 = "super" if batch_supervised else "unsuper"
            pv_hidden = self.cache[key1]["pv_hidden"]
            pv_state = self.cache[key1]["pv_state"]
            pv_hidden, pv_state = self.hidden_state_mask(pv_hidden, pv_state, inherited, materialistic)

            if state_train:
                vrbot_train_stage.state_train_tick()
                train_state_rets = self.model.forward(pv_state, pv_hidden, pv_r_u,
                                                      pv_r_u_len, gth_i, r, r_len,
                                                      gth_action=gth_a, gth_state=gth_s,
                                                      train_stage=TRAIN.TRAIN_STATE,
                                                      supervised=batch_supervised)

                gen_log_probs1 = train_state_rets[0]
                post_state_prob = train_state_rets[2]
                prior_state_prob = train_state_rets[3]
                state_gumbel_prob = train_state_rets[4]
                action_gumbel_prob = train_state_rets[7]
                prior_intention = train_state_rets[9]
                hidden4post = train_state_rets[10]

                # loss
                state_nll_loss, _ = self.model.nll_loss(gen_log_probs1, r.detach()[:, 1:])
                state_kl_loss = self.model.kl_loss(prior_state_prob, post_state_prob.detach())

                if prior_intention is None or gth_i is None:
                    prior_intention_ce_loss = pad_loss
                else:
                    prior_intention_ce_loss = self.model.cross_entropy_loss(prior_intention, gth_i)

                if batch_supervised:
                    aux_state_loss = (self.model.state_nll(prior_state_prob, gth_s) +
                                      self.model.state_nll(post_state_prob, gth_s))
                else:
                    aux_state_loss = pad_loss

                loss = state_nll_loss + state_kl_loss + prior_intention_ce_loss + aux_state_loss
                loss_template = "{} loss: {:.4f} r_nll: {:.4f} s_kl: {:.4f} p_i_ce: {:.4f} aux_s_nll: {:.4f}"
                loss_info = loss_template.format("SUPER" if batch_supervised else "UNSUPER",
                                                 loss.item(),
                                                 state_nll_loss.item(),
                                                 state_kl_loss.item(),
                                                 prior_intention_ce_loss.item(),
                                                 aux_state_loss.item())

            else:
                vrbot_train_stage.state_train_tick()
                vrbot_train_stage.action_train_tick()
                train_policy_rets = self.model.forward(pv_state, pv_hidden, pv_r_u, pv_r_u_len,
                                                       gth_i, r, r_len,
                                                       gth_action=gth_a, gth_state=gth_s,
                                                       train_stage=TRAIN.TRAIN_POLICY,
                                                       supervised=batch_supervised)

                gen_log_probs1 = train_policy_rets[0]
                gen_log_probs2 = train_policy_rets[1]
                post_state_prob = train_policy_rets[2]
                prior_state_prob = train_policy_rets[3]
                state_gumbel_prob = train_policy_rets[4]
                post_action_prob = train_policy_rets[5]
                prior_action_prob = train_policy_rets[6]
                action_gumbel_prob = train_policy_rets[7]
                post_intention = train_policy_rets[8]
                prior_intention = train_policy_rets[9]
                hidden4post = train_policy_rets[10]

                state_nll_loss, _ = self.model.nll_loss(gen_log_probs1, r.detach()[:, 1:])
                action_nll_loss, _ = self.model.nll_loss(gen_log_probs2, r.detach()[:, 1:])

                if batch_supervised:
                    state_kl_loss = pad_loss
                    action_kl_loss = pad_loss
                else:
                    state_kl_loss = self.model.kl_loss(prior_state_prob, post_state_prob.detach())
                    action_kl_loss = self.model.kl_loss(prior_action_prob, post_action_prob.detach())

                if prior_intention is None:
                    intention_loss = pad_loss
                else:
                    prior_intention_ce_loss = self.model.cross_entropy_loss(prior_intention, gth_i)
                    post_intention_ce_loss = self.model.cross_entropy_loss(post_intention, gth_i)
                    if batch_supervised:
                        intention_kl_loss = pad_loss
                    else:
                        intention_kl_loss = self.model.kl_loss(prior_intention, post_intention.detach())
                    intention_loss = prior_intention_ce_loss + post_intention_ce_loss + intention_kl_loss

                if batch_supervised:
                    aux_state_loss = (self.model.state_nll(prior_state_prob, gth_s) +
                                      self.model.state_nll(post_state_prob, gth_s))
                    aux_action_loss = (self.model.state_nll(prior_action_prob, gth_a) +
                                       self.model.state_nll(post_action_prob, gth_a))
                else:
                    aux_state_loss, aux_action_loss = pad_loss, pad_loss

                a_lambda, i_lambda = 0.2, 1.0
                loss = (state_nll_loss + a_lambda * action_nll_loss +
                        state_kl_loss + a_lambda * action_kl_loss +
                        i_lambda * intention_loss + aux_state_loss + aux_action_loss)

                log_tmp = "{} s-nll: {:.4f} a-nll: {:.4f} s-kl: {:.4f} " \
                          "a-kl: {:.4f} i-l: {:.4f} sa-l: {:.4f} aa-l: {:.4f}"
                loss_info = log_tmp.format("SUPER" if batch_supervised else "UNSUPER",
                                           state_nll_loss.item(), action_nll_loss.item(),
                                           state_kl_loss.item(), action_kl_loss.item(),
                                           intention_loss.item(), aux_state_loss.item(),
                                           aux_action_loss.item())

            loss_log_info = loss_log_template.format(loss_info=loss_info)
            train_bar.set_description(loss_log_info)
            if self.global_step % TO.log_loss_interval == 0:
                engine_logger.info(loss_log_info)

            self.optimizer.zero_grad()
            loss = loss / float(TO.gradient_stack)
            loss.backward(retain_graph=False)

            if self.global_step % TO.gradient_stack == 0:
                self.optimizer.step()

            if self.global_step % TO.decay_interval == 0:
                engine_logger.info("learning rate decay")
                self.adjust_learning_rate(self.optimizer, TO.decay_rate, TO.mini_lr)

            if self.global_step % TO.valid_eval_interval == 0:
                self.test_with_log(self.valid_dataset, epoch, model_name, mode="valid")

            if self.global_step % TO.test_eval_interval == 0:
                self.test_with_log(self.test_dataset, epoch, model_name, mode="test")

            # copy weight decay
            self.tick(state_train=state_train,
                      action_train=not state_train)

            pv_hidden = hidden4post
            if batch_supervised:  # use the previous gth state
                pv_state = one_hot_scatter(gth_s, state_gumbel_prob.size(-1), dtype=torch.float)
            else:
                pv_state = state_gumbel_prob

            self.cache[key1]["pv_hidden"] = pv_hidden.detach()
            self.cache[key1]["pv_state"] = pv_state.detach()

    def test(self, dataset: SessionDataset, mode="test"):
        assert mode == "test" or mode == "valid"
        print("SESSION NUM: {}".format(len(dataset.sessions)))
        dataset_bs = dataset.batch_size
        pv_hidden = torch.zeros((1, dataset_bs, self.model.hidden_dim), dtype=torch.float, device=TO.device)
        pv_state = torch.zeros((dataset_bs, self.model.state_num, self.model.inner_vocab_size),
                               dtype=torch.float, device=TO.device)
        pv_state[:, :, 0] = 1.0

        # cache
        all_targets = []
        all_outputs = []

        engine_logger.info("{} INFERENCE START ...".format(mode.upper()))
        session_cropper = SessionCropper(dataset.batch_size)

        self.model.eval()
        with torch.no_grad():
            for input_tensors, inherited, materialistic in tqdm(dataset.load_data()):
                if len(input_tensors) == 5:
                    pv_r_u, pv_r_u_len, r, r_len, gth_intention = input_tensors
                    gth_s, gth_a = None, None
                elif len(input_tensors) == 7:
                    pv_r_u, pv_r_u_len, r, r_len, gth_s, gth_intention, gth_a = input_tensors
                else:
                    raise RuntimeError

                pv_hidden, pv_state = self.hidden_state_mask(pv_hidden, pv_state, inherited, materialistic)

                gen_log_probs, state_index, action_index, hidden4post = self.model.forward(pv_state,
                                                                                           pv_hidden,
                                                                                           pv_r_u,
                                                                                           pv_r_u_len,
                                                                                           None)

                posts = self.iw_interpreter.interpret_tensor2nl(pv_r_u)
                targets = self.iw_interpreter.interpret_tensor2nl(r[:, 1:])
                outputs = self.iw_interpreter.interpret_tensor2nl(gen_log_probs)
                states = self.ii_interpreter.interpret_tensor2nl(state_index)
                actions = self.ii_interpreter.interpret_tensor2nl(action_index)

                if gth_s is not None:
                    gth_states = self.ii_interpreter.interpret_tensor2nl(gth_s)
                else:
                    gth_states = ["<pad>"] * len(posts)

                inherited = inherited.detach().cpu().numpy().tolist()
                materialistic = materialistic.detach().cpu().numpy().tolist()
                session_cropper.step_on(posts, targets, outputs, states, actions,
                                        inherited, materialistic, gth_states)
                all_targets += targets
                all_outputs += outputs

                # for next loop
                pv_hidden = hidden4post
                pv_state = one_hot_scatter(state_index, self.model.inner_vocab_size, dtype=torch.float)

        self.model.train()
        engine_logger.info("{} INFERENCE FINISHED".format(mode.upper()))
        metrics = eval_bleu([all_targets], all_outputs)

        return all_targets, all_outputs, metrics, session_cropper

    @staticmethod
    def hidden_state_mask(pv_hidden, pv_state, inherited, materialistic):
        pv_hidden = pv_hidden * inherited.unsqueeze(0).unsqueeze(2).float()

        B, S, K = pv_state.shape
        state_placeholder = torch.zeros(B, S, K, dtype=torch.float, device=pv_state.device)
        state_placeholder[:, :, 0] = 1.0

        vn = int(inherited.sum().item())
        inherited_batch_index = \
            torch.sort(torch.arange(0, B, dtype=torch.long, device=TO.device) * inherited)[0][- vn:]

        state_placeholder[inherited_batch_index] = pv_state[inherited_batch_index]
        pv_state = state_placeholder

        batch_size = pv_hidden.size(1)
        vn = int(materialistic.sum().item())
        reserved_batch_index = \
            torch.sort(torch.arange(0, batch_size, dtype=torch.long, device=TO.device) * materialistic)[0][- vn:]

        reserved_hidden = pv_hidden[:, reserved_batch_index, :]
        reserved_state = pv_state[reserved_batch_index, :]

        return reserved_hidden, reserved_state

    def tick(self, state_train=False, action_train=False):
        if self.global_step % VO.copy_lambda_decay_interval == 0:
            if VO.s_copy_lambda > VO.state_action_copy_lambda_mini and state_train:
                VO.s_copy_lambda = max(VO.s_copy_lambda - VO.copy_lambda_decay_value,
                                       VO.state_action_copy_lambda_mini)
            if VO.a_copy_lambda > VO.state_action_copy_lambda_mini and action_train:
                VO.a_copy_lambda = max(VO.a_copy_lambda - VO.copy_lambda_decay_value,
                                       VO.state_action_copy_lambda_mini)

    @staticmethod
    def balance_act(origin_value, base_num):
        assert base_num > 1.0
        return base_num ** (origin_value - 1.0)

    def test_with_log(self, dataset, epoch, model_name, mode):
        targets, outputs, metrics, session_cropper = self.test(dataset, mode=mode)
        metric_str = "(" + "-".join(["{:.4f}".format(x) for x in metrics]) + ")"
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
            if sum(metrics) >= sum(self.bst_metrics) or TO.force_ckpt_dump:
                log_tmp = "MODEL DO NOT REACH THE BEST RESULT IN EPOCH {} STEP {}"
                if sum(metrics) >= sum(self.bst_metrics):
                    log_tmp = "MODEL REACH THE BEST RESULT IN EPOCH {} STEP {}"
                    self.bst_metrics = metrics
                engine_logger.info(log_tmp.format(epoch, self.global_step))
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
            "vrbot_train_stage": vrbot_train_stage.dump()
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
        vrbot_train_stage.self_update(dump_dict["vrbot_train_stage"])
        VO.s_copy_lambda = vrbot_train_stage.s_copy_lambda
        VO.a_copy_lambda = vrbot_train_stage.a_copy_lambda
        return epoch, step, task_uuid
