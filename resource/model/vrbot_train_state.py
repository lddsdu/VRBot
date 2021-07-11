# -*- coding: utf-8 -*-

from resource.option.vrbot_option import VRBotOption as VO


class VRBotTrainState:
    def __init__(self, s_copy_lambda, s_copy_lambda_mini, s_copy_lambda_decay_interval, s_copy_lambda_decay_value,
                 s_tau, s_tau_mini, s_tau_decay_interval, s_tau_decay_rate,
                 a_copy_lambda, a_copy_lambda_mini, a_copy_lambda_decay_interval, a_copy_lambda_decay_value,
                 a_tau, a_tau_mini, a_tau_decay_interval, a_tau_decay_rate):
        self.s_copy_lambda = s_copy_lambda
        self.s_copy_lambda_mini = s_copy_lambda_mini
        self.s_copy_lambda_decay_interval = s_copy_lambda_decay_interval
        self.s_copy_lambda_decay_value = s_copy_lambda_decay_value

        self.s_tau = s_tau
        self.s_tau_mini = s_tau_mini
        self.s_tau_decay_interval = s_tau_decay_interval
        self.s_tau_decay_rate = s_tau_decay_rate

        self.a_copy_lambda = a_copy_lambda
        self.a_copy_lambda_mini = a_copy_lambda_mini
        self.a_copy_lambda_decay_interval = a_copy_lambda_decay_interval
        self.a_copy_lambda_decay_value = a_copy_lambda_decay_value

        self.a_tau = a_tau
        self.a_tau_mini = a_tau_mini
        self.a_tau_decay_interval = a_tau_decay_interval
        self.a_tau_decay_rate = a_tau_decay_rate

        self.state_global_step = 0
        self.action_global_step = 0

    def _update(self, key, value):
        setattr(self, key, value)

    def state_train_tick(self):
        self.state_global_step += 1

        if self.state_global_step % self.s_copy_lambda_decay_interval == 0:
            self.s_copy_lambda = max(self.s_copy_lambda_mini, self.s_copy_lambda - self.s_copy_lambda_decay_value)

        if self.state_global_step % self.s_tau_decay_interval == 0:
            self.s_tau = max(self.s_tau_mini, self.s_tau * self.s_tau_decay_rate)

    def action_train_tick(self):
        self.action_global_step += 1

        if self.action_global_step % self.a_copy_lambda_decay_interval == 0:
            self.a_copy_lambda = max(self.a_copy_lambda_mini, self.a_copy_lambda - self.a_copy_lambda_decay_value)

        if self.action_global_step % self.a_tau_decay_interval == 0:
            self.a_tau = max(self.a_tau_mini, self.a_tau * self.a_tau_decay_rate)

    def dump(self):
        return self.__dict__

    def self_update(self, vts_dict):
        for key, value in vts_dict.items():
            self._update(key, value)

    def update_relay(self):
        vns_dict = VRBotTrainState(VO.s_copy_lambda, VO.state_action_copy_lambda_mini,
                                   VO.copy_lambda_decay_interval, VO.copy_lambda_decay_value,
                                   VO.init_tau, VO.tau_mini, VO.tau_decay_interval, VO.tau_decay_rate,
                                   VO.a_copy_lambda, VO.state_action_copy_lambda_mini,
                                   VO.copy_lambda_decay_interval, VO.copy_lambda_decay_value,
                                   VO.init_tau, VO.tau_mini, VO.tau_decay_interval, VO.tau_decay_rate)
        self.self_update(vns_dict.__dict__)

    @staticmethod
    def load(vns_dict):
        vrbot_state = VRBotTrainState(**vns_dict)
        return vrbot_state


vrbot_train_stage = VRBotTrainState(VO.s_copy_lambda, VO.state_action_copy_lambda_mini,
                                    VO.copy_lambda_decay_interval, VO.copy_lambda_decay_value,
                                    VO.init_tau, VO.tau_mini, VO.tau_decay_interval,
                                    VO.tau_decay_rate,
                                    VO.a_copy_lambda, VO.state_action_copy_lambda_mini,
                                    VO.copy_lambda_decay_interval, VO.copy_lambda_decay_value,
                                    VO.init_tau, VO.tau_mini, VO.tau_decay_interval,
                                    VO.tau_decay_rate)
