# -*- coding: utf-8 -*-

import torch
from resource.module.rnn_decoder import RNNDecoder
from resource.option.dataset_option import DatasetOption
from resource.util.misc import nested_index_select
from resource.module.bs_funcs import expand_if_not_none
from resource.module.bs_funcs import Branch
from resource.module.bs_funcs import MatureBucket


class VRBotBeamSearchDecoder:
    def __init__(self, decoder: RNNDecoder, beam_width, decode_max_step, log_act=True):
        super(VRBotBeamSearchDecoder, self).__init__()
        self.decoder = decoder
        self.beam_width = beam_width
        self.max_decode_step = decode_max_step
        self.log_act = log_act

    def forward(self, hidden, inputs, hs_vectors,
                state_gumbel_prob, state_gumbel_embed,
                action_gumbel_prob, action_gumbel_embed,
                history_lens=None, history_word_indices=None, mask_state_prob=False):
        obs = hidden.size(1)  # obs, which means Origin Batch Size
        rnn_input = hidden.new_ones(obs, 1, dtype=torch.long) * DatasetOption.PreventWord.SOS_ID
        mature_buckets = [MatureBucket(self.beam_width) for _ in range(obs)]

        scores = torch.ones(obs * self.beam_width, dtype=torch.float, device=hidden.device)
        if self.log_act:
            scores = scores * 0.0
        hidden = self.decoder.init_consult_state_action(action_gumbel_embed, hidden,
                                                        False, state_gumbel_embed)

        history = None
        for i in range(self.max_decode_step):
            if i == 0:
                word_output, hidden = self.decoder.decode(rnn_input,
                                                          hidden,
                                                          hs_vectors,
                                                          state_gumbel_prob,
                                                          state_gumbel_embed,
                                                          action_gumbel_prob,
                                                          action_gumbel_embed,
                                                          history_lens=history_lens,
                                                          history_word_indices=history_word_indices,
                                                          mask_state_prob=mask_state_prob)
                topk_logits, word_index = word_output.topk(self.beam_width, dim=-1)

                # B * k, 1
                if self.log_act:
                    scores = scores + topk_logits.reshape(-1)
                else:
                    scores = scores * topk_logits.reshape(-1)
                history = word_index.reshape(-1, 1)
                hidden = expand_if_not_none(hidden, 1, self.beam_width)
                rnn_input = history

                hs_vectors = expand_if_not_none(hs_vectors, 0, self.beam_width)
                state_gumbel_prob = expand_if_not_none(state_gumbel_prob, 0, self.beam_width)
                state_gumbel_embed = expand_if_not_none(state_gumbel_embed, 0, self.beam_width)
                action_gumbel_prob = expand_if_not_none(action_gumbel_prob, 0, self.beam_width)
                action_gumbel_embed = expand_if_not_none(action_gumbel_embed, 0, self.beam_width)
                history_lens = expand_if_not_none(history_lens, 0, self.beam_width)
                history_word_indices = expand_if_not_none(history_word_indices, 0, self.beam_width)
            else:
                word_output, hidden = self.decoder.decode(rnn_input,
                                                          hidden,
                                                          hs_vectors,
                                                          state_gumbel_prob,
                                                          state_gumbel_embed,
                                                          action_gumbel_prob,
                                                          action_gumbel_embed,
                                                          history_lens=history_lens,
                                                          history_word_indices=history_word_indices,
                                                          mask_state_prob=mask_state_prob)
                topk_logits, word_index = word_output.topk(self.beam_width, dim=-1)

                # B' * k  ==> B, k * k
                topk_logits = topk_logits.reshape(obs, -1)
                if self.log_act:
                    # B, k * k
                    scores = scores.unsqueeze(-1).expand(-1, self.beam_width).reshape(obs, -1) + topk_logits
                else:
                    scores = scores.unsqueeze(-1).expand(-1, self.beam_width).reshape(obs, -1) * topk_logits

                rets = self.harvest(scores, history, word_index, obs)

                if rets is not None:
                    scores, harvest_info = rets
                    for bi, gain in harvest_info:
                        mature_buckets[bi].push(gain)

                topk_logits, topk_indices = scores.topk(self.beam_width, 1)
                scores = topk_logits.reshape(-1)

                reshaped_hidden = hidden.reshape(obs, self.beam_width, -1)
                next_hidden = nested_index_select(reshaped_hidden, (topk_indices / self.beam_width).long())
                hidden = next_hidden.reshape(1, obs * self.beam_width, -1)

                expand_history = history.unsqueeze(1).expand(-1, self.beam_width, -1)
                permute_word_output = word_index.permute(0, 2, 1)
                history = torch.cat([expand_history, permute_word_output], dim=2)
                history = history.reshape(obs, self.beam_width * self.beam_width, -1)
                history = nested_index_select(history, topk_indices.long()).reshape(obs * self.beam_width, -1)
                rnn_input = history[:, -1].unsqueeze(1)

        scores, bst_trajectory_index = scores.reshape(obs, self.beam_width).max(-1)
        scores = scores.detach().cpu().numpy().tolist()
        bst_trajectory = nested_index_select(history.reshape(obs, self.beam_width, -1),
                                             bst_trajectory_index.unsqueeze(-1).long()).squeeze(1)
        for i, s in enumerate(scores):
            traj = bst_trajectory[i]
            mature_buckets[i].push(Branch(s, traj, self.max_decode_step))

        bst_trajectory = torch.stack([mb.get_max() for mb in mature_buckets], dim=0)
        return bst_trajectory

    def harvest(self, scores, history, word_index, obs):
        # B, k * k
        word_index = word_index.reshape(obs, -1)
        eos_sign = (word_index == DatasetOption.PreventWord.EOS_ID)
        eos_num = eos_sign.long().sum().item()
        if eos_num <= 0:
            return None

        _, eos_indices = eos_sign.long().reshape(-1).sort(descending=True)
        eos_scores = scores.reshape(-1).index_select(0, eos_indices).cpu().numpy().tolist()

        eos_indices = eos_indices[:eos_num]
        eos_x = (eos_indices / self.beam_width).long()
        batch_index = ((eos_indices / (self.beam_width * self.beam_width)).long()).cpu().numpy().tolist()
        mature_traj = history[eos_x].clone()
        scores = scores.masked_fill(eos_sign, -1e20)
        grow_len = mature_traj.size(1)

        if grow_len < self.max_decode_step:
            padding = mature_traj.new_ones(eos_num,
                                           self.max_decode_step - grow_len,
                                           dtype=torch.long) * DatasetOption.PreventWord.EOS_ID
            mature_traj = torch.cat([mature_traj, padding], dim=-1)
            grow_len += 1

        return scores, [(batch_index[i], Branch(eos_scores[i], mature_traj[i, :], grow_len)) for i in range(eos_num)]
