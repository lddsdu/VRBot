# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum, unique
from resource.module.history_encoder import HistoryEncoder
from resource.module.prior_state_tracker import PriorStateTracker
from resource.module.posterior_state_tracker import PosteriorStateTracker
from resource.module.prior_policy_network import PriorPolicyNetwork
from resource.module.posterior_policy_network import PosteriorPolicyNetwork
from resource.module.rnn_encoder import RNNEncoder
from resource.module.rnn_decoder import RNNDecoder
from resource.module.vrbot_beam_search_decoder import VRBotBeamSearchDecoder
from resource.module.gumbel_softmax import GumbelSoftmax
from resource.module.embedder import Embedder
from resource.util.loc_glo_trans import LocGloInterpreter
from resource.option.vrbot_option import VRBotOption as VO
from resource.option.dataset_option import DatasetOption as DO
from resource.option.train_option import TrainOption as TO
from resource.util.misc import one_hot_scatter
from resource.model.vrbot_train_state import vrbot_train_stage
from resource.input.graph_db import GraphDB


@unique
class TRAIN(Enum):
    TRAIN_POLICY = 1
    TRAIN_STATE = 2
    TRAIN_DUAL = 3


class VRBot(nn.Module):
    def __init__(self,
                 loc2glo_tensor,
                 state_num,
                 action_num,
                 hidden_dim,
                 inner_vocab_size,
                 vocab_size,
                 response_max_len,
                 embed_dim,
                 lg_interpreter: LocGloInterpreter,
                 gen_strategy,
                 with_copy,
                 graph_db: GraphDB,
                 beam_width=1):
        super(VRBot, self).__init__()
        self.know2word_tensor = loc2glo_tensor
        self.state_num = state_num
        self.action_num = action_num
        self.hidden_dim = hidden_dim
        self.inner_vocab_size = inner_vocab_size
        self.vocab_size = vocab_size
        self.response_max_len = response_max_len
        self.embed_dim = embed_dim
        self.lg_interpreter = lg_interpreter
        self.gen_strategy = gen_strategy
        self.graph_db = graph_db
        self.beam_width = beam_width

        self.embedder = Embedder(vocab_size, embed_dim)
        self.encoder4post = HistoryEncoder(self.embed_dim,
                                           self.hidden_dim,
                                           embedder=self.embedder)
        self.encoder4state = HistoryEncoder(self.embed_dim,
                                            self.hidden_dim,
                                            embedder=self.embedder)
        self.encoder4action = HistoryEncoder(self.embed_dim,
                                             self.hidden_dim,
                                             embedder=self.embedder)
        self.know_encoder = RNNEncoder(self.embed_dim,
                                       self.hidden_dim,
                                       self.hidden_dim // 2,
                                       num_layers=1,
                                       bidirectional=True,
                                       dropout=0,
                                       embedder=self.embedder)
        self.encoder = HistoryEncoder(self.embed_dim,
                                      self.hidden_dim,
                                      embedder=self.embedder)

        self.intention_embedding = nn.Linear(4, hidden_dim)
        self.state_gumbel_softmax = GumbelSoftmax(normed=True)
        self.intention_gumbel_softmax = GumbelSoftmax(normed=True)
        self.action_gumbel_softmax = GumbelSoftmax(normed=True)

        self.pst = PriorStateTracker(state_num, hidden_dim, inner_vocab_size, embed_dim,
                                     self.embedder, lg_interpreter,
                                     gen_strategy, loc2glo_tensor, with_copy)

        self.qst = PosteriorStateTracker(state_num, hidden_dim, inner_vocab_size, embed_dim,
                                         self.embedder, lg_interpreter,
                                         gen_strategy, loc2glo_tensor, with_copy)

        self.ppn = PriorPolicyNetwork(action_num, hidden_dim, inner_vocab_size, embed_dim,
                                      self.embedder, lg_interpreter,
                                      gen_strategy, with_copy, self.graph_db,
                                      loc2glo_tensor, self.intention_gumbel_softmax)

        self.qpn = PosteriorPolicyNetwork(action_num, hidden_dim, inner_vocab_size, embed_dim,
                                          self.embedder, lg_interpreter,
                                          gen_strategy, with_copy, loc2glo_tensor)

        self.decoder = RNNDecoder(self.embed_dim, self.hidden_dim, self.vocab_size, self.response_max_len,
                                  embedder=self.embedder, attn_history_sentence=VO.attn_history_sentence,
                                  with_state_know=VO.with_state_know, with_action_know=VO.with_action_know,
                                  with_copy=VO.with_copy)

        if self.beam_width > 1:
            self.beam_decoder = VRBotBeamSearchDecoder(self.decoder, TO.beam_width, TO.beam_decode_max_step)
        else:
            self.beam_decoder = self.decoder

    @staticmethod
    def encode_sequence(encoder, seq, seq_len, pv_hidden=None):
        # Encoder encode
        # B, T, H | 1, B, H
        pv_r_u_enc, hidden = encoder.forward(seq, seq_len, pv_hidden)
        return pv_r_u_enc, hidden

    def forward(self, pv_state, pv_hidden, pv_r_u, pv_r_u_len,
                gth_intention=None, gth_r=None, gth_r_len=None,
                gth_action=None, gth_state=None,
                train_stage=TRAIN.TRAIN_STATE, supervised=False):
        if supervised:
            assert gth_intention is not None, "gth_intention should not be None"
            assert gth_r is not None, "gth_r should not be None"
            assert gth_r_len is not None, "gth_r_len should not be None"
            assert gth_action is not None, "gth_action should not be None"
            assert gth_state is not None, "gth_state should not be None"

        if self.training:
            rets = self.forward_train(pv_state, pv_hidden, pv_r_u, pv_r_u_len,
                                      gth_intention, gth_r, gth_r_len,
                                      train_stage=train_stage, gth_action=gth_action,
                                      gth_state=gth_state, supervised=supervised)

            return rets
        else:
            rets = self.forward4infer(pv_state, pv_hidden, pv_r_u, pv_r_u_len)
            gen_log_probs, state_index, action_index, hidden4post = rets

            return gen_log_probs, state_index, action_index, hidden4post

    def forward_train(self, pv_state, pv_hidden, pv_r_u, pv_r_u_len,
                      gth_intention, gth_r, gth_r_len,
                      train_stage=TRAIN.TRAIN_STATE,
                      gth_action=None, gth_state=None,
                      supervised=False):
        # *** Context Encoder ***
        bi_pv_hidden = self.hidden_bidirectional(pv_hidden)
        pv_r_u_enc4post, hidden4post = self.encode_sequence(self.encoder4post,
                                                            pv_r_u,
                                                            pv_r_u_len,
                                                            bi_pv_hidden)

        pv_r_u_enc4state, hidden4state = self.encode_sequence(self.encoder4state,
                                                              pv_r_u,
                                                              pv_r_u_len,
                                                              bi_pv_hidden)

        pv_r_u_enc4action, hidden4action = self.encode_sequence(self.encoder4action,
                                                                pv_r_u,
                                                                pv_r_u_len,
                                                                bi_pv_hidden)
        r_enc, r_hidden = self.encode_sequence(self.encoder,
                                               gth_r,
                                               gth_r_len,
                                               self.hidden_bidirectional(hidden4post.detach()))

        post_state_prob, state_gumbel_prob = self.qst.forward(hidden4state, pv_state,
                                                              pv_r_u, pv_r_u_enc4state,
                                                              gth_r, r_enc, gth_state,
                                                              supervised=supervised)

        if state_gumbel_prob is None:
            state_gumbel_prob = self.state_gumbel_softmax_sampling(post_state_prob)

        state_gumbel_embed = self.gumbel_prob_embed(state_gumbel_prob)
        state_index = VRBot.sampling(state_gumbel_prob, strategy="prob")
        state_gumbel_word_prob = self.know_prob2word_prob(state_gumbel_prob, mask_pad=False)

        prior_intention, prior_action_prob, action_gumbel_prob \
            = self.ppn.forward(hidden4action,
                               state_index,
                               gth_intention,
                               pv_r_u_enc4action,
                               pv_r_u_len,
                               gth_action=gth_action,
                               supervised=supervised)

        if action_gumbel_prob is None:
            action_gumbel_prob = self.action_gumbel_softmax_sampling(prior_action_prob)

        action_gumbel_embed = self.gumbel_prob_embed(action_gumbel_prob)
        action_index = VRBot.sampling(action_gumbel_prob, strategy="prob")
        action_gumbel_word_prob = self.know_prob2word_prob(action_gumbel_prob, mask_pad=False)

        prior_state_prob, prior_state_gumbel_prob = self.pst.forward(hidden4state, pv_state,
                                                                     pv_r_u, pv_r_u_enc4state,
                                                                     gth_state=gth_state, supervised=supervised)

        gen_log_probs1 = self.decoder.forward(hidden4post,
                                              gth_r.detach()[:, :-1],
                                              pv_r_u_enc4post,
                                              state_gumbel_word_prob,
                                              state_gumbel_embed,
                                              action_gumbel_word_prob,
                                              action_gumbel_embed,
                                              pv_r_u_len,
                                              pv_r_u,
                                              mask_action_prob=True if train_stage == TRAIN.TRAIN_POLICY else False)

        gen_log_probs2 = None
        post_action_prob = None
        post_intention = None

        if train_stage == TRAIN.TRAIN_POLICY:
            if prior_state_gumbel_prob is None:
                prior_state_gumbel_prob = self.state_gumbel_softmax_sampling(prior_state_prob)

            state_gumbel_embed = self.gumbel_prob_embed(prior_state_gumbel_prob)
            state_index = VRBot.sampling(prior_state_gumbel_prob, strategy="prob")
            state_gumbel_word_prob = self.know_prob2word_prob(prior_state_gumbel_prob, mask_pad=False)

            post_intention, post_action_prob, action_gumbel_prob = self.qpn.forward(hidden4action,
                                                                                    state_index,
                                                                                    pv_r_u_enc4action,
                                                                                    pv_r_u_len,
                                                                                    gth_r, r_enc,
                                                                                    mask_gen=False,
                                                                                    gth_action=gth_action,
                                                                                    supervised=supervised)
            if action_gumbel_prob is None:
                action_gumbel_prob = self.action_gumbel_softmax_sampling(post_action_prob)  # count step

            post_action_gumbel_embed = self.gumbel_prob_embed(action_gumbel_prob)
            post_action_index = VRBot.sampling(action_gumbel_prob, strategy="prob")
            post_action_gumbel_word_prob = self.know_prob2word_prob(action_gumbel_prob)
            action_gumbel_prob = action_gumbel_prob

            gen_log_probs2 = self.decoder.forward(hidden4post,
                                                  gth_r.detach()[:, :-1],
                                                  pv_r_u_enc4post,
                                                  state_gumbel_word_prob,
                                                  state_gumbel_embed,
                                                  post_action_gumbel_word_prob,
                                                  post_action_gumbel_embed,
                                                  pv_r_u_len,
                                                  pv_r_u,
                                                  state_detach=True,
                                                  mask_state_prob=VO.mask_state_prob)

        return [gen_log_probs1, gen_log_probs2,
                post_state_prob, prior_state_prob, state_gumbel_prob,
                post_action_prob, prior_action_prob, action_gumbel_prob,
                post_intention, prior_intention, hidden4post]

    def forward4infer(self, pv_state, pv_hidden, pv_r_u, pv_r_u_len):
        bi_pv_hidden = self.hidden_bidirectional(pv_hidden)
        pv_r_u_enc4post, hidden4post = self.encode_sequence(self.encoder4post,
                                                            pv_r_u,
                                                            pv_r_u_len,
                                                            bi_pv_hidden)

        pv_r_u_enc4state, hidden4state = self.encode_sequence(self.encoder4state,
                                                              pv_r_u,
                                                              pv_r_u_len,
                                                              bi_pv_hidden)

        pv_r_u_enc4action, hidden4action = self.encode_sequence(self.encoder4action,
                                                                pv_r_u,
                                                                pv_r_u_len,
                                                                bi_pv_hidden)

        prior_state_prob, _ = self.pst.forward(hidden4state, pv_state, pv_r_u, pv_r_u_enc4state)
        state_index = VRBot.sampling(prior_state_prob, strategy="prob")
        state_word_index = self.lg_interpreter.loc2glo(state_index)
        state_word_prob = one_hot_scatter(state_word_index, self.vocab_size, dtype=torch.float)
        state_embed = self.embedder(state_word_index)

        prior_gumbel_intention, prior_action_prob, _ = self.ppn.forward(hidden4action,
                                                                        state_index,
                                                                        None,
                                                                        pv_r_u_enc4action,
                                                                        pv_r_u_len)

        ranking_floor = 1
        if ranking_floor > 0:
            values, indices = prior_action_prob.topk(ranking_floor, -1)
            prior_action_prob = prior_action_prob.new_zeros(prior_action_prob.shape)
            values = values / values.sum(-1).unsqueeze(-1)
            prior_action_prob.scatter_(-1, indices, values)

        action_index = VRBot.sampling(prior_action_prob, strategy="max")
        action_word_index = self.lg_interpreter.loc2glo(action_index)
        action_word_prob = one_hot_scatter(action_word_index, self.vocab_size, dtype=torch.float)
        action_embed = self.embedder(action_word_index)

        gen_log_probs = self.beam_decoder.forward(hidden4post,
                                                  None,
                                                  pv_r_u_enc4post,
                                                  state_word_prob,
                                                  state_embed,
                                                  action_word_prob,
                                                  action_embed,
                                                  pv_r_u_len,
                                                  pv_r_u,
                                                  mask_state_prob=VO.mask_state_prob)

        if len(gen_log_probs.shape) == 3:
            bst_trajectory = torch.argmax(gen_log_probs, -1)
        else:
            bst_trajectory = gen_log_probs

        return [bst_trajectory, state_index, action_index, hidden4post]

    def gumbel_prob_embed(self, state_gumbel_prob):
        B, S, K = state_gumbel_prob.shape
        # K, E
        know_embedding = self.embedder(self.know2word_tensor)
        state_gumbel_embed = torch.bmm(state_gumbel_prob.reshape(B * S, 1, K),
                                       know_embedding.unsqueeze(0).expand(B * S, K, self.embed_dim))
        state_gumbel_embed = state_gumbel_embed.reshape(B, S, self.embed_dim)
        return state_gumbel_embed

    def know_prob2word_prob(self, state_prob, mask_pad=False):
        B, S, K = state_prob.shape
        state_word_prob = torch.zeros(B, S, self.vocab_size, dtype=torch.float, device=state_prob.device)
        index = self.know2word_tensor.unsqueeze(0).unsqueeze(0).expand(B, S, K)
        if mask_pad:
            state_word_prob.scatter_(-1, index[:, :, 1:], state_prob[:, :, 1:])
        else:
            state_word_prob.scatter_(-1, index, state_prob)
        return state_word_prob

    @staticmethod
    def sampling(state_prob, strategy="prob", return_sparse=False):
        state_prob_shape = state_prob.shape

        if strategy == "prob":
            # B, S, K
            state_prob_cumsum = state_prob.detach().cumsum(-1)[..., :-1]
            random_seed = torch.rand(list(state_prob_shape[:-1]) + [1], device=TO.device)
            selected_index = (random_seed > state_prob_cumsum).long().sum(-1)
        elif strategy == "max":
            selected_index = state_prob.argmax(-1)
        else:
            raise RuntimeError("error strategy {}, should be `prob` or `max`")

        if not return_sparse:
            return selected_index

        selected_index = selected_index.unsqueeze(-1)
        sparse_selected_index = torch.zeros(*state_prob_shape, device=TO.device).scatter(-1, selected_index, 1.0)
        return selected_index, sparse_selected_index

    def state_gumbel_softmax_sampling(self, probs):
        gumbel_probs = self.state_gumbel_softmax.forward(probs, vrbot_train_stage.s_tau)
        return gumbel_probs

    def action_gumbel_softmax_sampling(self, probs):
        gumbel_probs = self.action_gumbel_softmax.forward(probs, vrbot_train_stage.a_tau)
        return gumbel_probs

    def init4state_train(self):
        self.freeze_module(self.ppn)
        self.unfreeze_module(self.qst)
        self.unfreeze_module(self.pst)

    def init4policy_train(self):
        self.freeze_module(self.pst)
        self.unfreeze_module(self.qpn)
        self.unfreeze_module(self.ppn)

    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_module(module):
        for param in module.parameters():
            param.requires_grad = True

    @staticmethod
    def reconstruct_prob(hypothesis, target):
        flat_hypothesis = hypothesis.reshape(-1, hypothesis.size(-1))
        flat_target = target.reshape(-1)

        reconstruct_prob = torch.gather(flat_hypothesis, 1, flat_target.unsqueeze(1))
        reconstruct_prob = reconstruct_prob.reshape(hypothesis.size(0), hypothesis.size(1))
        return reconstruct_prob

    @staticmethod
    def kl_loss(prior_dist, posterior_dist, weight_tensor_mean=None):
        flat_prior_dist = prior_dist.reshape(-1, prior_dist.size(-1))
        flat_posterior_dist = posterior_dist.reshape(-1, posterior_dist.size(-1))
        bias = 1e-24

        if weight_tensor_mean is None:
            kl_div = F.kl_div((flat_prior_dist + bias).log(), flat_posterior_dist, reduce=False).sum(-1)
            kl_div = kl_div.mean()
        else:
            tmp1 = (((flat_posterior_dist + bias).log() - (
                    flat_prior_dist + bias).log()) * flat_posterior_dist)  # B * S, K
            tmp2 = tmp1 * weight_tensor_mean.unsqueeze(0)  # B * S, K
            kl_div = tmp2.sum(-1).mean()

        return kl_div

    @staticmethod
    def nll_loss(hypothesis, target):
        bias = 1e-24
        B, T = target.shape
        hypothesis = hypothesis.reshape(-1, hypothesis.size(-1))
        target = target.reshape(-1)
        nll_loss = F.nll_loss(hypothesis, target, ignore_index=DO.PreventWord.PAD_ID, reduce=False)
        not_ignore_tag = (target != DO.PreventWord.PAD_ID).float()
        not_ignore_num = not_ignore_tag.reshape(B, T).sum(-1)
        sum_nll_loss = nll_loss.reshape(B, T).sum(-1)
        nll_loss_vector = sum_nll_loss / (not_ignore_num + bias)
        nll_loss = nll_loss_vector.mean()
        return nll_loss, nll_loss_vector.detach()

    @staticmethod
    def state_nll(hypothesis, target):
        eps = 1e-6
        nll_loss = F.nll_loss(torch.log(hypothesis.reshape(-1, hypothesis.size(-1)) + eps), target.reshape(-1))
        return nll_loss

    @staticmethod
    def cross_entropy_loss(hypothesis, target):
        eps = 1e-24
        loss = - (target * torch.log(hypothesis + eps)).sum(-1).mean()
        return loss

    @staticmethod
    def sparse_ce(hypothesis, target):
        hypothesis = hypothesis.reshape(-1, hypothesis.size(-1))
        target = target.reshape(-1)
        logits = hypothesis.gather(1, target.unsqueeze(-1))
        neg_logits = - torch.log(logits)
        return neg_logits.mean()

    @staticmethod
    def hidden_bidirectional(hidden):
        B, H = hidden.size(1), hidden.size(2)
        hidden = hidden.reshape(1, B, H // 2, 2).permute(0, 3, 1, 2).reshape(2, B, H // 2)
        return hidden
