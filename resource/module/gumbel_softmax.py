# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_num", type=int, default=5000)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_epoch", type=int, default=10000)
    cfg = parser.parse_args()
    return cfg


def one_hot(indice, num_classes):
    I = torch.eye(num_classes).to(indice.device)
    T = I[indice]
    T.requires_grad = False
    return T


class GumbelSoftmax(nn.Module):
    def __init__(self, normed=True, origin_version=True, rep_penalize=False, reoper=10):
        super(GumbelSoftmax, self).__init__()
        self.normed = normed
        self.origin_version = origin_version
        self.eps = 1e-24
        self.step = 0
        self.rep_penalize = rep_penalize
        self.reoper = reoper

    def forward(self, inp, tau):
        if self.normed:
            inp = torch.log(inp + self.eps)

        if not self.origin_version:
            device = inp.device
            gk = -torch.log(-torch.log(torch.rand(inp.shape, device=device)))
            out = torch.softmax((inp + gk) / tau, dim=-1)
        else:
            if self.rep_penalize:
                expand_inp = inp.unsqueeze(1).expand(-1, self.reoper, -1, -1)  # B, 10, S, T
                out = torch.nn.functional.gumbel_softmax(expand_inp, tau=tau)  # B, 10, S, T
                max_index = out.argmax(-1)  # B, 10, S
                max_index = max_index.reshape(max_index.size(0), -1)  # B, 10 * S
                max_index = max_index.detach().cpu().tolist()  # B, 10 * S

                def find_index(rand_value, prob_list):
                    ceil = np.cumsum(prob_list[:-1])
                    index = (rand_value > ceil).astype(np.long).sum()
                    return int(index)

                batch_selected_indexs = []  # B, S,
                for b in range(expand_inp.size(0)):
                    c = Counter()
                    c.update(max_index[b])
                    index2prob = dict([(x, 1 / y) for x, y in c.most_common()])
                    probs = [index2prob[i] for i in max_index[b]]
                    probs_sum = sum(probs)
                    normalized_probs = [x / probs_sum for x in probs]
                    # S,
                    indexs = [find_index(random.random(), normalized_probs) for _ in range(expand_inp.size(2))]
                    batch_selected_indexs.append(indexs)

                B, _, S, T = out.shape
                flat_out = out.reshape(-1, T)  # B * 10 * S, T
                indexs = torch.tensor(batch_selected_indexs, device=inp.device).reshape(-1)  # B * S
                indexs = indexs + torch.arange(B, device=inp.device).unsqueeze(1).expand(-1, self.reoper).reshape(
                    -1) * self.reoper * S
                flat_out = flat_out.index_select(0, indexs)  # B * S, T
                out = flat_out.reshape(B, S, -1)
            else:
                out = torch.nn.functional.gumbel_softmax(inp, tau=tau)
        return out


class Argmax(nn.Module):
    def __init__(self):
        super(Argmax, self).__init__()

    def forward(self, inp):
        return torch.argmax(inp, dim=-1)


class GUMBEL(nn.Module):
    def __init__(self, sample_num, hidden_size, is_train=False, gumbel_act=True):
        super(GUMBEL, self).__init__()
        self.is_train = is_train
        self.gumbel_act = gumbel_act
        self.embedding_layer = nn.Linear(sample_num, hidden_size)
        self.pred_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, sample_num))
        self.train_act1 = nn.Softmax(dim=-1)
        self.train_act2 = GumbelSoftmax()
        self.test_act3 = Argmax()

    def get_act(self):
        act = self.test_act3 if not self.is_train else (self.train_act2 if self.gumbel_act else self.train_act1)
        return act

    def forward(self, sample):
        sample = sample.cuda()
        sample_embedding = self.embedding_layer(sample)
        pred = self.pred_layer(sample_embedding)  # B, sample_num
        # pred_norm = torch.softmax(pred, dim=-1)
        ret = self.get_act()(pred)

        return ret


def test():
    gumbel_softmax = GumbelSoftmax(normed=True)
    a = torch.tensor([0.1, 0.1, 0.5, 0.1, 0.2])
    print("origin a:")
    print(a)

    gumbel_softmax.update_tau(100.0)
    print("tau = 100.0")
    print(gumbel_softmax.forward(a))

    gumbel_softmax.update_tau(10.0)
    print("tau = 10.0")
    print(gumbel_softmax.forward(a))

    gumbel_softmax.update_tau(5.0)
    print("tau = 5.0")
    print(gumbel_softmax.forward(a))

    gumbel_softmax.update_tau(2.0)
    print("tau = 2.0")
    print(gumbel_softmax.forward(a))

    gumbel_softmax.update_tau(1.0)
    print("tau = 1.0")
    print(gumbel_softmax.forward(a))

    gumbel_softmax.update_tau(0.5)
    print("tau = 0.5")
    print(gumbel_softmax.forward(a))

    gumbel_softmax.update_tau(0.1)
    print("tau = 0.1")
    print(gumbel_softmax.forward(a))


def main():
    cfg = config()

    test()

    sample_num, hidden_size, batch_size, train_epoch = \
        [getattr(cfg, attr) for attr in ["sample_num", "hidden_size", "batch_size", "train_epoch"]]

    model = GUMBEL(sample_num, hidden_size, is_train=True, gumbel_act=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("训练之前")
    do_test(model, sample_num)

    # Train
    pbar = tqdm(list(range(train_epoch)))
    print_interval = train_epoch / 10
    for e in pbar:
        sample_index = [random.randint(0, sample_num - 1) for _ in range(batch_size)]
        sample_index = torch.tensor(sample_index).long().cuda()
        batch_sample = one_hot(sample_index, sample_num)  # B, N

        pred = model.forward(batch_sample)  # B, N
        # loss = ((batch_sample.detach() - pred) ** 2).mean()
        loss = torch.nn.functional.nll_loss(torch.log(pred + 1e-24), sample_index)

        pbar.set_description("{} : {:.6f}".format(e, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % print_interval == 0:
            do_test(model, sample_num)

    # Test
    print("训练之后")
    do_test(model, sample_num)


def do_test(model, sample_num):
    model.is_train = False

    test = list(range(sample_num))
    target = np.asarray(test)
    test = torch.tensor(test).long()
    test = one_hot(test, sample_num)  # B, sample_num
    model.test()
    with torch.no_grad():
        preds = model(test)  # test : B, sample_num      ret : B,
    pred_np = preds.cpu().numpy()
    correct = np.sum(np.asarray(pred_np == target, dtype=np.int))
    model.is_train = True
    model.train()
    print("{} / {} correct".format(correct, sample_num))


if __name__ == '__main__':
    main()
