# -*- coding: utf-8 -*-

import logging
import numpy as np
from resource.option.dataset_option import DatasetOption

vocab_logger = logging.getLogger("main.vocab")


class Vocab(object):
    def __init__(self, word_list, vocab_size):
        super(Vocab, self).__init__()
        self.word_list = word_list[:vocab_size]
        self.word2idx = dict(zip(self.word_list, range(len(self.word_list))))
        assert len(self.word_list) == len(self.word2idx), "{} != {}".format(len(self.word_list), len(self.word2idx))
        self.vocab_size = len(self.word_list)
        self.unk_id = self.word2idx.get(DatasetOption.PreventWord.UNK, None)
        self.pad_id = self.word2idx.get(DatasetOption.PreventWord.PAD, None)
        assert self.vocab_size == len(self.word2idx), "vocab size not equal to word_list"
        vocab_logger.info("Vocab, size={}, {} ... {}".format(self.vocab_size,
                                                             ",".join(self.word_list[:3]),
                                                             ",".join(self.word_list[-5:])))

    def item_in(self, word):
        return word in self.word2idx

    def __len__(self):
        return self.vocab_size

    def word2index(self, word):
        if isinstance(word, str):
            return self.word2idx.get(word, self.unk_id)
        elif isinstance(word, list):
            return [self.word2index(w) for w in word]
        else:
            raise ValueError("wrong type {}".format(type(word)))

    def index2word(self, index):
        if isinstance(index, int):
            if index < len(self.word_list):
                return self.word_list[index]
            else:
                raise ValueError("{} is out of {}".format(index, len(self.word_list)))
        elif isinstance(index, np.ndarray):
            index = index.tolist()
            return [self.index2word(i) for i in index]
        elif isinstance(index, list):
            return [self.index2word(i) for i in index]
        else:
            raise ValueError("wrong type {}".format(type(index)))
