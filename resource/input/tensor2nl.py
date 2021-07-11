# -*- coding: utf-8 -*-

import torch
from resource.input.vocab import Vocab
from resource.option.dataset_option import DatasetOption


class TensorNLInterpreter:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def interpret_tensor2nl(self, tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()

        words = self.vocab.index2word(tensor)  # B, T
        temp = []
        for word in words:
            eos_index = len(word)
            try:
                eos_index = word.index(DatasetOption.PreventWord.EOS)
            except (Exception, ):
                pass

            word = word[:eos_index]
            temp.append(" ".join(word))

        words = temp
        return words

    @staticmethod
    def word2sentence(words):
        sentences = [" ".join(word) for word in words]
        return sentences
