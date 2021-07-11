# -*- coding: utf-8 -*-

import pandas as pd
from resource.input.vocab import Vocab
from resource.option.dataset_option import DatasetOption as DO
from resource.tools.load_data import read_sessions_from_zip_filename


class DataProcessor:
    def __init__(self, task):
        super(DataProcessor, self).__init__()
        self.session_zip_filename = DO.session_zip_filename
        self.vocab_csv_filename = DO.vocab_csv_filename
        self.task = task

    def get_formatted_data(self):
        train_sessions, test_sessions, valid_sessions = self.get_session()
        word_vocab, know_vocab, glo2loc, loc2glo, word_vocab_size, inner_vocab_size = self.get_vocab()

        return train_sessions, test_sessions, valid_sessions, \
               word_vocab, know_vocab, glo2loc, loc2glo, word_vocab_size, inner_vocab_size

    def get_session(self):
        def _extract_pure_sessions(sessions):
            if self.task.startswith("meddg"):
                pure_sessions = []
                for session in sessions:
                    pure_session = []
                    state = []
                    for sentence in session["dialogues"]:
                        keywords = sentence["keywords"]
                        pure_sentence = list()
                        # [sentence, type, state / action]
                        pure_sentence.append(sentence["tokens"])
                        pure_sentence.append(sentence.get("type", None))

                        if sentence["role"] == "doctor":
                            pure_sentence.append(keywords[:DO.action_num])  # action -
                            to_add_state = [k for k in keywords if k not in state]
                            state = state + to_add_state
                            state = state[-DO.state_num:]
                        else:
                            to_add_state = [k for k in keywords if k not in state]
                            state = state + to_add_state
                            state = state[-DO.state_num:]
                            pure_sentence.append(state)  # state -

                        pure_session.append(pure_sentence)
                    pure_sessions.append(pure_session)
            else:
                # [sentence, type]
                pure_sessions = [[(sentence["tokens"], sentence.get("type", None)) for sentence in session["dialogues"]]
                                 for session in sessions]
            return pure_sessions

        train_sessions = list(
            read_sessions_from_zip_filename(
                self.session_zip_filename.format(dataset="{}_train".format(self.task))).values())
        test_sessions = list(
            read_sessions_from_zip_filename(
                self.session_zip_filename.format(dataset="{}_test".format(self.task))).values())
        valid_sessions = list(
            read_sessions_from_zip_filename(
                self.session_zip_filename.format(dataset="{}_valid".format(self.task))).values())

        train_sessions = _extract_pure_sessions(train_sessions)
        test_sessions = _extract_pure_sessions(test_sessions)
        valid_sessions = _extract_pure_sessions(valid_sessions)

        return train_sessions, test_sessions, valid_sessions

    def get_vocab(self):
        reserved_words = [DO.PreventWord.SOS, DO.PreventWord.EOS, DO.PreventWord.PAD, DO.PreventWord.UNK,
                          DO.PreventWord.SENTENCE_SPLITER]
        reserved_words += ["[CHITCHAT]", "[ASK_SYMPTOM]", "[DIAGNOSIS]", "[PRESCRIBE]"]
        vocab_df = pd.read_csv(self.vocab_csv_filename.format(dataset=self.task))
        word_vocab_items = reserved_words + [str(x) for x in list(vocab_df["Word"])]
        word_vocab = Vocab(word_vocab_items, DO.vocab_size)

        know_reserved_words = [DO.PreventWord.SOS, DO.PreventWord.EOS, DO.PreventWord.PAD, DO.PreventWord.UNK]
        know_vocab_items = know_reserved_words + [str(x) for x in list(vocab_df[vocab_df["Is_know"] > 0]["Word"])]
        know_vocab = Vocab(know_vocab_items, len(know_vocab_items))
        DO.know_vocab_size = len(know_vocab_items)

        glo2loc = []
        for word in word_vocab.word_list:
            glo2loc.append(know_vocab.word2index(word))

        loc2glo = []
        for know in know_vocab.word_list:
            loc2glo.append(word_vocab.word2index(know))

        word_vocab_size = len(glo2loc)
        inner_vocab_size = len(loc2glo)
        return word_vocab, know_vocab, glo2loc, loc2glo, word_vocab_size, inner_vocab_size
