# -*- coding: utf-8 -*-

import torch
import random
from enum import Enum
from tqdm import tqdm
from multiprocessing import Process, Queue
from resource.option.dataset_option import DatasetOption as DO
from resource.option.train_option import TrainOption as TO
from resource.input.vocab import Vocab
from resource.util.misc import clip_pad_sentence

intention2str = ["[CHITCHAT]", "[ASK_SYMPTOMS]", "[DIAGNOSIS]", "[PRESCRIBE]"]


class MetaType(Enum):
    FINISHED = 1
    UNFINISHED = 2


class MetaInfo:
    def __init__(self, meta_type: MetaType, info=None):
        self.meta_type = meta_type
        self.info = info


class SessionProcessor:
    def __init__(self, word_vocab: Vocab, inner_vocab: Vocab,
                 pv_r_u_max_len, r_max_len):
        self.word_vocab = word_vocab
        self.inner_vocab = inner_vocab
        self.pv_r_u_max_len = pv_r_u_max_len
        self.r_max_len = r_max_len

    def process(self, session, supervised=False):
        session_segs = []  # session_segs is used for final training/test data
        history = []  # history is a temporary buffer

        for i in range(0, len(session), 2):
            pv_r_u = [x[0] for x in session[max(0, i - 1):i + 1]]  # r_{t-1}, u_{t}
            history += pv_r_u

            response = session[i + 1]
            response_type = response[1]
            response = response[0]
            if response_type is None:
                continue

            origin_history_len = len(history)
            if origin_history_len <= (2 + TO.cache_turn):
                history = [x[0] for x in session[max(0, i - 1 - TO.cache_turn):i + 1 - origin_history_len]] + history

            intention_str = "[{}]".format(response_type.upper())
            intention_idx = intention2str.index(intention_str)
            intention_spa = [0 if intention_idx != i else 1 for i in range(4)]  # one-hot

            if len(history) >= 2:
                tmp = []
                for idx, sentence in enumerate(history[:-1]):
                    tmp += sentence + [DO.PreventWord.SENTENCE_SPLITER]
                tmp += history[-1]
                history = tmp
            else:
                history = history[0]

            history = history + [intention_str]
            history, history_len = clip_pad_sentence(history, DO.pv_r_u_max_len,
                                                     sos=DO.PreventWord.SOS, eos=DO.PreventWord.EOS,
                                                     save_prefix=False, return_length=True)
            response, response_len = clip_pad_sentence(response, DO.r_max_len + 1,
                                                       sos=DO.PreventWord.SOS, eos=DO.PreventWord.EOS,
                                                       save_prefix=True, return_length=True)

            history_index = [self.word_vocab.word2index(w) for w in history]
            r_index = [self.word_vocab.word2index(w) for w in response]

            if (len(session[i]) >= 3) and supervised:
                state_label = [self.inner_vocab.word2index(x) for x in session[i][2]]
                state_label = state_label + [self.inner_vocab.pad_id] * (DO.state_num - len(state_label))
                action_label = [self.inner_vocab.word2index(x) for x in session[i + 1][2]]
                action_label = self.pad_label(action_label, self.inner_vocab.pad_id)

                session_segs.append([history_index, history_len, r_index, response_len, state_label,
                                     intention_spa, action_label])
            else:
                session_segs.append([history_index, history_len, r_index, response_len, intention_spa])

            history = []

        return session_segs

    @staticmethod
    def pad_label(action_label, pad_idx):
        if len(action_label) == 0:
            action_label = [pad_idx] * DO.action_num
        elif len(action_label) < DO.action_num:
            pad_action_num = DO.action_num - len(action_label)
            pad_action_label = [random.choice(action_label) for _ in range(pad_action_num)]
            action_label = action_label + pad_action_label

        return action_label


class TaskAllocator(Process):
    def __init__(self, task_queue: Queue, sessions, worker_num, show_process_bar=False, identity=""):
        super(TaskAllocator, self).__init__()
        self.task_queue = task_queue
        self.sessions = sessions
        self.worker_num = worker_num
        self.show_process_bar = show_process_bar
        self.identity = identity

    def run(self) -> None:
        if self.show_process_bar:
            pbar = tqdm(self.sessions)
        else:
            pbar = self.sessions

        for idx, session in enumerate(pbar):
            if (not self.show_process_bar) and ((idx + 1) % 10 == 0):
                print("[TaskAllocator] allocate {}/{} {} sessions".format(idx + 1,
                                                                          len(self.sessions),
                                                                          self.identity.upper()))
            self.task_queue.put(MetaInfo(MetaType.UNFINISHED, session))

        for _ in range(self.worker_num):
            # finish tag
            self.task_queue.put(MetaInfo(MetaType.FINISHED, None))


class TaskWorker(Process):
    def __init__(self, queue: Queue, output_queue: Queue,
                 session_processor: SessionProcessor, supervised=False):
        super(TaskWorker, self).__init__()
        self.input_queue = queue
        self.output_queue = output_queue
        self.session_processor = session_processor
        self.supervised = supervised

    def run(self) -> None:

        while True:
            meta_info = self.input_queue.get()
            if meta_info.meta_type == MetaType.FINISHED:
                break

            session = meta_info.info
            processed_session = self.session_processor.process(session, supervised=self.supervised)

            if len(processed_session) > 0:
                self.output_queue.put(processed_session)


class MultiProcessSessionDataLoader:
    def __init__(self, sessions, batch_size, session_processor: SessionProcessor,
                 number_workers=3, show_process_bar=False, mode="train", shuffle=False,
                 task_queue_max_size=50, processed_queue_max_size=50, device="cpu:0",
                 dtypes=None, supervised=False):

        self.sessions = sessions
        self.batch_size = batch_size
        self.session_processor = session_processor
        self.number_workers = number_workers
        self.show_process_bar = show_process_bar
        self.supervised = supervised

        assert mode in ("train", "valid", "test"), \
            "SessionDataSet Mode should be train or test, but valued {}".format(mode)

        self.mode = mode
        self.shuffle = shuffle
        self.history_sessions = [[] for _ in range(self.batch_size)]
        self.sunset = False
        self.device = device
        self.dtypes = dtypes
        self.task_queue = Queue(maxsize=task_queue_max_size)
        self.processed_queue = Queue(maxsize=processed_queue_max_size)

    @staticmethod
    def clear_queue(queue):
        while True:
            try:
                queue.get_nowait()
            except (Exception,):
                break

    def __iter__(self):
        print("session data loader")

        self.clear_queue(self.task_queue)
        self.clear_queue(self.processed_queue)

        if self.shuffle:
            random.shuffle(self.sessions)

        task_allocator = TaskAllocator(self.task_queue, self.sessions,
                                       self.number_workers, self.show_process_bar,
                                       identity="SUPERVISED" if self.supervised else "UNSUPERVISED")
        print("new task allocated")
        task_allocator.start()

        # number_workers * TaskWorker - 4 process raw session and put it to processed_queue
        for _ in range(self.number_workers):
            task_worker = TaskWorker(self.task_queue, self.processed_queue,
                                     self.session_processor, supervised=self.supervised)
            task_worker.start()

        return self

    def __next__(self):
        is_continuous = []  # if continuous
        is_efficient = []  # if efficient
        if not self.sunset:
            for i in range(len(self.history_sessions)):
                if len(self.history_sessions[i]) == 0:
                    try:
                        processed_session = self.processed_queue.get(timeout=3)
                        self.history_sessions[i] = processed_session
                        is_efficient.append(1)
                    except (Exception,):
                        self.sunset = True
                        is_efficient.append(0)
                    is_continuous.append(0)
                else:
                    is_efficient.append(1)
                    is_continuous.append(1)

        else:
            for session in self.history_sessions:
                if len(session) > 0:
                    is_continuous.append(1)
                    is_efficient.append(1)
                else:
                    is_continuous.append(0)
                    is_efficient.append(0)

        self.history_sessions = [session for session in self.history_sessions if len(session) > 0]
        if len(self.history_sessions) == 0:
            raise StopIteration
        batch_sessions = [session[0] for session in self.history_sessions]  # the first segment of each session
        self.history_sessions = [session[1:] for session in self.history_sessions]  # truncate the first segment

        nn_inputs = []
        for idx, batch_data in enumerate(zip(*batch_sessions)):
            if self.dtypes is not None:
                nn_inputs.append(torch.tensor(batch_data, dtype=self.dtypes[idx], device=self.device))
            else:
                nn_inputs.append(torch.tensor(batch_data, dtype=torch.long, device=self.device))

        is_continuous = torch.tensor(is_continuous, dtype=torch.long, device=self.device)
        is_efficient = torch.tensor(is_efficient, dtype=torch.long, device=self.device)

        return nn_inputs, is_continuous, is_efficient


class SessionDataLoader:
    def __init__(self, sessions, batch_size, session_processor: SessionProcessor,
                 show_process_bar=True, mode="train", shuffle=False,
                 device="cpu:0", dtypes=None, supervised=False):

        self.sessions = sessions
        self.batch_size = batch_size
        self.session_processor = session_processor
        self.show_process_bar = show_process_bar

        assert mode in ("train", "valid", "test"), \
            "SessionDataSet Mode should be train or test, but valued {}".format(mode)

        self.mode = mode
        self.shuffle = shuffle
        self.history_sessions = [[] for _ in range(self.batch_size)]
        self.sunset = False
        self.device = device
        self.dtypes = dtypes
        self.supervised = supervised
        self.session_index = 0

    def __iter__(self):
        print("session data loader")
        if self.shuffle:
            random.shuffle(self.sessions)
        return self

    def load_processed_session(self):
        if self.session_index >= len(self.sessions):
            self.sunset = True
            return None

        session = self.sessions[self.session_index]
        processed_session = self.session_processor.process(session, supervised=self.supervised)
        self.session_index += 1
        return processed_session

    def __next__(self):
        is_continuous = []
        is_efficient = []

        for i in range(len(self.history_sessions)):
            if len(self.history_sessions[i]) == 0:
                if self.sunset:
                    is_efficient.append(0)
                else:
                    while True:
                        processed_session = self.load_processed_session()
                        if processed_session is None:
                            break
                        if len(processed_session) > 0:
                            break

                    if (processed_session is not None) and (len(processed_session) > 0):
                        self.history_sessions[i] = processed_session
                        is_efficient.append(1)
                    else:
                        is_efficient.append(0)

                is_continuous.append(0)
            else:
                is_efficient.append(1)
                is_continuous.append(1)

        self.history_sessions = [session for session in self.history_sessions if len(session) > 0]
        if len(self.history_sessions) == 0:
            raise StopIteration

        batch_sessions = [session[0] for session in self.history_sessions]
        self.history_sessions = [session[1:] for session in self.history_sessions]

        nn_inputs = []
        for idx, batch_data in enumerate(zip(*batch_sessions)):
            if self.dtypes is not None:
                nn_inputs.append(torch.tensor(batch_data, dtype=self.dtypes[idx], device=self.device))
            else:
                nn_inputs.append(torch.tensor(batch_data, dtype=torch.long, device=self.device))

        is_continuous = torch.tensor(is_continuous, dtype=torch.long, device=self.device)
        is_efficient = torch.tensor(is_efficient, dtype=torch.long, device=self.device)

        return nn_inputs, is_continuous, is_efficient


class SessionDataset:
    def __init__(self, session_processor, mode, batch_size, sessions, supervised=False):
        self.session_processor = session_processor
        self.mode = mode
        self.batch_size = batch_size
        self.sessions = sessions
        self.supervised = supervised

    def load_data(self):
        if TO.worker_num > 1:
            session_dataloader = MultiProcessSessionDataLoader(self.sessions,
                                                               self.batch_size,
                                                               self.session_processor,
                                                               mode=self.mode,
                                                               shuffle=True if self.mode == "train" else False,
                                                               device=TO.device,
                                                               task_queue_max_size=self.batch_size * 3,
                                                               processed_queue_max_size=self.batch_size * 3,
                                                               supervised=self.supervised)
            return session_dataloader
        else:
            session_dataloader = SessionDataLoader(self.sessions,
                                                   self.batch_size,
                                                   self.session_processor,
                                                   mode=self.mode,
                                                   shuffle=True if self.mode == "train" else False,
                                                   device=TO.device,
                                                   supervised=self.supervised)
            return session_dataloader


class MixedSessionDataloader:
    def __init__(self, supervised_dataloader, unsupervised_dataloader,
                 super_rate=0.0, super_1st=False, switch_interval=10):
        self.supervised_dataloader = supervised_dataloader
        self.unsupervised_dataloader = unsupervised_dataloader
        self.supervised_iterator = None
        self.unsupervised_iterator = None
        self.super_rate = super_rate
        self.super_alive = True
        self.unsuper_alive = True
        self.super_1st = super_1st
        self.switch_interval = switch_interval
        self.switch_idx = 0
        self.toss_v = self.toss()

    def __iter__(self):
        print("mixed session data loader")
        self.super_alive = True
        self.unsuper_alive = True
        self.supervised_iterator = iter(self.supervised_dataloader)
        self.unsupervised_iterator = iter(self.unsupervised_dataloader)
        return self

    def toss(self):
        if self.super_1st:
            return True
        toss_value = random.random()
        if (toss_value <= self.super_rate) and self.super_alive:
            return True
        return False

    def __next__(self):
        self.switch_idx += 1
        if self.switch_idx == self.switch_interval:
            self.switch_idx = 0
            self.toss_v = self.toss()

        def super_load():
            try:
                ret_data_ = next(self.supervised_iterator)
                return ret_data_
            except (StopIteration,):
                self.super_alive = False
                return None

        def unsuper_load():
            try:
                ret_data_ = next(self.unsupervised_iterator)
                return ret_data_
            except (StopIteration,):
                self.unsuper_alive = False
                return None

        if (not self.super_alive) and (not self.unsuper_alive):
            raise StopIteration

        if self.toss_v:
            ret_data = None

            if self.super_alive:
                ret_data = super_load()

            if (ret_data is None) and self.unsuper_alive:
                ret_data = unsuper_load()
        else:
            ret_data = None

            if self.unsuper_alive:
                ret_data = unsuper_load()

            if (ret_data is None) and self.super_alive:
                ret_data = super_load()

        if ret_data is None:
            raise StopIteration

        return ret_data


class MixedSessionDataset(SessionDataset):
    def __init__(self, session_processor, batch_size, sup_sessions, unsup_sessions, super_rate):
        super(MixedSessionDataset, self).__init__(session_processor, "mixed", batch_size, None)
        self.sup_sessions = sup_sessions
        self.unsup_sessions = unsup_sessions
        self.super_rate = super_rate

    def load_data(self):
        if TO.worker_num > 1:
            sup_session_dataloader = MultiProcessSessionDataLoader(self.sup_sessions,
                                                                   self.batch_size,
                                                                   self.session_processor,
                                                                   number_workers=TO.worker_num,
                                                                   shuffle=True,
                                                                   device=TO.device,
                                                                   task_queue_max_size=self.batch_size * 3,
                                                                   processed_queue_max_size=self.batch_size * 3,
                                                                   supervised=True)

            unsup_session_dataloader = MultiProcessSessionDataLoader(self.unsup_sessions,
                                                                     self.batch_size,
                                                                     self.session_processor,
                                                                     number_workers=TO.worker_num,
                                                                     shuffle=True,
                                                                     device=TO.device,
                                                                     task_queue_max_size=self.batch_size * 3,
                                                                     processed_queue_max_size=self.batch_size * 3,
                                                                     supervised=False)
        else:
            sup_session_dataloader = SessionDataLoader(self.sup_sessions,
                                                       self.batch_size,
                                                       self.session_processor,
                                                       mode="train",
                                                       shuffle=True,
                                                       device=TO.device,
                                                       supervised=True)

            unsup_session_dataloader = SessionDataLoader(self.unsup_sessions,
                                                         self.batch_size,
                                                         self.session_processor,
                                                         mode="train",
                                                         shuffle=True,
                                                         device=TO.device,
                                                         supervised=False)

        mixed_session_dataloader = MixedSessionDataloader(supervised_dataloader=sup_session_dataloader,
                                                          unsupervised_dataloader=unsup_session_dataloader,
                                                          super_rate=self.super_rate,
                                                          super_1st=False)
        return mixed_session_dataloader
