# -*- coding: utf-8 -*-

import json


class Session:
    def __init__(self, posts, gth_responses, hyp_responses, states, actions, gth_states):
        self.posts = posts
        self.gth_responses = gth_responses
        self.hyp_responses = hyp_responses
        self.states = states
        self.actions = actions
        self.gth_states = gth_states

    def to_dict(self):
        session_dict = []

        for turn_idx, (post, gth_response, hyp_response, state, action, gth_state) in enumerate(
                zip(self.posts, self.gth_responses, self.hyp_responses, self.states, self.actions, self.gth_states)):
            episode = dict()
            episode["turn_idx"] = turn_idx + 1
            episode["post"] = post
            episode["gth"] = gth_response
            episode["hyp"] = hyp_response
            episode["state"] = state
            episode["action"] = action
            episode["gth_state"] = gth_state
            session_dict.append(episode)

        return session_dict


class SessionCropper:
    def __init__(self, batch_size, indent=4):
        self.batch_size = batch_size
        self.session_garner = []
        self.episode_cache = [[] for _ in range(self.batch_size)]
        self.indent = indent

    def step_on(self, batch_post, batch_gth_response, batch_hyp_response,
                batch_state, batch_action, inherited, materialistic, gth_states):

        for m_idx in range(len(materialistic) - 1, -1, -1):
            if materialistic[m_idx] == 0 and inherited[m_idx] == 0:
                session_data = self.episode_cache[m_idx]
                del self.episode_cache[m_idx]
                if len(session_data) <= 0:
                    continue
                session = Session(*zip(*session_data))
                self.session_garner.append(session)
            elif materialistic[m_idx] == 1 and inherited[m_idx] == 0:
                session_data = self.episode_cache[m_idx]
                self.episode_cache[m_idx] = []
                if len(session_data) <= 0:
                    continue
                session = Session(*zip(*session_data))
                self.session_garner.append(session)
            elif materialistic[m_idx] == 1 and inherited[m_idx] == 1:
                pass  # do not process
            else:
                raise RuntimeError

        batch_data = list(zip(batch_post, batch_gth_response, batch_hyp_response,
                              batch_state, batch_action, gth_states))

        for idx, data in enumerate(batch_data):
            self.episode_cache[idx].append(data)

    def __finished__(self):
        for session_data in self.episode_cache:
            if len(session_data) <= 0:
                continue
            session = Session(*zip(*session_data))
            self.session_garner.append(session)
        self.episode_cache = []

    def to_dict(self):
        self.__finished__()
        sessions = []

        for session_idx, session in enumerate(self.session_garner):
            session_dict = session.to_dict()
            session_dict = {"session_idx": session_idx + 1, "session": session_dict}
            sessions.append(session_dict)

        return sessions

    def __str__(self):
        return json.dumps(self.session_garner, indent=4, ensure_ascii=False)

    def __len__(self):
        self.__finished__()
        return len(self.session_garner)
