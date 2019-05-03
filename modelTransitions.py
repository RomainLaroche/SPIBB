# authors: anonymized

import numpy as np

# Build a model of the transitions
class ModelTransitions():
    def __init__(self, batch, nb_states, nb_actions,zero_unseen=True):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.count_state_action_next = np.zeros((self.nb_states, nb_actions, self.nb_states))
        for [action, state, next_state, _] in batch:
            self.count_state_action_next[int(state), action, int(next_state)] += 1
        self.transitions = self.count_state_action_next/np.sum(self.count_state_action_next, 2)[:, :, np.newaxis]
        if zero_unseen:
            self.transitions = np.nan_to_num(self.transitions)
        else:
            self.transitions[np.isnan(self.transitions)] = 1./nb_states
        # stop here if you want the unknown (never encountered) state-action couples to be considered terminal (no reward afterwards). It forces Q-value to equal 0 and also speeds up training.

        # do this if you want the unknown state action to be agnostic (transit uniformly to all existing states)
        # unknown_vect = np.ones(self.nb_states)/self.nb_states
        # for s in range(self.nb_states - 1):
        #     for a in range(self.nb_actions):
        #         if np.sum(self.transitions[s,a]) == 0:
        #             self.transitions[s,a] = unknown_vect

        # do this if you want the unknown state action to be worst case scenario (transit to the same state, which in our environment yields the worst possible reward)
        # for s in range(self.nb_states - 1):
        #     for a in range(self.nb_actions):
        #         if np.sum(self.transitions[s,a]) == 0:
        #             self.transitions[s,a ,s] = 1

    def sample(self, state, action):
        next_state=np.random.choice(self.nb_states, 1, p=self.transitions[state,action,:].squeeze())
        return next_state

    def proba(self, state, action):
        return self.transitions[state, action, :]

    def masked_model(self, mask):
        masked_model = self.transitions.copy()
        masked_model[~mask] = 0
        return masked_model