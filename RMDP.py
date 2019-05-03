# authors: anonymized

'''
Implementation inspired from Safe Policy Improvement by Minimizing Robust Baseline Regret
Optimization method from Robust Control of Markov Decision Processes with. Uncertain Transition Matrices (2005)
'''

from modelTransitions import *
import numpy as np


class RMDP_based_alorithm():
    def __init__(self, gamma, nb_states, nb_actions, delta, R, pi_b, terminal_state, nb_iter=200, max_residual=0.005):
        self.gamma = gamma
        self.R = R
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.delta = delta
        self.nb_iter = nb_iter
        self.V = np.zeros((self.nb_states, 1))
        self.max_residual = max_residual
        self.theta = 0.000001  # convergence threshold
        self.pi_t = np.zeros((self.nb_states, self.nb_actions))
        self.pi_b = pi_b
        self.terminal_state = terminal_state
        self.count_state_action = np.zeros((self.nb_states, self.nb_actions))

    def predict(self, state):
        return np.random.choice(self.nb_actions, 1, p=self.pi_t[state])

    def _compute_worst_case(self, model, value, uncertainty, inverse=False):
        """
		Apply the bijection algorithm to find a solution to the following optimisation problem
		Implemenation from the paper 'Robust Control of Markov Decision Processes with uncertainty Transition Matrices'
		()
		min P^T V
		so that ||P-model||<=uncertainty
		         1^TP=1
		         P>0
		"""
        new_value = np.copy(value)
        current_to_sort = self.gamma*new_value.reshape((self.nb_states, 1))+self.R
        if inverse:
            current_to_sort = -self.gamma*new_value.reshape((self.nb_states, 1)) - self.R
        order_indexes_v = np.argsort(current_to_sort.squeeze())
        k = order_indexes_v[0]
        eps = min(uncertainty / 2., 1 - model[k])
        o = np.copy(model)
        o[k] += eps
        i = len(order_indexes_v) - 1
        while eps > 0:
            k = order_indexes_v[i]
            diff = min((eps, o[k]))
            o[k] -= diff
            eps -= diff
            i -= 1
        return o, np.dot(o, current_to_sort)

    def _initialize_uniform(self, p):
        """ Initial policy """
        # Begin with uniform policy: array of shape |S| x |A|
        pi = np.full((self.nb_states, self.nb_actions), 1)
        pi = pi * np.sum(p, axis=2)  # remove invalid actions
        base = np.sum(pi, axis=1)  # get number of valid actions per state
        np.seterr(divide='ignore')
        pi = np.nan_to_num(pi / base[:, None])  # divide by number of actions, broadcast
        # np.seterr(divide='raise')
        return pi

    def _policy_evaluation_exact(self, pi, v, p):
        p_pi = np.einsum('ijk, ij->ik', p, pi)
        r = np.einsum('ijk,ik->ij', p, self.R)
        v = np.dot(np.linalg.inv((np.eye(p_pi.shape[0]) - self.gamma * p_pi)), r)
        return v

    def _value_iteration(self, p):
        n_states, n_actions = p.shape[:2]
        v = np.zeros(n_states)
        max_iteration = 10000
        r_pi = np.einsum('ijk,ik->ij', p, self.R)
        for it in range(max_iteration):
            q = self.R + self.gamma * np.einsum('ijk, k->ij', p, v)  # get q values
            q_mask = np.ma.masked_array(q, mask=(np.sum(p, axis=2) - 1) * (-1))  # mask invalid actions
            v_new = np.max(q_mask, axis=1)  # state-values equal max possible values
            v_new = v_new.filled(0)  # Masked states should have value 0
            if np.max(np.absolute(v - v_new)) < self.theta:
                v = v_new
                break;
            v = v_new
        pi = self._values_to_argmax(q_mask)
        pi = pi.filled(0)
        return v, pi, q

    def _values_to_argmax(self, q):
        def f_axis(q):
            z = np.zeros_like(q)
            z[np.argmax(q)] = 1
            return z

        return np.apply_along_axis(f_axis, 1, q)

    def fit(self, batch):
        self.model = ModelTransitions(batch, self.nb_states, self.nb_actions)
        for [action, state, next_state, reward] in batch:
            self.count_state_action[int(state), action] += 1
        self.count_state_action[self.count_state_action == 0] = 0.01
        self.uncertainty = np.zeros((self.nb_states, self.nb_actions))
        for i in range(self.nb_actions):
            for j in range(self.nb_states):
                self.uncertainty[j, i] = np.sqrt(2. / self.count_state_action[j, i] * np.log(
                    self.nb_actions * self.nb_states * 2 ** self.nb_states / self.delta))
        it = 0
        residual = np.inf
        self.V = np.zeros((self.nb_states, 1))
        while it < self.nb_iter and residual > self.max_residual:
            previous_V = np.copy(self.V)
            worst_cases_value = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
            current_worst_sigma = np.zeros((self.nb_states, self.nb_actions))
            for j in range(self.nb_states):
                for i in range(self.nb_actions):
                    worst_cases_value[j, i, :], current_worst_sigma[j, i] = self._compute_worst_case(
                        self.model.proba(j, i), self.V,
                        self.uncertainty[j, i])
            q = current_worst_sigma
            self.V = np.max(q, 1)
            residual = np.max(np.absolute(previous_V - self.V))
            it += 1
        q_mask = np.ma.masked_array(q, mask=(np.sum(worst_cases_value, axis=2) - 1) * (-1))
        self.pi_t = self._values_to_argmax(q_mask)

    def _evaluation_worst(self, pi):
        V = np.zeros((self.nb_states, 1))
        it = 0
        residual = np.inf
        while it < self.nb_iter and residual > self.max_residual:
            previous_V = np.copy(V)
            worst_cases_value = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
            current_worst_sigma = np.zeros((self.nb_states, self.nb_actions))
            for j in range(self.nb_states):
                for i in range(self.nb_actions):
                    worst_cases_value[j, i, :], current_worst_sigma[j, i] = self._compute_worst_case(
                        self.model.proba(j, i), V,
                        self.uncertainty[j, i])
            V = np.multiply(current_worst_sigma, pi).sum(axis=1)
            residual = np.max(np.absolute(previous_V - V))
            it += 1
        return V

    def _evaluation_optimism(self, pi):
        V = np.zeros((self.nb_states, 1))
        it = 0
        residual = np.inf
        while it < self.nb_iter and residual > self.max_residual:
            previous_V = np.copy(V)
            worst_cases_value = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
            current_worst_sigma = np.zeros((self.nb_states, self.nb_actions))
            for j in range(self.nb_states):
                for i in range(self.nb_actions):
                    worst_cases_value[j, i, :], current_worst_sigma[j, i] = self._compute_worst_case(
                        self.model.proba(j, i), V,
                        self.uncertainty[j, i], True)
            V = np.multiply(-current_worst_sigma, pi).sum(axis=1)
            residual = np.max(np.absolute(previous_V - V))
            it += 1
        return V

    def safety_test(self):
        worst_case_current = self._evaluation_worst(self.pi_t.data)
        best_case_baseline = self._evaluation_optimism(self.pi_b)
        return worst_case_current > best_case_baseline
