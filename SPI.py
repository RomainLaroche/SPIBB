# authors: anonymized

import numpy as np
from HCPI import *


class SPI:
    def __init__(self,gamma,pi_b,confidence,lower_bound_strategy,estimator_strategy,trajectories,rho_min,rho_max,perf_baseline,R,training_size = 0.2,regularized = False):
        self.gamma = gamma
        self.pi_b = pi_b
        self.R=R
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.performance_baseline=perf_baseline
        self.lower_bound_strategy = lower_bound_strategy
        self.estimator_strategy = estimator_strategy
        self.trajectories = trajectories
        self.confidence = confidence
        training_index = int(training_size*len(trajectories))
        self.training_trajectories = self.trajectories[:training_index]
        self.testing_trajectories = self.trajectories[training_index:]
        self.pi_t,self.lower_bound_target,self.lower_bound_regularization = self.compute_policy()
        self.regularized = regularized

    def get_policy(self):
        if self.lower_bound_target>self.performance_baseline:
            return self.pi_t
        else:
            return self.pi_b

    def compute_policy(self):
        batch = []
        for trajectory in self.training_trajectories:
            for [action, state, next_state, reward] in trajectory:
                batch.append([action, state, next_state, reward])
        q_optimal = self._q_learning(batch, self.gamma, self.pi_b.shape[0], self.pi_b.shape[1],5000)
        pi_optimal = self._values_to_argmax(q_optimal)
        # Regularization on the optimal policy
        regularization_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        # Evaluate all the policy and return the one with the higher lower bound
        best_lower_bound = - np.inf
        current_best_policy = pi_optimal
        current_regularization = 0
        for regularization_parameter in regularization_list:
            current_pi = (1-regularization_parameter)*pi_optimal+regularization_parameter*self.pi_b
            estimator = LowerBoundEstimator(self.gamma, self.pi_b, current_pi, self.lower_bound_strategy,self.confidence,self.estimator_strategy,self.rho_min,self.rho_max,self.R,self.training_trajectories)
            lower_bound = estimator(self.testing_trajectories)
            if lower_bound > best_lower_bound:
                current_best_policy = current_pi
                best_lower_bound = lower_bound
                current_regularization=regularization_parameter
        return current_best_policy,best_lower_bound,current_regularization

    def _q_learning(self,batch, gamma, state_size, nb_actions, nb_steps):
        """
        Implementation of offline version of q-learning algorithm
        :param batch:
        :param gamma:
        :param state_size:
        :param nb_actions:
        :param nb_steps:
        :return: Q: the state-action value function corresponding to the optimal policy
        """
        nb_states = int(state_size)
        Q = np.zeros((nb_states, nb_actions))
        current_alpha = 0.1
        for step in range(int(nb_steps)):
            id_choice = np.random.randint(len(batch))
            [action, state, next_state, reward] = batch[id_choice]
            Q[state, action] = Q[state, action] + current_alpha * (
                    reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        return Q

    def _values_to_argmax(self,q):
        def f_axis(q):
            z = np.zeros_like(q)
            z[np.argmax(q)] = 1
            return z

        return np.apply_along_axis(f_axis, 1, q)
