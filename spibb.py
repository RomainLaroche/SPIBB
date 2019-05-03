# authors: anonymized

import numpy as np
from scipy.optimize import linprog
import itertools


# Computes the non-bootstrapping mask
def compute_mask(nb_states, nb_actions, epsilon, delta, batch):
    N_wedge = 2*(np.log((2*nb_states*nb_actions*2**nb_states)/delta))/epsilon**2
    return compute_mask_N_wedge(nb_states, nb_actions, N_wedge, batch), N_wedge

def compute_mask_N_wedge(nb_states, nb_actions, N_wedge, batch):
    count_state_action = np.zeros((nb_states, nb_actions))
    for [action, state, next_state, reward] in batch:
        count_state_action[state, action] += 1
    return count_state_action > N_wedge


# Computes the transition errors for all state-action pairs
def compute_errors(nb_states, nb_actions, delta, batch):
    count_state_action = np.zeros((nb_states, nb_actions))
    errors = np.zeros((nb_states, nb_actions))
    for [action, state, next_state, reward] in batch:
        count_state_action[state, action] += 1
    for state in range(nb_states):
        for action in range(nb_actions):
            if count_state_action[state, action] == 0:
                errors[state, action] = np.inf
            else:
                errors[state, action] = np.sqrt(
                    2*(np.log(2*(nb_states*nb_actions)/delta))/count_state_action[state, action]
                )
    return errors


def policy_evaluation_exact(pi, r, p, gamma):
    """
    Evaluate policy by taking the inverse
    Args:
      pi: policy, array of shape |S| x |A|
      r: the true rewards, array of shape |S| x |A|
      p: the true state transition probabilities, array of shape |S| x |A| x |S|
    Return:
      v: 1D array with updated state values
    """
    # Rewards according to policy: Hadamard product and row-wise sum
    r_pi = np.einsum('ij,ij->i', pi, r)

    # Policy-weighted transitions:
    # multiply p by pi by broadcasting pi, then sum second axis
    # result is an array of shape |S| x |S|
    p_pi = np.einsum('ijk, ij->ik', p, pi)
    v = np.dot(np.linalg.inv((np.eye(p_pi.shape[0]) - gamma * p_pi)), r_pi)
    return v, r + gamma*np.einsum('i, jki->jk', v, p)


def softmax(q, temp):
    exp = np.exp(temp*(q - np.max(q, axis=1)[:,None]))
    pi = exp / np.sum(exp, axis=1)[:,None]
    return pi

class spibb():
    # gamma is the discount factor,
    # nb_states is the number of states in the MDP,
    # nb_actions is the number of actions in the MDP,
    # pi_b is the baseline policy,
    # mask is the mask where the one does not need to bootstrap,
    # model is the transition model,
    # reward is the reward model,
    # space denotes the type of policy bootstrapping,
    # q_pib_est is the MC estimator of the state values of baseline policy
    def __init__(self, gamma, nb_states, nb_actions, pi_b, mask, model, reward, space, env_type, q_pib_est, errors=None, epsilon=None, nb_traj=0, lmbda=0):
        self.gamma = gamma
        self.nb_actions = nb_actions
        self.nb_states = nb_states
        if env_type == 1:
            self.nb_states = self.nb_states * 2
        self.nb_actions = nb_actions
        self.P = model
        self.pi_b = pi_b
        self.pi_b_masked = self.pi_b.copy()
        self.pi_b_masked[mask] = 0
        self.mask = mask
        self.R = reward.reshape(self.nb_states * self.nb_actions)
        self.space = space
        self.q_pib_est_masked = None
        if q_pib_est is not None:
            self.q_pib_est_masked = q_pib_est.copy()
            self.q_pib_est_masked[mask] = 0
        self.errors = errors
        self.epsilon = epsilon
        self.nb_traj = nb_traj
        self.lmbda = lmbda

    # starts a new episode (during the policy exploitation)
    def new_episode(self):
        self.has_bootstrapped = False

    # trains the policy
    def fit(self):
        pi = self.pi_b.copy()
        q = np.zeros((self.nb_states, self.nb_actions))
        old_q = np.ones((self.nb_states, self.nb_actions))
        nb_sa = self.nb_states * self.nb_actions
        nb_it = 0
        old_pi = None
        while np.linalg.norm(q - old_q) > 0.000000001:
            old_q = q.copy()
            M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', self.P, pi).reshape(nb_sa, nb_sa)
            q = np.dot(np.linalg.inv(M), self.R).reshape(self.nb_states, self.nb_actions)
            if self.q_pib_est_masked is not None:
                q += self.q_pib_est_masked
            pi = self.update_pi(q, old_pi)
            old_pi = pi
            nb_it += 1
            if nb_it > 1000:
                with open("notconverging.txt", "a") as myfile:
                    myfile.write(str(self.space) + " epsilon=" + str(self.epsilon) + " nb_traj=" + str(self.nb_traj) + " is not converging. \n")
                break
        self.pi = pi
        self.q = q

    # does the policy improvement inside the policy iteration loop
    def update_pi(self, q, old_pi=None):
        if self.space == 'Pi_b_SPIBB':
            pi = self.pi_b_masked.copy()
            for s in range(self.nb_states):
                if len(q[s, self.mask[s]]) > 0:
                    pi_b_masked_sum = np.sum(self.pi_b_masked[s])
                    pi[s][np.where(self.mask[s])[0][np.argmax(q[s, self.mask[s]])]] = 1 - pi_b_masked_sum
        elif self.space == 'Pi_leq_b_SPIBB':
            pi = np.zeros(self.pi_b_masked.shape)
            for s in range(self.nb_states):
                A = np.argsort(-q[s, :])
                pi_current_sum = 0
                for a in A:
                    if self.mask[s, a] or self.pi_b[s, a] > 1 - pi_current_sum:
                        pi[s, a] = 1 - pi_current_sum
                        break
                    else:
                        pi[s, a] = self.pi_b[s, a]
                        pi_current_sum += pi[s, a]
        # 'default' behaviour is used when there is no constraint in the the policy improvement projection
        else:
            pi = np.zeros(self.pi_b_masked.shape)
            for s in range(self.nb_states):
                pi[s, np.argmax(q[s, :])] = 1
        return pi

    # implements the trained policy
    def predict(self, state, bootstrap):
        if self.has_bootstrapped:
            choice = np.random.choice(self.nb_actions, 1, p=self.pi_b[state])
        else:
            choice = np.random.choice(self.nb_actions, 1, p=self.pi[state])
            if bootstrap and np.sum(self.P[state, choice]) < 0.5:
                self.has_bootstrapped = True
        return choice
