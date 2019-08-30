import numpy as np
import spibb
import spibb_utils


class Garnets:
    def __init__(self, nb_states, nb_actions, nb_next_state_transition, env_type=1, self_transitions=0):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.nb_next_state_transition = nb_next_state_transition
        self.transition_function = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        self.is_done = False
        self.initial_state = 0
        self.self_transitions = self_transitions
        self.current_state = self.initial_state
        self.final_state = nb_states - 1
        self._generate_transition_function()
        self.env_type = env_type

    def _generate_transition_function(self):
        for id_state in range(self.nb_states):
            for id_action in range(self.self_transitions):
                self_transition_prob = np.random.uniform(0.5, 1)
                partition = np.sort(np.random.uniform(0, 1 - self_transition_prob, self.nb_next_state_transition - 2))
                partition = np.concatenate(([0], partition, [1 - self_transition_prob]))
                probabilities = np.ediff1d(partition)
                choice_state = np.random.choice(self.nb_states, self.nb_next_state_transition - 1, replace=False)
                self.transition_function[id_state, id_action, choice_state] = probabilities
                self.transition_function[id_state, id_action, id_state] += self_transition_prob

            for id_action in range(self.self_transitions, self.nb_actions):
                partition = np.sort(np.random.uniform(0, 1, self.nb_next_state_transition - 1))
                partition = np.concatenate(([0], partition, [1]))
                probabilities = np.ediff1d(partition)
                choice_state = np.random.choice(self.nb_states, self.nb_next_state_transition, replace=False)
                self.transition_function[id_state, id_action, choice_state] = probabilities

    def reset(self):
        self.current_state = self.initial_state
        return int(self.current_state)

    def sample_action(self):
        return int(np.random.choice(self.nb_actions, 1))

    def _get_reward(self, state, action, next_state):
        if next_state == self.final_state:
            return 1
        else:
            return 0

    def step(self, action, easter_egg):
        if self.transition_function[int(self.current_state), action, :].squeeze().sum() != 1:
            print(self.transition_function[int(self.current_state), action, :].squeeze())
            print(self.current_state)
            print(easter_egg)
            print(self.final_state)
        next_state = np.random.choice(self.nb_states, 1,
                                      p=self.transition_function[int(self.current_state), action, :].squeeze())
        reward = self._get_reward(self.current_state, action, next_state)
        if self.env_type == 2 and next_state == easter_egg:  # easter
            reward = 1
        state = self.current_state
        if self.env_type == 2:  # easter
            self.is_done = (next_state == self.final_state or next_state == easter_egg)
        else:
            self.is_done = (next_state == self.final_state)
        self.current_state = next_state
        return int(state), reward, int(next_state), self.is_done

    # Reward matrix
    def compute_reward(self):
        R = np.zeros((self.nb_states, self.nb_states))
        for s in range(self.nb_states):
            R[s, self.final_state] = 1
        return R

    # Transition function matrix
    def compute_transition_function(self):
        t = self.transition_function.copy()
        t[self.final_state, :, :] = 0
        return t

    def start_state(self):
        return self.initial_state

    def generate_baseline_policy(self, gamma, softmax_target_perf_ratio=0.75,
                                 baseline_target_perf_ratio=0.5, softmax_reduction_factor=0.9,
                                 perturbation_reduction_factor=0.9):
        if softmax_target_perf_ratio < baseline_target_perf_ratio:
            softmax_target_perf_ratio = baseline_target_perf_ratio

        farther_state, pi_star_perf, q_star, pi_rand_perf = self._find_farther_state(gamma)
        p, r = self._set_temporary_final_state(farther_state)
        self.transition_function = p.copy()
        r_reshaped = spibb_utils.get_reward_model(p, r)

        softmax_target_perf = softmax_target_perf_ratio * (pi_star_perf - pi_rand_perf) \
                              + pi_rand_perf
        pi, _, _ = self._generate_softmax_policy(q_star, p, r_reshaped,
                                                 softmax_target_perf,
                                                 softmax_reduction_factor, gamma)

        baseline_target_perf = baseline_target_perf_ratio * (pi_star_perf - pi_rand_perf) \
                               + pi_rand_perf
        pi, v, q = self._perturb_policy(pi, q_star, p, r_reshaped, baseline_target_perf,
                                        perturbation_reduction_factor, gamma)

        return pi, q, pi_star_perf, v[0], pi_rand_perf

    def _perturb_policy(self, pi, q_star, p, r_reshaped, baseline_target_perf,
                        reduction_factor, gamma):
        v = np.ones(1)
        while v[0] > baseline_target_perf:
            x = np.random.randint(self.nb_states)
            pi[x, np.argmax(q_star[x, :])] *= reduction_factor
            pi[x, :] /= np.sum(pi[x, :])
            v, q = spibb.policy_evaluation_exact(pi, r_reshaped, p, gamma)

        avg_time_to_goal = np.log(v[0]) / np.log(gamma)
        print("Perturbed policy performance : " + str(v[0]))
        print("Perturbed policy average time to goal: " + str(avg_time_to_goal))
        return pi, v, q

    def _generate_softmax_policy(self, q_star, p, r_reshaped, softmax_target_perf,
                                 reduction_factor, gamma):
        temp = 2 * 10 ** 6  # Actually starts exploring for half its value.
        v = np.ones(1)
        while v[0] > softmax_target_perf:
            temp *= reduction_factor
            pi = spibb.softmax(q_star, temp)
            v, q = spibb.policy_evaluation_exact(pi, r_reshaped, p, gamma)

        avg_time_to_goal = np.log(v[0]) / np.log(gamma)
        print("Softmax performance : " + str(v[0]))
        print("Softmax temperature : " + str(temp))
        print("Softmax average time to goal: " + str(avg_time_to_goal))
        return pi, v, q

    def _set_temporary_final_state(self, final_state):
        self.final_state = final_state
        p = self.compute_transition_function()
        r = self.compute_reward()
        return p, r

    def _find_farther_state(self, gamma):
        argmin = -1
        min_value = 1
        rand_value = 0
        best_q_star = 0
        mask_0, thres = spibb.compute_mask(self.nb_states, self.nb_actions, 1, 1, [])
        mask_0 = ~mask_0
        rand_pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        for final_state in range(1, self.nb_states):
            p, r = self._set_temporary_final_state(final_state)
            r_reshaped = spibb_utils.get_reward_model(p, r)

            rl = spibb.spibb(gamma, self.nb_states, self.nb_actions, mask_0, mask_0, p, r_reshaped, 'default')
            rl.fit()
            v_star, q_star = spibb.policy_evaluation_exact(rl.pi, r_reshaped, p, gamma)
            v_rand, q_rand = spibb.policy_evaluation_exact(rand_pi, r_reshaped, p, gamma)

            perf_star = v_star[0]
            perf_rand = v_rand[0]

            if perf_star < min_value and perf_star > gamma ** 50:
                min_value = perf_star
                argmin = final_state
                rand_value = perf_rand
                best_q_star = q_star.copy()

        avg_time_to_goal = np.log(min_value) / np.log(gamma)
        avg_time_to_goal_rand = np.log(rand_value) / np.log(gamma)
        print("Optimal performance : " + str(min_value))
        print("Optimal average time to goal: " + str(avg_time_to_goal))
        print("Random policy performance : " + str(rand_value))
        print("Random policy average time to goal: " + str(avg_time_to_goal_rand))

        return argmin, min_value, best_q_star, rand_value
