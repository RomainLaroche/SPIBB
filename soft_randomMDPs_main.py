import os
import sys

expname = sys.argv[1]
index = int(sys.argv[2])
import pandas as pd
import garnets
import spibb
import modelTransitions
from RMDP import *
from SPI import *
from shutil import copyfile

print('Start of experiment')


def safe_save(filename, df):
    df.to_excel(filename + '.temp.xlsx')
    copyfile(filename + '.temp.xlsx', filename + '.xlsx')
    os.remove(filename + '.temp.xlsx')
    print(str(len(results)) + ' lines saved to ' + filename + '.xlsx')


nb_trajectories_list = [10, 20, 50, 100, 200, 500, 1000, 2000]
delta = 1
epsilons = [0.1, 0.2, 0.5, 1, 2, 5]
ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

seed = index
np.random.seed(seed)
gamma = 0.95
nb_states = 50
nb_actions = 4
nb_next_state_transition = 4
env_type = 2  # 1 for one terminal state, 2 for two terminal states

mask_0, thres = spibb.compute_mask(nb_states, nb_actions, 1, 1, [])
mask_0 = ~mask_0
rand_pi = np.ones((nb_states, nb_actions)) / nb_actions

filename = 'results/' + expname + '/results_' + str(index)

results = []
if not os.path.isdir('results'):
    os.mkdir('results')
if not os.path.isdir('results/' + expname):
    os.mkdir('results/' + expname)

self_transitions = 0

while True:
    for ratio in ratios:
        garnet = garnets.Garnets(nb_states, nb_actions, nb_next_state_transition,
                                 env_type=env_type, self_transitions=self_transitions)

        softmax_target_perf_ratio = (ratio + 1) / 2
        baseline_target_perf_ratio = ratio
        pi_b, q_pi_b, pi_star_perf, pi_b_perf, pi_rand_perf = \
            garnet.generate_baseline_policy(gamma,
                                            softmax_target_perf_ratio=softmax_target_perf_ratio,
                                            baseline_target_perf_ratio=baseline_target_perf_ratio)

        reward_current = garnet.compute_reward()
        current_proba = garnet.transition_function
        if env_type == 2:  # easter
            # Randomly pick a second terminal state and update model parameters
            potential_final_states = [s for s in range(nb_states) if s != garnet.final_state and s != 0]
            easter_egg = np.random.choice(potential_final_states)
            # Or pick the one with the least transitions
            # current_proba_sum = current_proba.reshape(-1, current_proba.shape[-1]).sum(axis=0)
            # mask_easter = np.ma.array(current_proba_sum, mask=False)
            # mask_easter.mask[garnet.final_state] = True
            # easter_egg = np.argmin(mask_easter)
            assert (garnet.final_state != easter_egg)
            reward_current[:, easter_egg] = 1
            current_proba[easter_egg, :, :] = 0
            r_reshaped = spibb_utils.get_reward_model(current_proba, reward_current)
            # Compute optimal policy in this new environment
            true_rl = spibb.spibb(gamma, nb_states, nb_actions, mask_0, mask_0, current_proba, r_reshaped, 'default')
            true_rl.fit()
            pi_star_perf = spibb.policy_evaluation_exact(true_rl.pi, r_reshaped, current_proba, gamma)[0][0]
            print("Optimal perf in easter egg environment:\t\t\t" + str(pi_star_perf))
            pi_b_perf = spibb.policy_evaluation_exact(pi_b, r_reshaped, current_proba, gamma)[0][0]
            print("Baseline perf in easter egg environment:\t\t\t" + str(pi_b_perf))
        else:
            easter_egg = None
            r_reshaped = spibb_utils.get_reward_model(current_proba, reward_current)

        for nb_trajectories in nb_trajectories_list:
            # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
            trajectories, batch_traj = spibb_utils.generate_batch(nb_trajectories, garnet, pi_b, easter_egg)
            print("GENERATED A DATASET OF " + str(nb_trajectories) + " TRAJECTORIES")

            # Compute the maximal likelihood model for transitions and rewards.
            # NB: the true reward function can be used for ease of implementation since it is not stochastic in our environment.
            # One should compute it from the samples when it is stochastic.
            model = modelTransitions.ModelTransitions(batch_traj, nb_states, nb_actions)
            reward_model = spibb_utils.get_reward_model(model.transitions, reward_current)

            # Estimates the values of the baseline policy with a monte-carlo estimation from the batch data:
            # q_pib_est = spibb_utils.compute_q_pib_est(gamma, nb_states, nb_actions, trajectories)

            # Computes the RL policy
            rl = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_0, model.transitions, reward_model, 'default')
            rl.fit()
            # Evaluates the RL policy performance
            perfrl = spibb.policy_evaluation_exact(rl.pi, r_reshaped, current_proba, gamma)[0][0]
            print("perf RL:\t\t\t" + str(perfrl))

            # Computes the Reward-adjusted MDP RL policy:
            count_state_action = 0.00001 * np.ones((nb_states, nb_actions))
            kappa = 0.003
            for [action, state, next_state, reward] in batch_traj:
                count_state_action[state, action] += 1
            ramdp_reward_model = reward_model - kappa / np.sqrt(count_state_action)
            ramdp = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_0, model.transitions, ramdp_reward_model,
                                'default')
            ramdp.fit()
            # Evaluates the RL policy performance
            perf_RaMDP = spibb.policy_evaluation_exact(ramdp.pi, r_reshaped, current_proba, gamma)[0][0]
            print("perf RaMDP:\t\t\t" + str(perf_RaMDP))

            # Computes the Robust MDP policy:
            terminal_state = 24
            delta_RobustMDP = 0.001
            rmdp = RMDP_based_alorithm(gamma, nb_states, nb_actions, delta_RobustMDP,
                                       reward_current[0].reshape((nb_states, 1)), pi_b, terminal_state)
            rmdp.fit(batch_traj)
            safety_test = rmdp.safety_test()[0]
            perf_RMDP_based_alorithm = spibb.policy_evaluation_exact(rmdp.pi_t, r_reshaped, current_proba, gamma)[0][0]
            if safety_test:
                perf_RMDP_based_alorithm_safe = perf_RMDP_based_alorithm
            else:
                perf_RMDP_based_alorithm_safe = pi_b_perf
            print("perf RMDP_based_algorithm:\t" + str(perf_RMDP_based_alorithm))

            # Computes the HCPI doubly robust policy:
            delta_HCPI = 0.9
            spi = SPI(gamma, pi_b, delta_HCPI, "student_t_test", "doubly_robust", trajectories, 0, 1, pi_b_perf,
                      reward_current)
            pi_hcope = spi.get_policy()
            perfhcope_doubly_robust = spibb.policy_evaluation_exact(pi_hcope, r_reshaped, current_proba, gamma)[0][0]
            print("perf HCPI doubly_robust:\t" + str(perfhcope_doubly_robust))

            N_wedge = 10
            # Computation of the binary mask for the bootstrapped state actions
            mask = spibb.compute_mask_N_wedge(nb_states, nb_actions, N_wedge, batch_traj)

            ## Policy-based SPIBB ##

            # Computes the Pi_b_SPIBB policy:
            pib_SPIBB = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model,
                                    'Pi_b_SPIBB')
            pib_SPIBB.fit()
            # Evaluates the Pi_b_SPIBB performance:
            perf_Pi_b_SPIBB = spibb.policy_evaluation_exact(pib_SPIBB.pi, r_reshaped, current_proba, gamma)[0][0]
            print("perf Pi_b_SPIBB:\t\t" + str(perf_Pi_b_SPIBB))

            # Computes the Pi_<b_SPIBB policy:
            pi_leq_b_SPIBB = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model,
                                         'Pi_leq_b_SPIBB')
            pi_leq_b_SPIBB.fit()
            # Evaluates the Pi_<b_SPIBB performance:
            perf_Pi_leq_b_SPIBB = \
                spibb.policy_evaluation_exact(pi_leq_b_SPIBB.pi, r_reshaped, current_proba, gamma)[0][0]
            print("perf Pi_leq_b_SPIBB:\t\t" + str(perf_Pi_leq_b_SPIBB))

            for epsilon in epsilons:
                # Computation of the binary mask for the bootstrapped state actions
                mask = spibb.compute_mask(nb_states, nb_actions, epsilon, delta, batch_traj)[0]
                # Computation of the transition errors
                errors = spibb.compute_errors(nb_states, nb_actions, delta, batch_traj)

                ## Soft-SPIBB 1-step ##

                # Simplex (more variables than constraints)
                soft_SPIBB_simplex_1step = spibb.spibb(
                    gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Soft_SPIBB_simplex',
                    errors=errors, epsilon=2 * epsilon, max_nb_it=1
                )
                soft_SPIBB_simplex_1step.fit()
                # Evaluates the Soft-SPIBB-simplex performance
                perf_soft_SPIBB_simplex_1step = \
                spibb.policy_evaluation_exact(soft_SPIBB_simplex_1step.pi, r_reshaped, current_proba, gamma)[0][0]
                print("perf Exact-Soft-SPIBB 1-step:\t" + str(perf_soft_SPIBB_simplex_1step))

                # Computes the Soft-SPIBB-sort-Q policy
                soft_SPIBB_sort_Q_1step = spibb.spibb(
                    gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Soft_SPIBB_sort_Q',
                    errors=errors, epsilon=2 * epsilon, max_nb_it=1
                )
                soft_SPIBB_sort_Q_1step.fit()
                # Evaluates the Soft-SPIBB-sort-Q performance
                perf_soft_SPIBB_sort_Q_1step = \
                spibb.policy_evaluation_exact(soft_SPIBB_sort_Q_1step.pi, r_reshaped, current_proba, gamma)[0][0]
                print("perf Approx-Soft-SPIBB 1-step:\t\t" + str(perf_soft_SPIBB_sort_Q_1step))

                ## Soft-SPIBB multi-steps ##
                # Simplex (more variables than constraints)
                soft_SPIBB_simplex = spibb.spibb(
                    gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Soft_SPIBB_simplex',
                    errors=errors, epsilon=2 * epsilon
                )
                soft_SPIBB_simplex.fit()
                # Evaluates the Soft-SPIBB-simplex performance
                perf_soft_SPIBB_simplex = \
                spibb.policy_evaluation_exact(soft_SPIBB_simplex.pi, r_reshaped, current_proba, gamma)[0][0]
                print("perf Exact-Soft-SPIBB:\t" + str(perf_soft_SPIBB_simplex))

                # Computes the Soft-SPIBB-sort-Q policy
                soft_SPIBB_sort_Q = spibb.spibb(
                    gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Soft_SPIBB_sort_Q',
                    errors=errors, epsilon=2 * epsilon
                )
                soft_SPIBB_sort_Q.fit()
                # Evaluates the Soft-SPIBB-sort-Q performance
                perf_soft_SPIBB_sort_Q = \
                spibb.policy_evaluation_exact(soft_SPIBB_sort_Q.pi, r_reshaped, current_proba, gamma)[0][0]
                print("perf Approx-Soft-SPIBB:\t\t" + str(perf_soft_SPIBB_sort_Q))

                results.append([seed, gamma, nb_states, nb_actions, 4,
                                nb_trajectories, softmax_target_perf_ratio, baseline_target_perf_ratio,
                                pi_b_perf, pi_rand_perf, pi_star_perf, perfrl, perf_RaMDP, perf_RMDP_based_alorithm,
                                perfhcope_doubly_robust, perf_Pi_b_SPIBB, perf_Pi_leq_b_SPIBB,
                                perf_soft_SPIBB_simplex, perf_soft_SPIBB_sort_Q,
                                perf_soft_SPIBB_simplex_1step, perf_soft_SPIBB_sort_Q_1step,
                                kappa, delta_RobustMDP, delta_HCPI, N_wedge, epsilon
                                ])

    df = pd.DataFrame(results, columns=['seed', 'gamma', 'nb_states', 'nb_actions', 'nb_next_state_transition',
                                        'nb_trajectories', 'softmax_target_perf_ratio', 'baseline_target_perf_ratio',
                                        'baseline_perf', 'pi_rand_perf', 'pi_star_perf', 'perfrl',
                                        'perf_RaMDP', 'perf_RMDP_based_algorithm', 'perfhcope_doubly_robust',
                                        'perf_Pi_b_SPIBB', 'perf_Pi_leq_b_SPIBB',
                                        'perf_soft_SPIBB_simplex', 'perf_soft_SPIBB_sort_Q',
                                        'perf_soft_SPIBB_simplex_1step', 'perf_soft_SPIBB_sort_Q_1step',
                                        'kappa', 'delta_RobustMDP', 'delta_HCPI', 'N_wedge', 'epsilon'])

    # Save it to an excel file
    safe_save(filename, df)
