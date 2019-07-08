# authors: anonymized

import os
import sys
expname = sys.argv[1]
index = int(sys.argv[2])
import numpy as np
import pandas as pd
import garnets 
import spibb
import spibb_utils
import modelTransitions
from RMDP import *
from SPI import *

from shutil import copyfile
from math import ceil, floor
spibb_utils.prt('Start of experiment')


def safe_save(filename, df):
	df.to_excel(filename + '.temp.xlsx')
	copyfile(filename + '.temp.xlsx', filename + '.xlsx')
	os.remove(filename + '.temp.xlsx')
	spibb_utils.prt(str(len(results)) + ' lines saved to ' + filename + '.xlsx')

N_wedges = [5,7,10,15,20,30,50,70,100]
nb_trajectories_list = [10, 20, 50, 100, 200, 500, 1000, 2000]

ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

seed = index
np.random.seed(seed)

gamma = 0.95
nb_states = 50
nb_actions = 4
nb_next_state_transition = 4

mask_0, thres = spibb.compute_mask(nb_states, nb_actions, 1, 1, [])
mask_0 = ~mask_0
rand_pi = np.ones((nb_states,nb_actions)) / nb_actions

filename = 'results/' + expname + '/results_' + str(index)

results = []
if not os.path.isdir('results'):
	os.mkdir('results')
if not os.path.isdir('results/' + expname):
	os.mkdir('results/' + expname)

while True:
	for ratio in ratios:
		garnet = garnets.Garnets(nb_states, nb_actions, nb_next_state_transition, self_transitions=0)

		softmax_target_perf_ratio = (ratio + 1) / 2
		baseline_target_perf_ratio = ratio
		pi_b, q_pi_b, pi_star_perf, pi_b_perf, pi_rand_perf = \
								garnet.generate_baseline_policy(gamma,
																softmax_target_perf_ratio=softmax_target_perf_ratio,
																baseline_target_perf_ratio=baseline_target_perf_ratio)
		
		reward_current = garnet.compute_reward()
		current_proba = garnet.transition_function
		r_reshaped = spibb_utils.get_reward_model(current_proba, reward_current)
		
		for nb_trajectories in nb_trajectories_list:
			# Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
			trajectories, batch_traj = spibb_utils.generate_batch(nb_trajectories, garnet, pi_b)
			spibb_utils.prt("GENERATED A DATASET OF " + str(nb_trajectories) + " TRAJECTORIES")

			# Compute the maximal likelihood model for transitions and rewards.
			# NB: the true reward function can be used for ease of implementation since it is not stochastic in our environment.
			# One should compute it fro mthe samples when it is stochastic.
			model = modelTransitions.ModelTransitions(batch_traj, nb_states, nb_actions)
			reward_model = spibb_utils.get_reward_model(model.transitions, reward_current)

			# Estimates the values of the baseline policy with a monte-carlo estimation from the batch data:
			# q_pib_est = spibb_utils.compute_q_pib_est(gamma, nb_states, nb_actions, trajectories)

			# Computes the RL policy
			rl = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_0, model.transitions, reward_model, 'default')
			rl.fit()
			# Evaluates the RL policy performance
			perfrl = spibb.policy_evaluation_exact(rl.pi, r_reshaped, current_proba, gamma)[0][0]
			print("perf RL: " + str(perfrl))

			# Computes the Reward-adjusted MDP RL policy:
			count_state_action = 0.00001 * np.ones((nb_states, nb_actions))
			kappa = 0.003
			for [action, state, next_state, reward] in batch_traj:
				count_state_action[state, action] += 1
			ramdp_reward_model = reward_model - kappa/np.sqrt(count_state_action)
			ramdp = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_0, model.transitions, ramdp_reward_model, 'default')
			ramdp.fit()
			# Evaluates the RL policy performance
			perf_RaMDP = spibb.policy_evaluation_exact(ramdp.pi, r_reshaped, current_proba, gamma)[0][0]
			print("perf RaMDP: " + str(perf_RaMDP))

			# Computes the Robust MDP policy:
			terminal_state = 24
			delta_RobustMDP = 0.001
			rmdp = RMDP_based_alorithm(gamma, nb_states, nb_actions, delta_RobustMDP, reward_current[0].reshape((nb_states, 1)), pi_b, terminal_state)
			rmdp.fit(batch_traj)
			safety_test = rmdp.safety_test()[0]
			perf_RMDP_based_alorithm = spibb.policy_evaluation_exact(rmdp.pi_t, r_reshaped, current_proba, gamma)[0][0]
			if safety_test:
				perf_RMDP_based_alorithm_safe = perf_RMDP_based_alorithm
			else:
				perf_RMDP_based_alorithm_safe = pi_b_perf
			print("delta: "+str(delta_RobustMDP)+" ;perf RMDP_based_algorithm: " + str(perf_RMDP_based_alorithm)+" ;with_safety_test: "+str(perf_RMDP_based_alorithm_safe))

			# Computes the HCPI doubly robust policy:
			delta_HCPI = 0.9
			spi = SPI(gamma, pi_b, delta_HCPI, "student_t_test", "doubly_robust", trajectories, 0, 1, pi_b_perf, reward_current)
			pi_HCPI = spi.get_policy()
			perfHCPI_doubly_robust = spibb.policy_evaluation_exact(pi_HCPI, r_reshaped, current_proba, gamma)[0][0]
			print("delta: "+str(delta_HCPI)+" ;strategy: "+"doubly_robust"+" ;perf HCPI: " + str(perfHCPI_doubly_robust))

			for N_wedge in N_wedges:
				# Computation of the binary mask for the bootstrapped state actions
				mask = spibb.compute_mask_N_wedge(nb_states, nb_actions, N_wedge, batch_traj)
				# Computation of the model mask for the bootstrapped state actions
				masked_model = model.masked_model(mask)

				## Policy-based SPIBB ##

				# Computes the Pi_b_SPIBB policy:
				pib_SPIBB = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Pi_b_SPIBB')
				pib_SPIBB.fit()
				# Evaluates the Pi_b_SPIBB performance:
				perf_Pi_b_SPIBB = spibb.policy_evaluation_exact(pib_SPIBB.pi, r_reshaped, current_proba, gamma)[0][0]
				print("perf Pi_b_SPIBB: " + str(perf_Pi_b_SPIBB))

				# Computes the Pi_<b_SPIBB policy:
				pi_leq_b_SPIBB = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Pi_leq_b_SPIBB')
				pi_leq_b_SPIBB.fit()
				# Evaluates the Pi_<b_SPIBB performance:
				perf_Pi_leq_b_SPIBB = spibb.policy_evaluation_exact(pi_leq_b_SPIBB.pi, r_reshaped, current_proba, gamma)[0][0]
				print("perf Pi_leq_b_SPIBB: " + str(perf_Pi_leq_b_SPIBB))

				results.append([seed,gamma,nb_states,nb_actions,4,
					nb_trajectories, softmax_target_perf_ratio, baseline_target_perf_ratio,
					pi_b_perf, pi_rand_perf, pi_star_perf, perfrl, perf_RaMDP, perf_RMDP_based_alorithm,
					perfHCPI_doubly_robust, perf_Pi_b_SPIBB, perf_Pi_leq_b_SPIBB, kappa,
					delta_RobustMDP, delta_HCPI, N_wedge
				])

	df = pd.DataFrame(results, columns=['seed','gamma','nb_states','nb_actions','nb_next_state_transition',
		'nb_trajectories', 'softmax_target_perf_ratio', 'baseline_target_perf_ratio',
		'baseline_perf', 'pi_rand_perf', 'pi_star_perf', 'perfrl', 'perf_RaMDP', 
		'perf_RMDP_based_algorithm',	'perfHCPI_doubly_robust', 'perf_Pi_b_SPIBB',
		'perf_Pi_leq_b_SPIBB', 'kappa',	'delta_RobustMDP', 'delta_HCPI', 'N_wedge'])

	# Save it to an excel file
	safe_save(filename, df)

