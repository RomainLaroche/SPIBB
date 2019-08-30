# authors: anonymized

import os
import sys
expname = sys.argv[1]
index = int(sys.argv[2])
import spibb_utils
import mazeDiscrete
import spibb
import modelTransitions
import garnets
import pandas as pd
import numpy as np
from SPI import *
from shutil import copyfile
from math import ceil, floor
from RMDP import *
spibb_utils.prt('Start of experiment')


def safe_save(filename, df):
	df.to_excel(filename + '.temp.xlsx')
	copyfile(filename + '.temp.xlsx', filename + '.xlsx')
	os.remove(filename + '.temp.xlsx')
	spibb_utils.prt(str(len(results)) + ' lines saved to ' + filename + '.xlsx')


seed = index
np.random.seed(seed)

# Definition of the environment
x_max = 5
y_max = 5
x_end = int(x_max - 1)
y_end = int(y_max - 1)
walls = [[[2.5, 0], [2.5, 3]], [[3.5, 2], [3.5, 4]]]
walls = [[[x_max/2., 0], [x_max/2., ceil(y_max/2.)]], [[x_max/2.+1., floor(y_max/2.)], [x_max/2.+1., ceil(y_max/2.)+1.]]]
maze = mazeDiscrete.Maze(x_max, y_max, walls, x_end, y_end, env_type=0)
nb_states = int(x_max * y_max)
if maze.env_type == 1:
	nb_states = nb_states * 2
nb_actions = 4

# Definition of the objective function:
gamma = 0.95
# Load the baseline policy state-action function
npy_filename = "state_action_val_used_size_" + str(int(x_max)) + "_env_type_0.npy"

Q_baseline = np.load(npy_filename)

# Compute the baseline policy:
pi_b = spibb_utils.compute_baseline(Q_baseline)

pi_behavioural = np.ones(pi_b.shape)/nb_actions


# The batch sizes:
nb_trajectories_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
N_wedges = [5,7,10,15,20,30,50,70,100]
v = np.zeros(nb_states)

# Pre-compute the true reward function in function of SxA:
current_proba = maze.transition_function
garnet = garnets.Garnets(nb_states, nb_actions, 1, self_transitions=0)
garnet.transition_function = current_proba
reward_current = garnet.compute_reward()
r_reshaped = spibb_utils.get_reward_model(current_proba, reward_current)


# Compute the baseline policy performance:
pi_b_perf = spibb.policy_evaluation_exact(pi_b, r_reshaped, current_proba, gamma)[0][0]
print("baseline_perf: " + str(pi_b_perf))


# Creates a mask that is always True for classical RL and other non policy-based SPIBB algorithms
mask_0, thres = spibb.compute_mask(nb_states, nb_actions, 1, 1, [])
mask_0 = ~mask_0

pi_star = spibb.spibb(gamma, nb_states, nb_actions, mask_0, mask_0, current_proba, r_reshaped, 'default')
pi_star.fit()
pi_star_perf = spibb.policy_evaluation_exact(pi_star.pi, r_reshaped, current_proba, gamma)[0][0]
print("pi_star_perf: " + str(pi_star_perf))

# Place to save the results
filename = 'results/' + expname + '/results_' + str(index)

results = []
if not os.path.isdir('results'):
	os.mkdir('results')
if not os.path.isdir('results/' + expname):
	os.mkdir('results/' + expname)

while True:
	for nb_trajectories in nb_trajectories_list:
		# Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
		trajectories, batch_traj = spibb_utils.generate_batch(nb_trajectories, garnet, pi_behavioural)
		spibb_utils.prt("GENERATED A DATASET OF " + str(nb_trajectories) + " TRAJECTORIES")

		# Compute the maximal likelihood model for transitions and rewards.
		# NB: the true reward function can be used for ease of implementation since it is not stochastic in our environment.
		# One should compute it from the samples when it is stochastic.
		model = modelTransitions.ModelTransitions(batch_traj, nb_states, nb_actions)
		reward_model = spibb_utils.get_reward_model(model.transitions, reward_current)

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
				nb_trajectories, 0, 0,
				pi_b_perf, 0, pi_star_perf, perfrl, perf_RaMDP, perf_RMDP_based_alorithm,
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
