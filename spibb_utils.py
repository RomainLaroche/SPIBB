# authors: anonymous

import numpy as np
import time


# Sectect action based on the the action-state function with a softmax strategy
def softmax_action(Q, s):
	proba=np.exp(Q[s, :])/np.exp(Q[s, :]).sum()
	nb_actions = Q.shape[1]
	return np.random.choice(nb_actions, p=proba)


# Select the best action based on the action-state function
def best_action(Q, s):
	return np.argmax(Q[s, :])


# Compute the baseline policy, which is a softmax ovec a given function Q.
def compute_baseline(Q):
	baseline = np.exp(Q)
	norm = np.sum(baseline, axis=1).reshape(Q.shape[0], 1)
	return baseline/norm


# Prints with a time stamp
def prt(s):
	format1 = ';'.join([str(0), str(30), str(41)])
	format2 = ';'.join([str(0), str(31), str(40)])
	s1 = '\x1b[%sm %s \x1b[0m' % (format1, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
	s2 = '\x1b[%sm %s \x1b[0m' % (format2, s)
	print(s1 + '  '+ s2)


# The reward function is defined on SxS, but we need it on SxA.
# This function makes the transformation based on the transition function P.
def get_reward_model(P, R):
	return np.einsum('ijk,ik->ij', P, R)


# Compute the performance of a policy given the corresponding action-state function
def compute_perf(env, gamma, Q=None, nb_trajectories=1000, max_steps=50, model=None, bootstrap=False, strategy_best=True):
	cum_rew_arr = []
	for _ in np.arange(nb_trajectories):
		isNotOver = True
		cum_rew = 0
		nb_steps = 0
		state = env.reset()
		if model != None:
			model.new_episode()
		while isNotOver and nb_steps < max_steps:
			if model != None:
				action_choice = model.predict(int(state), bootstrap)
			else:
				if strategy_best:
					action_choice = best_action(Q, int(state))
				else:
					action_choice = softmax_action(Q, int(state))
			state, reward, next_state, is_done = env.step(action_choice)
			isNotOver = not(is_done)
			cum_rew += reward*gamma**nb_steps
			nb_steps += 1
			state = next_state
		cum_rew_arr.append(cum_rew)
	expt_return = np.mean(cum_rew_arr)
	return expt_return


# Computes the monte-carlo estimation of the Q function of the behavioural policy given a batch of trajectories
def compute_q_pib_est(gamma, nb_states, nb_actions, batch):
	count_state_action = np.zeros((nb_states, nb_actions))
	q_pib_est = np.zeros((nb_states, nb_actions))
	for traj in batch:
		rev_traj = traj[::-1]
		ret = 0
		for elm in rev_traj:
			count_state_action[elm[1], elm[0]] += 1
			ret = elm[3] + gamma * ret
			q_pib_est[elm[1], elm[0]] += ret
	q_pib_est = np.divide(q_pib_est, count_state_action)
	return np.nan_to_num(q_pib_est)


# Generates a batch of trajectories
def generate_batch(nb_trajectories, env, pi, easter_egg=None, max_steps=50):
	trajectories = []
	for _ in np.arange(nb_trajectories):
		nb_steps = 0
		trajectorY = []
		state = env.reset()
		is_done = False
		while nb_steps < max_steps and not is_done:
			action_choice = np.random.choice(pi.shape[1], p=pi[state])
			state, reward, next_state, is_done = env.step(action_choice, easter_egg)
			trajectorY.append([action_choice, state, next_state, reward])
			state = next_state
			nb_steps += 1
		trajectories.append(trajectorY)
	batch_traj = [val for sublist in trajectories for val in sublist]
	return trajectories, batch_traj
