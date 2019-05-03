# authors: anonymized

from __future__ import division, print_function
import numpy as np
from scipy.spatial.distance import cdist, squareform


class Maze():
	# The walls are defined using the coordinates of two points
	def __init__(self, x_max, y_max, walls, x_end, y_end, RBF_shape='small', env_type=0):
		self.x_max = x_max
		self.y_max = y_max
		self.walls = walls
		self.x_end = x_end
		self.y_end = y_end
		self.env_type = env_type
		self.good_rare_pairs = [(int(x_max*(y_max-1)), 1), (int(2 + x_max*(y_max-1)), 3), (int(1 + x_max*(y_max-2)), 0)] if env_type == 1 else None
		self.nb_states = int(x_max * y_max)
		self.nb_actions = 4
		self.set_transition_function()

	def set_transition_function(self):
		self.transition_function = np.zeros((self.nb_states, 4, self.nb_states))
		for x in range(self.x_max):
			for y in range(self.y_max):
				s = x + y*self.y_max

				if s < 24:

					if self.has_hit(x, x, y, y+1):
						s2 = s
					else:
						s2 = x + (y+1) * self.y_max
					self.transition_function[s,0,s2] += 0.75
					self.transition_function[s,1,s2] += 0.10
					self.transition_function[s,2,s2] += 0.05
					self.transition_function[s,3,s2] += 0.10

					if self.has_hit(x, x, y, y-1):
						s2 = s
					else:
						s2 = x + (y-1) * self.y_max
					self.transition_function[s,0,s2] += 0.05
					self.transition_function[s,1,s2] += 0.10
					self.transition_function[s,2,s2] += 0.75
					self.transition_function[s,3,s2] += 0.10

					if self.has_hit(x, x+1, y, y):
						s2 = s
					else:
						s2 = x + 1 + y * self.y_max
					self.transition_function[s,0,s2] += 0.10
					self.transition_function[s,1,s2] += 0.75
					self.transition_function[s,2,s2] += 0.10
					self.transition_function[s,3,s2] += 0.05

					if self.has_hit(x, x-1, y, y):
						s2 = s
					else:
						s2 = x - 1 + y * self.y_max
					self.transition_function[s,0,s2] += 0.10
					self.transition_function[s,1,s2] += 0.05
					self.transition_function[s,2,s2] += 0.10
					self.transition_function[s,3,s2] += 0.75


	# Check if the agent has hit the wall
	def has_hit(self, x, x_new, y, y_new):
		if x_new < 0 or x_new >= self.x_max or y_new < 0 or y_new >= self.y_max:
			return True
		for wall in self.walls:
			if self.intersect(x, y, x_new, y_new, wall):
				return True
		return False

	# Reward matrix
	def compute_reward(self):
		nb_states = int(self.x_max * self.y_max)
		if self.env_type == 1:
			nb_states = nb_states * 2
		R = np.zeros((nb_states, nb_states))
		for s in range(nb_states):
			R[s, nb_states - 1] = 100
			if self.env_type == 1:
				R[s, int(nb_states/2) - 1] = 100
				R[s, int(1 + self.x_max*(self.y_max-1))] = 100  # add a second closer goal
				R[s, int(1 + self.x_max*(self.y_max-1) + self.x_max*self.y_max)] = 0
			R[s, s] = -10
		return R

	# Get the current feature
	def get_features(self, x, y):
		return np.array([x, y])

	# Check if a wall was hit
	def intersect(self, x, y, x_new, y_new, coord):
		if x_new != x or y_new != y:
			if x_new == x:
				x = x + 0.01
			else:
				y = y_new + 0.1
			x_inter = ((x * y_new - y * x_new) * (coord[0][0] - coord[1][0]) - (x - x_new) * (
						coord[0][0] * coord[1][1] - coord[0][1] * coord[1][0])) / (
						          (x - x_new) * (coord[0][1] - coord[1][1]) - (y - y_new) * (coord[0][0] - coord[1][0]))
			y_inter = ((x * y_new - y * x_new) * (coord[0][1] - coord[1][1]) - (y - y_new) * (
						coord[0][0] * coord[1][1] - coord[0][1] * coord[1][0])) / (
						          (x - x_new) * (coord[0][1] - coord[1][1]) - (y - y_new) * (coord[0][0] - coord[1][0]))
			if (coord[0][0] == coord[1][0]):
				if (x_inter <= max(x, x_new)) and (x_inter >= min(x, x_new)):
					if ((y_inter <= max(coord[0][1], coord[1][1]))) and (
					(y_inter >= (min(coord[0][1], coord[1][1])))) and (y_inter <= max(y, y_new)) and (
					(y_inter >= min(y, y_new))):
						return True
			if coord[0][1] == coord[1][1]:
				if (x_inter <= max(coord[0][0], coord[1][0])) and (x_inter >= min(coord[0][0], coord[1][0])) and (
						x_inter <= max(x, x_new)) and (x_inter >= min(x, x_new)):
					if (y_inter <= max(y, y_new)) and (y_inter >= min(y, y_new)):
						return True
			if (x_inter <= max(coord[0][0], coord[1][0])) and (x_inter >= min(coord[0][0], coord[1][0])) and (
					x_inter <= max(x, x_new)) and (x_inter >= min(x, x_new)):
				if ((y_inter <= max(coord[0][1], coord[1][1]))) and ((y_inter >= (min(coord[0][1], coord[1][1])))) and (
						y_inter <= max(y, y_new)) and ((y_inter >= min(y, y_new))):
					return True
			return False
		else:
			return False

	# Return the coordinates of the intersection with the wall
	def get_intersect(self, x, x_new, y, y_new):
		walls = self.walls + [[[0, 0], [self.x_max, 0]], [[self.x_max, 0], [self.x_max, self.y_max]],
		                      [[0, 0], [0, self.y_max]], [[0, self.y_max], [self.x_max, self.y_max]]]
		if x_new != x or y_new != y:
			if x_new < 0:
				if y_new > 0 and y_new < self.y_max:
					return (0.01, y_new)
				elif y_new > self.y_max:
					return (0.01, self.y_max - 0.02)
				else:
					return (0.01, 0.01)
			if y_new < 0:
				if x_new > 0 and x_new < self.x_max:
					return (x_new, 0.01)
			elif x_new > self.x_max:
				return (self.x_max - 0.02, 0.01)
			else:
				return (0.01, 0.01)
			if y_new > self.y_max:
				if x_new < self.x_max:
					return (x_new, self.y_max - 0.02)
				else:
					return (self.x_max - 0.02, self.y_max - 0.02)
			if x_new > self.x_max:
				return (self.x_max - 0.02, y_new)
			for coord in walls:
				x_inter = ((x * y_new - y * x_new) * (coord[0][0] - coord[1][0]) - (x - x_new) * (
							coord[0][0] * coord[1][1] - coord[0][1] * coord[1][0])) / (
							          (x - x_new) * (coord[0][1] - coord[1][1]) - (y - y_new) * (
								          coord[0][0] - coord[1][0]))
				y_inter = ((x * y_new - y * x_new) * (coord[0][1] - coord[1][1]) - (y - y_new) * (
							coord[0][0] * coord[1][1] - coord[0][1] * coord[1][0])) / (
							          (x - x_new) * (coord[0][1] - coord[1][1]) - (y - y_new) * (
								          coord[0][0] - coord[1][0]))
				if (coord[0][0] == coord[1][0]):
					if (x_inter <= max(x, x_new)) and (x_inter >= min(x, x_new)):
						if ((y_inter <= max(coord[0][1], coord[1][1]))) and (
						(y_inter >= (min(coord[0][1], coord[1][1])))) and (y_inter <= max(y, y_new)) and (
						(y_inter >= min(y, y_new))):
							return (x_inter, y_inter)
				if coord[0][1] == coord[1][1]:
					if (x_inter <= max(coord[0][0], coord[1][0])) and (x_inter >= min(coord[0][0], coord[1][0])) and (
							x_inter <= max(x, x_new)) and (x_inter >= min(x, x_new)):
						if (y_inter <= max(y, y_new)) and (y_inter >= min(y, y_new)):
							return (x_inter, y_inter)
				if (x_inter <= max(coord[0][0], coord[1][0])) and (x_inter >= min(coord[0][0], coord[1][0])) and (
						x_inter <= max(x, x_new)) and (x_inter >= min(x, x_new)):
					if ((y_inter <= max(coord[0][1], coord[1][1]))) and (
					(y_inter >= (min(coord[0][1], coord[1][1])))) and (y_inter <= max(y, y_new)) and (
					(y_inter >= min(y, y_new))):
						return (x_inter, y_inter)
		else:
			return (x_new, y_new)

	# Check if the maze is complete
	def isComplete(self, x, y):
		return (x == self.x_end) and (y == self.y_end)

class Player():
	def __init__(self, x_init, y_init, maze):
		self.x_init = x_init
		self.y_init = y_init
		self.x = x_init
		self.y = y_init
		self.has_hit = False
		self.reward = 0
		self.maze = maze
		self.game_over = False
		self.env_type = maze.env_type
		self.candy_eaten = np.zeros(2)

	# Define a move for the agent
	def move(self, action):
		new_x, new_y = action(self.x, self.y)
		self.has_hit = self.maze.has_hit(self.x, new_x, self.y, new_y)
		if self.has_hit:
			self.x = self.x
			self.y = self.y
		else:
			self.x, self.y = new_x, new_y

	# Update the state for a given action
	def update_state(self, action):
		self.move(action)
		if self.has_hit:
			self.reward = -10
		elif self.maze.isComplete(self.x, self.y):
			# This value can be modified
			self.reward = 100
			self.game_over = True
		elif self.env_type == 1 and (self.x, self.y) == (1, self.maze.y_max-1):
			if not self.candy_eaten[0]:
				self.reward = 100
				self.candy_eaten[0] = 1
			else:
				self.reward = 0
				self.candy_eaten[1] = 1
		else:
			self.reward = 0

	# Return the state of the agent
	def get_current_state(self):
		itd = self.has_hit
		st = (self.x, self.y, itd, self.reward, self.game_over, self.candy_eaten)
		self.has_hit = False
		return st


class Action():
	def __init__(self, action="N"):
		self.action_dict = {0: self.north,
							1: self.east,
							2: self.south,
							3: self.west}
		self.action = action

	def __call__(self, x, y):
		return self.action_dict[self.action % 4](x, y)

	def east(self, x, y):
		choice = np.random.rand()
		if choice < 0.75:
			return (x + 1, y)
		elif choice < 0.85:
			return (x, y + 1)
		elif choice < 0.95:
			return (x, y - 1)
		else:
			return (x - 1, y)

	def west(self, x, y):
		choice = np.random.rand()
		if choice < 0.75:
			return (x - 1, y)
		elif choice < 0.85:
			return (x, y + 1)
		elif choice < 0.95:
			return (x, y - 1)
		else:
			return (x + 1, y)

	def north(self, x, y):
		choice = np.random.rand()
		if choice < 0.75:
			return (x, y + 1)
		elif choice < 0.85:
			return (x + 1, y)
		elif choice < 0.95:
			return (x - 1, y)
		else:
			return (x, y - 1)

	def south(self, x, y):
		choice = np.random.rand()
		if choice < 0.75:
			return (x, y - 1)
		elif choice < 0.85:
			return (x + 1, y)
		elif choice < 0.95:
			return (x - 1, y)
		else:
			return (x, y + 1)
