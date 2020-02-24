import math
import random
from collections import namedtuple, deque

import torch.optim as optim
import torch.nn.functional as F

from rl_main.main_constants import *

from rl_main import rl_utils
from rl_main.utils import print_torch
import numpy as np
import random

ALPHA = 0.1

class Monte_Carlo_Control_v0:
    def __init__(self, env, worker_id, gamma, env_render, logger, verbose):
        self.env = env

        self.worker_id = worker_id

        # discount rate
        self.gamma = gamma

        self.trajectory = []

        self.learning_rate = LEARNING_RATE

        self.env_render = env_render
        self.logger = logger
        self.verbose = verbose

        self.gamma = GAMMA

        self.Q = {}
        self.epsilon = EPSILON_START

    def check_if_state_and_all_actions_in_Q(self, state):
        is_state_and_action_in_Q = True
        if state in self.Q:
            for action in range(self.env.n_actions):
                if action not in self.Q[state]:
                    is_state_and_action_in_Q = False
                    break
        else:
            is_state_and_action_in_Q = False
        return is_state_and_action_in_Q

    def get_epsilon_greedy_action_from_Q(self, state):
        is_state_and_all_actions_in_Q = self.check_if_state_and_all_actions_in_Q(state)

        if is_state_and_all_actions_in_Q:
            if random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(np.arange(self.env.n_actions))
            else:
                max_q_value = -1000000
                greedy_action = -1
                for action in range(self.env.n_actions):
                    if self.Q[state][action] >= max_q_value:
                        max_q_value = self.Q[state][action]
                        greedy_action = action
                action = greedy_action
        else:
            action = np.random.choice(np.arange(self.env.n_actions))

        return action

    def get_episode_trajectory(self):
        trajectory = []
        state = self.env.reset()
        trajectory_with_g = []

        win = False

        while True:
            if self.env_render:
                self.env.render()

            action = self.get_epsilon_greedy_action_from_Q(state)

            next_state, reward, adjusted_reward, done, info = self.env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            if done:
                if reward > 0:
                    win = True
                break

        first_transition = trajectory[0]
        trajectory = trajectory[1:]
        trajectory.reverse()
        g = 0.0
        for state, action, reward in trajectory:
            g = g + self.gamma * reward
            trajectory_with_g.append((state, action, g))

        trajectory_with_g.reverse()
        trajectory_with_g.insert(0, first_transition)
        return trajectory_with_g, win

    def print_q_table(self):
        for state in self.Q:
            print(state)
            for action in range(self.env.n_actions):
                if action in self.Q[state]:
                    print(" action: {0} --> q_value: {1}".format(action, self.Q[state][action]))

    def on_episode(self, episode):
        if EPSILON_DECAY:
            self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * episode / EPSILON_DECAY_RATE)

        episode_trajectory, win = self.get_episode_trajectory()

        state_action_visit_table = {}
        for state, action, g in episode_trajectory:
            first_visit = True
            if state in state_action_visit_table:
                if action in state_action_visit_table[state]:
                    first_visit = False
                else:
                    state_action_visit_table[state][action] = True
            else:
                state_action_visit_table[state] = {}

            if first_visit:
                if state in self.Q:
                    if action in self.Q[state]:
                        self.Q[state][action] = self.Q[state][action] + self.learning_rate * (g - self.Q[state][action])
                    else:
                        self.Q[state][action] = ALPHA * g
                else:
                    self.Q[state] = {}
                    self.Q[state][action] = ALPHA * g

        gradients = None
        loss = 0.0
        score = 1.0 if win else 0.0

        #self.print_q_table()
        return gradients, loss, score

