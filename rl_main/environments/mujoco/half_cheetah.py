import gym

from rl_main.conf.names import EnvironmentName
from rl_main.environments.environment import Environment


class HalfCheetah_v2(Environment):
    def __init__(self):
        self.env = gym.make(EnvironmentName.HALF_CHEETAH_V2.value)
        super(HalfCheetah_v2, self).__init__()
        self.action_shape = self.get_action_shape()
        self.state_shape = self.get_state_shape()

        self.continuous = True
        self.WIN_AND_LEARN_FINISH_SCORE = 195
        self.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES = 100

    def get_n_states(self):
        n_states = self.env.observation_space.shape[0]
        return n_states

    def get_n_actions(self):
        n_actions = self.env.action_space.shape[0]
        return n_actions

    def get_state_shape(self):
        state_shape = self.env.observation_space.shape
        return state_shape

    def get_action_shape(self):
        action_shape = (self.env.action_space.shape[0], )
        return action_shape

    def get_action_space(self):
        return self.env.action_space

    @property
    def action_meanings(self):
        action_meanings = ["-1. ~ 1.", ]
        return action_meanings

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        adjusted_reward = reward / 100

        return next_state, reward, adjusted_reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
