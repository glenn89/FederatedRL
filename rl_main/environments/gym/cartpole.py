import gym

from rl_main.conf.names import EnvironmentName
from rl_main.environments.environment import Environment


class CartPole_v0(Environment):
    def __init__(self):
        self.env = gym.make(EnvironmentName.CARTPOLE_V0.value)
        super(CartPole_v0, self).__init__()
        self.action_shape = self.get_action_shape()
        self.state_shape = self.get_state_shape()

        self.continuous = False
        self.WIN_AND_LEARN_FINISH_SCORE = 195
        self.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES = 100

    def get_n_states(self):
        n_states = int(self.env.observation_space.shape[0] / 2)
        return n_states

    def get_n_actions(self):
        n_actions = self.env.action_space.n
        return n_actions

    def get_state_shape(self):
        state_shape = list(self.env.observation_space.shape)
        state_shape[0] = int(state_shape[0] / 2)
        return tuple(state_shape)

    def get_action_shape(self):
        action_shape = (self.env.action_space.n, )
        return action_shape

    def get_action_space(self):
        return self.env.action_space

    @property
    def action_meanings(self):
        action_meanings = ["LEFT", "RIGHT"]
        return action_meanings

    def reset(self):
        state = self.env.reset()
        return state[2:]

    def step(self, action):
        action = int(action.item())
        next_state, reward, done, info = self.env.step(action)

        next_state = next_state[2:]
        adjusted_reward = reward / 100

        return next_state, reward, adjusted_reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class CartPole_v1(CartPole_v0):
    def __init__(self):
        self.env = gym.make(EnvironmentName.CARTPOLE_V1.value)
        super(CartPole_v1, self).__init__()
        self.env._max_episode_steps = 5000
        self.WIN_AND_LEARN_FINISH_SCORE = 2500
        self.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES = 10


