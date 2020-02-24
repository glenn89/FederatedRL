# https://github.com/openai/gym/blob/master/gym/envs/__init__.py#L449
import gym
import numpy as np

from rl_main.conf.names import EnvironmentName, DeepLearningModelName
from rl_main.environments.environment import Environment
from rl_main.main_constants import DEEP_LEARNING_MODEL


class BreakoutDeterministic_v4(Environment):
    def __init__(self):
        self.env = gym.make(EnvironmentName.BREAKOUT_DETERMINISTIC_V4.value)
        super(BreakoutDeterministic_v4, self).__init__()
        self.action_shape = self.get_action_shape()
        self.state_shape = self.get_state_shape()

        self.cnn_input_height = self.state_shape[0]
        self.cnn_input_width = self.state_shape[1]
        self.cnn_input_channels = self.state_shape[2]
        self.continuous = False

        self.last_ball_lives = -1

        self.skipping_state_fq = 3
        self.skipping_state_index = 0

    @staticmethod
    def to_grayscale(img):
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    @staticmethod
    def downsample(img):
        return img[::2, ::2]

    @staticmethod
    def transform_reward(reward):
        return np.sign(reward)

    def preprocess(self, img):
        gray_frame = self.to_grayscale(self.downsample(img))

        if DEEP_LEARNING_MODEL == DeepLearningModelName.ActorCriticCNN:
            state = np.expand_dims(gray_frame, axis=0)
        elif DEEP_LEARNING_MODEL == DeepLearningModelName.ActorCriticMLP:
            state = gray_frame.flatten()
        else:
            state = None

        return state

    def get_n_states(self):
        if DEEP_LEARNING_MODEL == DeepLearningModelName.ActorCriticCNN:
            return 1, 105, 80                   # input_channels, input_height, input_width
        elif DEEP_LEARNING_MODEL == DeepLearningModelName.ActorCriticMLP:
            return 8400
        else:
            return None

    def get_n_actions(self):
        return self.env.action_space.n - 1

    @property
    def action_meanings(self):
        action_meanings = self.env.get_action_meanings()
        action_meanings.remove('FIRE')
        return action_meanings

    def get_state_shape(self):
        state_shape = (int(self.env.observation_space.shape[0]/2), int(self.env.observation_space.shape[1]/2), 1)
        return state_shape

    def get_action_shape(self):
        action_shape = self.env.action_space.n - 1
        return action_shape,

    def get_action_space(self):
        return self.env.action_space

    def reset(self):
        self.env.reset()
        next_state, reward, done, info = self.env.step(1)
        self.last_ball_lives = info['ale.lives']
        info["dead"] = False    #if a ball fall down, dead is true

        return self.preprocess(next_state)

    def step(self, action):
        if action == 1:
            env_action = 2
        elif action == 2:
            env_action = 3
        else:
            env_action = 0

        next_state, reward, done, info = self.env.step(env_action)

        if self.last_ball_lives != info['ale.lives']:
            env_action = 1
            self.last_ball_lives = info['ale.lives']
            next_state, reward, done, info = self.env.step(env_action)
            info["dead"] = True
            reward = -5.0

        # info["skipping"] = True
        # if self.skipping_state_index == self.skipping_state_fq:
        #     self.skipping_state_index = 0
        #     info["skipping"] = False

        adjusted_reward = self.transform_reward(reward)

        # self.skipping_state_index += 1

        return self.preprocess(next_state), reward, adjusted_reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
