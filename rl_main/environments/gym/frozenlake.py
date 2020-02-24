import gym
import numpy as np
from rl_main.conf.names import EnvironmentName
from rl_main.environments.environment import Environment

"""
    FrozenLake-v0 environment

    The agent controls the movement of a character in a grid world. 
    
    Some tiles of the grid are walkable, and others lead to the agent falling into the water. 
    
    Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. 
    
    The agent is rewarded for finding a walkable path to a goal tile.

    The surface is described using a grid like the following:

    SFFF       (S: starting point, safe)
    FHFH       (F: frozen surface, safe)
    FFFH       (H: hole, fall to your doom)
    HFFG       (G: goal, where the frisbee is located)


    The episode ends when you reach the goal or fall in a hole. 

    the ice is slippery, so you won't always move in the direction you intend.
        
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    
    https://gym.openai.com/envs/FrozenLake-v0/
"""


class FrozenLake_v0(Environment):
    def __init__(self):
        self.env = gym.make(EnvironmentName.FROZENLAKE_V0.value, is_slippery=False)
        super(FrozenLake_v0, self).__init__()
        self.action_shape = self.get_action_shape()
        self.state_shape = self.get_state_shape()

        self.continuous = False
        self.WIN_AND_LEARN_FINISH_SCORE = 1.0
        self.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES = 25

        # self.P[a, s, s'] = Transition Probability
        gridworld = np.arange(
            self.get_n_states()
        ).reshape((4, 4))
        # terminal states
        gridworld[1, 1] = 0
        gridworld[1, 3] = 0
        gridworld[2, 3] = 0
        gridworld[3, 0] = 0
        # state transition matrix
        self.P = np.zeros((self.action_space.n,
                           self.get_n_states(),
                           self.get_n_states()))
        # any action taken in terminal state has no effect
        self.P[:, 0, 0] = 1

        for s in gridworld.flat:
            if (s != 0) and (s not in self.get_goal_states()):
                row, col = np.argwhere(gridworld == s)[0]
                for a, d in zip(
                        range(self.action_space.n),
                        [(0, -1), (1, 0), (0, 1), (-1, 0)]
                ):
                    next_row = max(0, min(row + d[0], 3))
                    next_col = max(0, min(col + d[1], 3))
                    s_prime = gridworld[next_row, next_col]
                    self.P[a, s, s_prime] = 1

        # self.R[a, s] = Rewards
        self.R = np.full((self.action_space.n,
                          self.get_n_states()), 0)
        for t in self.get_terminal_states():
            self.R[:, t] = -1
        for g in self.get_goal_states():
            self.R[:, g] = 1

    def get_n_states(self):
        n_states = self.env.observation_space.n
        return n_states

    def get_n_actions(self):
        n_actions = self.env.action_space.n
        return n_actions

    def get_state_shape(self):
        return 1,

    def get_action_shape(self):
        action_shape = (self.env.action_space.n, )
        return action_shape

    def get_action_space(self):
        return self.env.action_space

    @property
    def action_meanings(self):
        action_meanings = ["LEFT", "DOWN", "RIGHT", "UP"]
        return action_meanings

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        if "torch" in str(type(action)):
            action = int(action.item())

        next_state, reward, done, info = self.env.step(action)

        adjusted_reward = reward

        return next_state, reward, adjusted_reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def get_state(self, post_state, action):
        next_state = 0
        for i, p in enumerate(self.P[action, post_state, :]):
            if p > 0.0:
                next_state = i
        return next_state

    def get_reward(self, action, state):
        reward = self.R[action, state]
        return reward

    def get_terminal_states(self):
        return [0, 5, 7, 11, 12]

    def get_goal_states(self):
        return [15]


if __name__ == "__main__":
    env = FrozenLake_v0()

    for i_episode in range(10):
        state = env.reset()
        env.render()
        step = 0
        while True:
            print("\nstep: {0}".format(step))
            action = env.action_space.sample()
            next_state, reward, adjusted_reward, done, info = env.step(action)
            print(state, action, next_state, reward, adjusted_reward, done)
            env.render()
            step += 1
            if done:
                print('End game! Reward: ', reward)
                print('You won :)\n') if reward > 0 else print('You lost :(\n')
                break
            state = next_state

