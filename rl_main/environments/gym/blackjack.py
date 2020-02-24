import gym

from rl_main.conf.names import EnvironmentName
from rl_main.environments.environment import Environment

"""
    Simple blackjack environment

    Blackjack is a card game where the goal is to obtain cards that sum to as near as possible to 21 without going over.
    They're playing against a fixed dealer.

    - Face cards (Jack, Queen, King) have point value 10.
    - Aces can either count as 11 or 1, and it's called 'usable' at 11.

    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one face down card.

    The player can request additional cards (hit=1) until they decide to stop (stick=0) or exceed 21 (bust).

    After the player sticks, the dealer reveals their facedown card, and draws until their sum is 17 or greater.  
    If the dealer goes bust the player wins.

    If neither player nor dealer busts, the outcome (win, lose, draw) is decided by whose sum is closer to 21.  
    The reward for winning is +1, drawing is 0, and losing is -1.

    The observation of a 3-tuple of: 
      - the players current sum, 
      - the dealer's one showing card (1-10 where 1 is ace),
      - whether or not the player holds a usable ace (False or True).

    This environment corresponds to the version of the blackjack problem described in Example 5.1 
    in Reinforcement Learning: An Introduction by Sutton and Barto (1998).
    http://incompleteideas.net/sutton/book/the-book.html
"""


class Blackjack_v0(Environment):
    def __init__(self):
        self.env = gym.make(EnvironmentName.BLACKJACK_V0.value)
        super(Blackjack_v0, self).__init__()
        self.action_shape = self.get_action_shape()
        self.state_shape = self.get_state_shape()

        self.continuous = False
        self.WIN_AND_LEARN_FINISH_SCORE = 1.0
        self.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES = 25

    def get_n_states(self):
        n_states = len(self.env.observation_space)
        return n_states

    def get_n_actions(self):
        n_actions = self.env.action_space.n
        return n_actions

    def get_state_shape(self):
        state_shape = list(self.env.observation_space)
        state_shape[0] = state_shape[0]
        return tuple(state_shape)

    def get_action_shape(self):
        action_shape = (self.env.action_space.n, )
        return action_shape

    def get_action_space(self):
        return self.env.action_space

    @property
    def action_meanings(self):
        action_meanings = ["STICK", "HIT"]
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


if __name__ == "__main__":
    env = Blackjack_v0()
    print(env.n_actions)

    for i_episode in range(10):
        state = env.reset()
        while True:
            action = env.action_space.sample()
            next_state, reward, adjusted_reward, done, info = env.step(action)
            print(state, action, next_state, reward, adjusted_reward, done)
            if done:
                print('End game! Reward: ', reward)
                print('You won :)\n') if reward > 0 else print('You lost :(\n')
                break
            state = next_state

