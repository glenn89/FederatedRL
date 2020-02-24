from random import *
from gym_unity.envs import UnityEnv

from rl_main.conf.names import OSName, EnvironmentName
from rl_main.environments.environment import Environment


class Drone_Racing(Environment):
    worker_id = 0

    def __init__(self, platform):
        if platform == OSName.MAC:
            env_filename = EnvironmentName.DRONE_RACING_MAC.value
        elif platform == OSName.WINDOWS:
            env_filename = EnvironmentName.DRONE_RACING_WINDOWS.value
        else:
            env_filename = None

        self.env = UnityEnv(
            environment_filename=env_filename,
            worker_id=randrange(65536),
            use_visual=False,
            multiagent=False
        ).unwrapped

        super(Drone_Racing, self).__init__()
        Drone_Racing.worker_id += 1
        self.action_shape = self.get_action_shape()
        self.action_space = self.env.action_space

        self.continuous = False

        self.skipping_state_fq = 3
        self.skipping_state_index = 0

        self.WIN_AND_LEARN_FINISH_SCORE = 200

    def get_n_states(self):
        return self.env.observation_space.shape[0]

    def get_n_actions(self):
        return self.env.action_space.shape[0]

    def get_state_shape(self):
        return self.env.observation_space

    def get_action_shape(self):
        return self.env.action_space

    def get_action_space(self):
        return self.env.action_space

    @property
    def action_meanings(self):
        action_meanings = ["FORWARD", "BACKWARD", "RIGHT", "LEFT", "UP", "DOWN", "R_ROTATE", "L_ROTATE", "HOVER"]
        return action_meanings

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        action_list = [0] * 9

        if self.is_skip_phase():
            action_list[8] = 1  # hover action
        else:
            action_list[action] = 1

        next_state, reward, done, info = self.env.step(action_list)
        adjusted_reward = reward

        info["skipping"] = True
        if not self.is_skip_phase():
            self.skipping_state_index = 0
            info["skipping"] = False

        self.skipping_state_index += 1

        return next_state, reward, adjusted_reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def is_skip_phase(self):
        return self.skipping_state_index != self.skipping_state_fq


if __name__ == "__main__":
    env = Drone_Racing(OSName.MAC)
    # Reset it, returns the starting frame
    frame = env.reset()
    print(env.get_state_shape())
    print(env.get_action_shape())
    print(frame.shape)

    # Render
    env.render()

    is_done = False
    last_frame = frame

    idx = 0
    while not is_done:
        # Perform a random action, returns the new frame, reward and whether the game is over
        action_space = [0] * 9
        action_space[randint(0, 8)] = 1
        frame, reward, adjusted_reward, is_done, _ = env.step(action_space)

        state = frame - last_frame

        # print(idx, state.mean(), reward, adjusted_reward, is_done)

        last_frame = frame
        idx = idx + 1

        # Render
        env.render()
