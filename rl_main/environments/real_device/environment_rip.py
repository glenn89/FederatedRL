import time
import numpy as np

# MQTT Topic for RIP
from rl_main.environments.environment import Environment

MQTT_PUB_TO_SERVO_POWER = 'motor_power_2'
MQTT_PUB_RESET = 'reset_2'
MQTT_SUB_FROM_SERVO = 'servo_info_2'
MQTT_SUB_MOTOR_LIMIT = 'motor_limit_info_2'
MQTT_SUB_RESET_COMPLETE = 'reset_complete_2'

STATE_SIZE = 4

balance_motor_power_list = [-60, 0, 60]

PUB_ID = 0


class EnvironmentRIP(Environment):
    def __init__(self, mqtt_client):
        self.episode = 0

        self.state_space_shape = (STATE_SIZE,)
        self.action_space_shape = (len(balance_motor_power_list),)

        self.reward = 0

        self.steps = 0
        self.pendulum_radians = []
        self.state = []
        self.current_pendulum_radian = 0
        self.current_pendulum_velocity = 0
        self.current_motor_velocity = 0
        self.previous_time = 0.0

        self.is_swing_up = True
        self.is_state_changed = False
        self.is_motor_limit = False
        self.is_limit_complete = False
        self.is_reset_complete = False

        self.mqtt_client = mqtt_client
        super(EnvironmentRIP, self).__init__()

        self.n_states = self.get_n_states()
        self.n_actions = self.get_n_actions()

        self.state_shape = self.get_state_shape()
        self.action_shape = self.get_action_shape()

        self.continuous = False

    def __pub(self, topic, payload, require_response=True):
        global PUB_ID
        self.mqtt_client.publish(topic=topic, payload=payload)
        PUB_ID += 1

        if require_response:
            is_sub = False
            while not is_sub:
                if self.is_state_changed or self.is_limit_complete or self.is_reset_complete:
                    is_sub = True
                time.sleep(0.0001)

        self.is_state_changed = False
        self.is_limit_complete = False
        self.is_reset_complete = False

    def set_state(self, motor_radian, motor_velocity, pendulum_radian, pendulum_velocity):
        self.is_state_changed = True
        self.state = [pendulum_radian, pendulum_velocity, motor_radian, motor_velocity]
        # self.state = [pendulum_radian, pendulum_velocity]

        self.current_pendulum_radian = pendulum_radian
        self.current_pendulum_velocity = pendulum_velocity
        self.current_motor_velocity = motor_velocity

    def __pendulum_reset(self):
        self.__pub(
            MQTT_PUB_TO_SERVO_POWER,
            "0|pendulum_reset|{0}".format(PUB_ID),
            require_response=False
        )

    # RIP Manual Swing & Balance
    def manual_swingup_balance(self):
        self.__pub(MQTT_PUB_RESET, "reset|{0}".format(PUB_ID))

    # for restarting episode
    def wait(self):
        self.__pub(MQTT_PUB_TO_SERVO_POWER, "0|wait|{0}".format(PUB_ID))

    def get_n_states(self):
        n_states = 4
        return n_states

    def get_n_actions(self):
        n_actions = 3
        return n_actions

    def get_state_shape(self):
        state_shape = (2,)
        return state_shape

    def get_action_shape(self):
        action_shape = (3,)
        return action_shape

    @property
    def action_meanings(self):
        action_meanings = ["LEFT", "STOP", "RIGHT"]
        return action_meanings

    @property
    def action_meanings(self):
        action_meanings = ["LEFT", "STOP", "RIGHT"]
        return action_meanings

    def reset(self):
        self.steps = 0
        self.pendulum_radians = []
        self.reward = 0
        self.is_motor_limit = False

        wait_time = 1 if self.episode == 0 else 15  # if self.episode % 10 == 0 else 3
        previousTime = time.perf_counter()
        time_done = False

        while not time_done:
            currentTime = time.perf_counter()
            if currentTime - previousTime >= wait_time:
                time_done = True
            time.sleep(0.0001)

        self.__pendulum_reset()
        self.wait()
        self.manual_swingup_balance()
        self.is_motor_limit = False

        self.episode += 1
        self.previous_time = time.perf_counter()

        return np.asarray(self.state)

    def step(self, action):
        motor_power = balance_motor_power_list[int(action)]

        self.__pub(MQTT_PUB_TO_SERVO_POWER, "{0}|{1}|{2}".format(motor_power, "balance", PUB_ID))
        pendulum_radian = self.current_pendulum_radian
        pendulum_angular_velocity = self.current_pendulum_velocity

        next_state = np.asarray(self.state)
        self.reward = 1.0
        adjusted_reward = self.reward / 100
        self.steps += 1
        self.pendulum_radians.append(pendulum_radian)
        done, info = self.__isDone()

        if not done:
            while True:
                current_time = time.perf_counter()
                if current_time - self.previous_time >= 6 / 1000:
                    break
        else:
            self.wait()

        self.previous_time = time.perf_counter()

        return next_state, self.reward, adjusted_reward, done, info

    def __isDone(self):
        info = {}

        def insert_to_info(s):
            info["result"] = s

        if self.steps >= 5000:
            insert_to_info("*** Success ***")
            return True, info
        elif self.is_motor_limit:
            self.reward = 0
            insert_to_info("*** Limit position ***")
            return True, info
        elif abs(self.pendulum_radians[-1]) > 3.14 / 24:
            self.is_fail = True
            self.reward = 0
            insert_to_info("*** Success ***")
            return True, info
        else:
            insert_to_info("")
            return False, info

    def close(self):
        self.pub.publish(topic=MQTT_PUB_TO_SERVO_POWER, payload=str(0))
        # self.env.close()