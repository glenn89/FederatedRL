import numpy as np
import random

from rl_main.main_constants import MAX_EPISODES


class Policy_Iteration:
    def __init__(self, env, gamma):
        self.env = env

        # discount rate
        self.gamma = gamma

        self.max_iteration = MAX_EPISODES

        self.n_states = self.env.get_n_states()
        self.n_actions = self.env.get_n_actions()

        self.terminal_states = self.env.get_terminal_states()
        self.goal_states = self.env.get_goal_states()

        self.state_values = np.zeros([self.n_states], dtype=float)
        self.actions = [act for act in range(self.n_actions)]
        self.policy = np.empty([self.n_states, self.n_actions], dtype=float)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if s in self.terminal_states:
                    self.policy[s][a] = 0.00
                else:
                    self.policy[s][a] = 0.25

        # policy evaluation
        self.delta = 0.0
        # policy evaluation threshold
        self.theta = 0.001
        # policy stable verification
        self.is_policy_stable = False

    def policy_evaluation(self, state_values, policy):
        # table initialize
        next_state_values = np.zeros([self.n_states], dtype=float)
        # iteration
        for s in range(self.n_states):
            if s in self.terminal_states:
                value_t = 0
            else:
                value_t = 0
                for a in range(self.n_actions):
                    s_ = int(self.env.get_state(s, a))
                    value = policy[s][a] * (self.env.get_reward(a, s) + self.gamma * state_values[s_])
                    value_t += value
            next_state_values[s] = round(value_t, 3)

        return next_state_values

    def policy_improvement(self, state_values):
        new_policy = np.empty([self.n_states, self.n_actions], dtype=float)

        is_policy_stable = True

        # get Q-func.
        for s in range(self.n_states):
            q_func_list = []
            if s in self.terminal_states:
                for a in range(self.n_actions):
                    new_policy[s][a] = 0.00
            else:
                for a in range(self.n_actions):
                    s_ = int(self.env.get_state(s, a))
                    q_func_list.append(state_values[s_])
                max_actions = [action_v for action_v, x in enumerate(q_func_list) if x == max(q_func_list)]

                # update policy
                for act in self.actions:
                    if act in max_actions:
                        new_policy[s][act] = (1 / len(max_actions))
                    else:
                        new_policy[s][act] = 0.00

        if False in np.equal(self.policy, new_policy):
            is_policy_stable = False

        return is_policy_stable, new_policy

    def start_iteration(self):
        iter_num = 0
        while not self.is_policy_stable and iter_num < self.max_iteration:
            # policy_evaluation
            print("*** Policy Evaluation Started/Restarted ***\n")
            for i in range(1000000):
                next_state_values = self.policy_evaluation(self.state_values, self.policy)
                self.delta = np.max(np.abs(self.state_values - next_state_values))
                self.state_values = next_state_values
                if self.delta < self.theta:
                    print("*** Policy Evaluation Conversed at {0} iterations! ***\n".format(i))
                    break
            # policy_improvement
            self.is_policy_stable, self.policy = self.policy_improvement(self.state_values)
            iter_num += 1
            print("*** Policy Improvement --> Policy Stable: {0} ***\n".format(self.is_policy_stable))
        print("Policy Iteration Ended!\n\n")

        # view created action_table
        action_meanings = self.env.action_meanings
        action_table = []
        for s in range(self.n_states):
            if s in self.terminal_states:
                action_table.append('T')
            elif s in self.goal_states:
                action_table.append('G')
            else:
                idx = np.argmax(self.policy[s])
                action_table.append(action_meanings[idx])

        return self.state_values, self.policy, action_table
