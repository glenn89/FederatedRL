import numpy as np


class Value_Iteration:
    def __init__(self, env, gamma):
        self.env = env

        # discount rate
        self.gamma = gamma

        self.n_states = self.env.get_n_states()
        self.n_actions = self.env.get_n_actions()

        self.terminal_states = self.env.get_terminal_states()
        self.goal_states = self.env.get_goal_states()

        self.state_values = np.zeros([self.n_states], dtype=float)
        self.actions = [act for act in range(self.n_actions)]

        # policy evaluation
        self.delta = 0.0
        # policy evaluation threshold
        self.theta = 0.001

    def policy_evaluation(self, state_values):
        # table initialize
        next_state_values = np.zeros([self.n_states], dtype=float)

        # iteration
        for s in range(self.n_states):
            if s in self.terminal_states:
                value_t = 0.0
                next_state_values[s] = value_t
            else:
                value_t_list = []
                for a in range(self.n_actions):
                    s_ = self.env.get_state(s, a)
                    value = self.env.get_reward(a, s) + self.gamma * state_values[s_]
                    value_t_list.append(value)
                next_state_values[s] = max(value_t_list)

        return next_state_values

    def deterministic_policy(self, state_values):
        deterministic_policy = np.empty([self.n_states, self.n_actions], dtype=float)

        # get Q-func.
        for s in range(self.n_states):
            q_func_list = []
            if s in self.terminal_states:
                for a in range(self.n_actions):
                    deterministic_policy[s][a] = 0.00
            else:
                for a in range(self.n_actions):
                    s_ = self.env.get_state(s, a)
                    q_func_list.append(state_values[s_])
                max_actions = [action_v for action_v, x in enumerate(q_func_list) if x == max(q_func_list)]

                # update policy
                for act in self.actions:
                    if act in max_actions:
                        deterministic_policy[s][act] = (1 / len(max_actions))
                    else:
                        deterministic_policy[s][act] = 0.00

        return deterministic_policy

    def start_iteration(self):
        # policy_evaluation
        print("*** Policy Evaluation Started/Restarted ***\n")
        for i in range(1000000):
            next_state_values = self.policy_evaluation(self.state_values)
            self.delta = np.max(np.abs(self.state_values - next_state_values))
            self.state_values = next_state_values
            if self.delta < self.theta:
                print("*** Policy Evaluation Conversed at {0} iterations! ***\n".format(i))
                break
        # deterministic_policy generation
        deterministic_policy = self.deterministic_policy(self.state_values)
        print("Deterministic Policy Generation Ended!\n\n")

        # view created action_table
        action_meanings = self.env.action_meanings
        action_table = []
        for s in range(self.n_states):
            if s in self.terminal_states:
                action_table.append('T')
            elif s in self.goal_states:
                action_table.append('G')
            else:
                idx = np.argmax(deterministic_policy[s])
                action_table.append(action_meanings[idx])

        return self.state_values, deterministic_policy, action_table
