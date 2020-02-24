import sys, os

idx = os.getcwd().index("{0}rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "rl{0}".format(os.sep)
sys.path.append(PROJECT_HOME)

from rl_main import rl_utils

env = rl_utils.get_environment()

if __name__ == "__main__":
    algorithm = rl_utils.get_rl_algorithm(env)
    state_values, policy, action_table = algorithm.start_iteration()

    print("State Values:\n{0}".format(state_values))
    print()
    print("Policy:\n{0}".format(policy))
    print()
    print("Action Table\n{0}".format(action_table))
    print()
