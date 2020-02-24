import sys, os
from multiprocessing import Process

idx = os.getcwd().index("{0}rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "rl{0}".format(os.sep)
sys.path.append(PROJECT_HOME)

import rl_main.utils as utils
from rl_main import rl_utils


if __name__ == "__main__":
    utils.make_output_folders()
    utils.ask_file_removal()

    env = rl_utils.get_environment()
    rl_model = rl_utils.get_rl_model(env)

    utils.print_configuration(env, rl_model)

    try:
        chief = Process(target=utils.run_chief, args=())
        chief.start()
        chief.join()
    except KeyboardInterrupt as error:
        print("=== {0:>8} is aborted by keyboard interrupt".format('Main-Chief'))
