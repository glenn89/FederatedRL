import glob
import math
import os
import subprocess
import sys
import numpy as np
import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True

idx = os.getcwd().index("{0}rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "rl{0}".format(os.sep)
sys.path.append(PROJECT_HOME)

from rl_main.conf.names import RLAlgorithmName, DeepLearningModelName

from rl_main.main_constants import MODE_SYNCHRONIZATION, MODE_GRADIENTS_UPDATE, MODE_PARAMETERS_TRANSFER, \
    ENVIRONMENT_ID, RL_ALGORITHM, DEEP_LEARNING_MODEL, PROJECT_HOME, PYTHON_PATH, MY_PLATFORM, OPTIMIZER, PPO_K_EPOCH, \
    HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE, device, PPO_EPSILON_CLIP, \
    PPO_VALUE_LOSS_WEIGHT, PPO_ENTROPY_WEIGHT, MODEL_SAVE, EMA_WINDOW, SEED, GAMMA, EPSILON_GREEDY_ACT, EPSILON_DECAY, \
    EPSILON_START, EPSILON_DECAY_RATE, EPSILON_END, LEARNING_RATE

torch.manual_seed(0) # set random seed


def exp_moving_average(values, window):
    """ Numpy implementation of EMA
    """
    if window >= len(values):
        if len(values) == 0:
            sma = 0.0
        else:
            sma = np.mean(np.asarray(values))
        a = [sma] * len(values)
    else:
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a = np.convolve(values, weights, mode='full')[:len(values)]
        a[:window] = a[window]
    return a


def get_conv2d_size(h, w, kernel_size, padding, stride):
    return math.floor((h - kernel_size + 2 * padding) / stride + 1), math.floor((w - kernel_size + 2 * padding) / stride + 1)


def get_pool2d_size(h, w, kernel_size, stride):
    return math.floor((h - kernel_size) / stride + 1), math.floor((w - kernel_size) / stride + 1)


def print_configuration(env, rl_model):
    print("\n*** GENERAL ***")
    print(" MODEL SAVE: {0}".format(MODEL_SAVE))
    print(" PLATFORM: {0}".format(MY_PLATFORM))
    print(" EMA WINDOW: {0}".format(EMA_WINDOW))
    print(" SEED: {0}".format(SEED))

    print("\n*** MODE ***")
    if MODE_SYNCHRONIZATION:
        print(" MODE1: [SYNCHRONOUS_COMMUNICATION] vs. ASYNCHRONOUS_COMMUNICATION")
    else:
        print(" MODE1: SYNCHRONOUS_COMMUNICATION vs. [ASYNCHRONOUS_COMMUNICATION]")

    if MODE_GRADIENTS_UPDATE:
        print(" MODE2: [GRADIENTS_UPDATE] vs. NO GRADIENTS_UPDATE")
    else:
        print(" MODE2: GRADIENTS_UPDATE vs. [NO GRADIENTS_UPDATE]")

    if MODE_PARAMETERS_TRANSFER:
        print(" MODE3: [PARAMETERS_TRANSFER] vs. NO PARAMETERS_TRANSFER")
    else:
        print(" MODE3: PARAMETERS_TRANSFER vs. [NO PARAMETERS_TRANSFER]")

    print("\n*** MY_PLATFORM & ENVIRONMENT ***")
    print(" Platform: " + MY_PLATFORM.value)
    print(" Environment Name: " + ENVIRONMENT_ID.value)
    print(" Action Space: {0} - {1}".format(env.get_n_actions(), env.action_meanings))

    print("\n*** RL ALGORITHM ***")
    print(" RL Algorithm: {0}".format(RL_ALGORITHM.value))
    if RL_ALGORITHM == RLAlgorithmName.PPO_V0:
        print(" PPO_K_EPOCH: {0}".format(PPO_K_EPOCH))
        print(" PPO_EPSILON_CLIP: {0}".format(PPO_EPSILON_CLIP))
        print(" PPO_VALUE_LOSS_WEIGHT: {0}".format(PPO_VALUE_LOSS_WEIGHT))
        print(" PPO_ENTROPY_WEIGHT: {0}".format(PPO_ENTROPY_WEIGHT))

    print("\n*** MODEL ***")
    print(" Deep Learning Model: {0}".format(DEEP_LEARNING_MODEL.value))
    if DEEP_LEARNING_MODEL == DeepLearningModelName.ActorCriticCNN:
        print(" input_width: {0}, input_height: {1}, input_channels: {2}, a_size: {3}, continuous: {4}".format(
            rl_model.input_width,
            rl_model.input_height,
            rl_model.input_channels,
            rl_model.a_size,
            rl_model.continuous
        ))
    elif DEEP_LEARNING_MODEL == DeepLearningModelName.ActorCriticMLP:
        print(" s_size: {0}, hidden_1: {1}, hidden_2: {2}, hidden_3: {3}, a_size: {4}, continuous: {5}".format(
            rl_model.s_size,
            rl_model.hidden_1_size,
            rl_model.hidden_2_size,
            rl_model.hidden_3_size,
            rl_model.a_size,
            rl_model.continuous
        ))
    elif DEEP_LEARNING_MODEL == DeepLearningModelName.NoModel:
        pass
    else:
        pass

    print("\n*** Optimizer ***")
    print(" Optimizer: {0}".format(OPTIMIZER.value))
    print(" Learning Rate: {0}".format(LEARNING_RATE))
    print(" Gamma (Discount Factor): {0}".format(GAMMA))
    print(" Epsilon Greedy Action: {0}".format(EPSILON_GREEDY_ACT))
    if EPSILON_GREEDY_ACT:
        print(" EPSILON_DECAY: {0}".format(EPSILON_DECAY))
        if EPSILON_DECAY:
            print(" EPSILON_START: {0}, EPSILON_END: {1}, EPSILON_DECAY_RATE: {2}".format(EPSILON_START, EPSILON_END, EPSILON_DECAY_RATE))

    print()
    response = input("Are you OK for All environmental variables? [y/n]: ")
    if not (response == "Y" or response == "y"):
        sys.exit(-1)


def ask_file_removal():
    print("CPU/GPU Devices:{0}".format(device))
    response = input("DELETE All Graphs, Logs, and Model Files? [y/n]: ")
    if not (response == "Y" or response == "y"):
        sys.exit(-1)

    files = glob.glob(os.path.join(PROJECT_HOME, "graphs", "*"))
    for f in files:
        os.remove(f)

    files = glob.glob(os.path.join(PROJECT_HOME, "logs", "*"))
    for f in files:
        os.remove(f)

    files = glob.glob(os.path.join(PROJECT_HOME, "out_err", "*"))
    for f in files:
        os.remove(f)

    # files = glob.glob(os.path.join(PROJECT_HOME, "model_save_files", "*"))
    # for f in files:
    #     os.remove(f)

    files = glob.glob(os.path.join(PROJECT_HOME, "save_results", "*"))
    for f in files:
        os.remove(f)


def make_output_folders():
    if not os.path.exists(os.path.join(PROJECT_HOME, "graphs")):
        os.makedirs(os.path.join(PROJECT_HOME, "graphs"))

    if not os.path.exists(os.path.join(PROJECT_HOME, "logs")):
        os.makedirs(os.path.join(PROJECT_HOME, "logs"))

    if not os.path.exists(os.path.join(PROJECT_HOME, "out_err")):
        os.makedirs(os.path.join(PROJECT_HOME, "out_err"))

    if not os.path.exists(os.path.join(PROJECT_HOME, "model_save_files")):
        os.makedirs(os.path.join(PROJECT_HOME, "model_save_files"))

    if not os.path.exists(os.path.join(PROJECT_HOME, "save_results")):
        os.makedirs(os.path.join(PROJECT_HOME, "save_results"))


def run_chief():
    try:
        # with subprocess.Popen([PYTHON_PATH, os.path.join(PROJECT_HOME, "rl_main", "chief_workers", "chief_mqtt_main.py")], shell=False, bufsize=1, stdout=sys.stdout, stderr=sys.stdout) as proc:
        #     output = ""
        #     while True:
        #         # Read line from stdout, break if EOF reached, append line to output
        #         line = proc.stdout.readline()
        #         line = line.decode()
        #         if line == "":
        #             break
        #         output += line
        os.system(PYTHON_PATH + " " + os.path.join(PROJECT_HOME, "rl_main", "chief_workers", "chief_mqtt_main.py"))
        sys.stdout = open(os.path.join(PROJECT_HOME, "out_err", "chief_stdout.out"), "wb")
        sys.stderr = open(os.path.join(PROJECT_HOME, "out_err", "chief_stderr.out"), "wb")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stdout.flush()


def run_worker(worker_id):
    try:
        os.system(PYTHON_PATH + " " + os.path.join(PROJECT_HOME, "rl_main", "chief_workers", "worker_mqtt_main.py") + " {0}".format(worker_id))
        sys.stdout = open(os.path.join(PROJECT_HOME, "out_err", "worker_{0}_stdout.out").format(worker_id), "wb")
        sys.stderr = open(os.path.join(PROJECT_HOME, "out_err", "worker_{0}_stderr.out").format(worker_id), "wb")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()


def util_init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def print_torch(torch_tensor_name, torch_tensor):
    print("{0}:{1} --> size:{2} --> require_grad:{3}".format(
        torch_tensor_name,
        torch_tensor,
        torch_tensor.size(),
        torch_tensor.requires_grad
    ))


class AddBiases(nn.Module):
    def __init__(self, bias):
        super(AddBiases, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
