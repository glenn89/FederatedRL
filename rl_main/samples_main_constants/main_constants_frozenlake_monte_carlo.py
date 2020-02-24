from rl_main.conf.constants_general import *
from rl_main.conf.names import *

# [GENERAL]
SEED = 1
MY_PLATFORM = OSName.MAC
PYTHON_PATH = "~/anaconda/envs/rl/bin/python"
ENV_RENDER = False
MODEL_SAVE = False

# [MQTT]
MQTT_SERVER = "localhost"
MQTT_PORT = 1883
MQTT_LOG = False

# [WORKER]
NUM_WORKERS = 1

# [OPTIMIZATION]
MAX_EPISODES = 10000
GAMMA = 0.98

# [MODE]
MODE_SYNCHRONIZATION = True
MODE_GRADIENTS_UPDATE = False      # Distributed
MODE_PARAMETERS_TRANSFER = False    # Transfer

# [TRAINING]
EPSILON_GREEDY_ACT = True
EPSILON_DECAY = True
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY_RATE = 20000 # Large value means low decaying
LEARNING_RATE = 0.1

# [1. ENVIRONMENTS]
ENVIRONMENT_ID = EnvironmentName.FROZENLAKE_V0

# [2. DEEP_LEARNING_MODELS]
DEEP_LEARNING_MODEL = DeepLearningModelName.NoModel

# [3. ALGORITHMS]
RL_ALGORITHM = RLAlgorithmName.Monte_Carlo_Control_V0