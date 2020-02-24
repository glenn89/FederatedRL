from rl_main.conf.constants_general import *
from rl_main.conf.names import *

# [GENERAL]
SEED = 1
MY_PLATFORM = OSName.MAC
PYTHON_PATH = "~/anaconda/envs/rl/bin/python"
ENV_RENDER = False
MODEL_SAVE = False

# [OPTIMIZATION]
MAX_EPISODES = 5000
GAMMA = 0.98

# [1. ENVIRONMENTS]
ENVIRONMENT_ID = EnvironmentName.FROZENLAKE_V0

# [2. DEEP_LEARNING_MODELS]
DEEP_LEARNING_MODEL = DeepLearningModelName.NoModel

# [3. ALGORITHMS]
RL_ALGORITHM = RLAlgorithmName.Policy_Iteration