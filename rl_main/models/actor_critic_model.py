# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import glob

import numpy as np
import torch
import torch.nn as nn

from rl_main.main_constants import *
from rl_main.models.distributions import DistCategorical, DistDiagGaussian

import torch.nn.functional as F
# from torch.distributions import Categorical
from random import random, randint
import math
from rl_main.utils import AddBiases, util_init, print_torch

EPS_START = 0.9     # e-greedy threshold start value
EPS_END = 0.05      # e-greedy threshold end value
EPS_DECAY = 200     # e-greedy threshold decay


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ActorCriticModel(nn.Module):
    def __init__(self, s_size, a_size, continuous, worker_id, device):
        super(ActorCriticModel, self).__init__()

        self.worker_id = worker_id

        if DEEP_LEARNING_MODEL == DeepLearningModelName.ActorCriticCNN:
            self.input_channels = s_size[0]
            self.input_height = s_size[1]
            self.input_width = s_size[2]

            self.base = CNNBase(
                input_channels=self.input_channels,
                input_height=self.input_height,
                input_width=self.input_width,
                continuous=continuous
            )
        elif DEEP_LEARNING_MODEL == DeepLearningModelName.ActorCriticMLP:
            self.base = MLPBase(
                num_inputs=s_size,
                continuous=continuous
            )

            self.s_size = s_size
            self.hidden_1_size = self.base.hidden_1_size
            self.hidden_2_size = self.base.hidden_2_size
            self.hidden_3_size = self.base.hidden_3_size
        else:
            raise NotImplementedError

        self.continuous = continuous

        self.a_size = a_size

        if self.continuous:
            self.dist = DistDiagGaussian(self.base.output_size, self.a_size)
        else:
            self.dist = DistCategorical(self.base.output_size, self.a_size)

        self.avg_gradients = {}
        self.device = device

        self.reset_average_gradients()

        self.steps_done = 0

        files = glob.glob(os.path.join(PROJECT_HOME, "model_save_files", "{0}_{1}_{2}_*".format(
            self.worker_id,
            ENVIRONMENT_ID.name,
            DEEP_LEARNING_MODEL.value,
        )))

        if self.worker_id >= 0:
            if len(files) > 1:
                print("Worker ID - {0}: Problem occurs since there are two or more save files".format(self.worker_id))
            elif len(files) == 1:
                filename = files[0]
                self.load_state_dict(torch.load(filename))
                self.eval()
                print("Worker ID - {0}: Successful Model Load From {1}".format(self.worker_id, filename))
            else:
                print("Worker ID - {0}: There is no saved model".format(self.worker_id))

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        if not (type(inputs) is torch.Tensor):
            inputs = torch.tensor([inputs], dtype=torch.float).to(self.device)
        _, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # if self.continuous:
        #     action = torch.tensor([action.item()], device=device, dtype=torch.float)
        # else:
        #     action = torch.tensor([action.item()], device=device, dtype=torch.long)

        action_log_probs = dist.log_probs(action)

        return action, action_log_probs

    def get_critic_value(self, inputs):
        critic_value, _ = self.base(inputs)
        return critic_value

    def evaluate_for_other_actions(self, inputs, actions):
        critic_value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()

        return critic_value, action_log_probs, dist_entropy

    def evaluate(self, inputs):
        critic_value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)
        return critic_value, dist.probs

    def reset_average_gradients(self):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            self.avg_gradients[layer_name] = {}
            for name, param in named_parameters:
                self.avg_gradients[layer_name][name] = torch.zeros(size=param.size()).to(self.device)

        named_parameters = self.dist.named_parameters()
        self.avg_gradients["actor_linear"] = {}
        for name, param in named_parameters:
            self.avg_gradients["actor_linear"][name] = torch.zeros(size=param.size()).to(self.device)

    def get_gradients_for_current_parameters(self):
        gradients = {}

        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            gradients[layer_name] = {}
            for name, param in named_parameters:
                gradients[layer_name][name] = param.grad

        named_parameters = self.dist.named_parameters()
        gradients["actor_linear"] = {}
        for name, param in named_parameters:
            gradients["actor_linear"][name] = param.grad

        return gradients

    def set_gradients_to_current_parameters(self, gradients):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            for name, param in named_parameters:
                param.grad = gradients[layer_name][name]

        named_parameters = self.dist.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["actor_linear"][name]

    def accumulate_gradients(self, gradients):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            for name, param in named_parameters:
                self.avg_gradients[layer_name][name] += gradients[layer_name][name]

        named_parameters = self.dist.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["actor_linear"][name] += gradients["actor_linear"][name]

    def get_average_gradients(self, num_workers):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            for name, param in named_parameters:
                self.avg_gradients[layer_name][name] /= num_workers
        named_parameters = self.dist.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["actor_linear"][name] /= num_workers

    def get_parameters(self):
        parameters = {}

        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            parameters[layer_name] = {}
            for name, param in named_parameters:
                parameters[layer_name][name] = param.data

        named_parameters = self.dist.named_parameters()
        parameters["actor_linear"] = {}
        for name, param in named_parameters:
            parameters["actor_linear"][name] = param.data

        return parameters

    def transfer_process(self, parameters, soft_transfer, soft_transfer_tau):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            for name, param in named_parameters:
                if soft_transfer:
                    param.data = param.data * soft_transfer_tau + parameters[layer_name][name] * (1 - soft_transfer_tau)
                else:
                    param.data = parameters[layer_name][name]

        named_parameters = self.dist.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["actor_linear"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["actor_linear"][name]


class MLPBase(nn.Module):
    def __init__(self, num_inputs, continuous):
        super(MLPBase, self).__init__()

        self.hidden_1_size = HIDDEN_1_SIZE
        self.hidden_2_size = HIDDEN_2_SIZE
        self.hidden_3_size = HIDDEN_3_SIZE
        self.continuous = continuous

        init_ = lambda m: util_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        if self.continuous:
            activation = nn.Tanh()
        else:
            activation = nn.LeakyReLU()

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, self.hidden_1_size)), activation,
            init_(nn.Linear(self.hidden_1_size, self.hidden_2_size)), activation,
            init_(nn.Linear(self.hidden_2_size, self.hidden_3_size)), activation,
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, self.hidden_1_size)), activation,
            init_(nn.Linear(self.hidden_1_size, self.hidden_2_size)), activation,
            init_(nn.Linear(self.hidden_2_size, self.hidden_3_size)), activation,
        )

        self.critic_linear = init_(nn.Linear(self.hidden_3_size, 1))

        self.layers_info = {'actor':self.actor, 'critic':self.critic, 'critic_linear':self.critic_linear}

        self.train()

    def forward(self, inputs):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return self.critic_linear(hidden_critic), hidden_actor

    @property
    def output_size(self):
        return self.hidden_3_size


class CNNBase(nn.Module):
    def __init__(self, input_channels, input_height, input_width, continuous):
        super(CNNBase, self).__init__()
        self.cnn_critic_hidden_1_size = CNN_CRITIC_HIDDEN_1_SIZE
        self.cnn_critic_hidden_2_size = CNN_CRITIC_HIDDEN_2_SIZE

        self.continuous = continuous

        from rl_main.utils import get_conv2d_size, get_pool2d_size
        h, w = get_conv2d_size(h=input_height, w=input_width, kernel_size=3, padding=0, stride=1)
        print(h, w)
        h, w = get_conv2d_size(h=h, w=w, kernel_size=3, padding=1, stride=1)
        print(h, w)
        h, w = get_conv2d_size(h=h, w=w, kernel_size=3, padding=1, stride=1)
        print(h, w)
        # h, w = get_conv2d_size(h=input_height, w=input_width, kernel_size=8, padding=0, stride=4)
        # h, w = get_conv2d_size(h=h, w=w, kernel_size=4, padding=0, stride=2)
        # h, w = get_conv2d_size(h=h, w=w, kernel_size=3, padding=0, stride=1)

        if self.continuous:
            init_ = lambda m: util_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                                        nn.init.calculate_gain('tanh'))
            activation = nn.Tanh()
        else:
            init_ = lambda m: util_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                                        nn.init.calculate_gain('leaky_relu'))
            activation = nn.LeakyReLU()

        self.actor = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, padding=0, stride=4)),
            activation,
            init_(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=0, stride=2)),
            activation,
            init_(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=0, stride=1)),
            activation,
            Flatten(),
            init_(nn.Linear(32 * h * w, self.cnn_critic_hidden_1_size)),
            activation
        )

        init_ = lambda m: util_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, padding=0, stride=4)),
            activation,
            init_(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=0, stride=2)),
            activation,
            init_(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=0, stride=1)),
            activation,
            Flatten(),
            init_(nn.Linear(32 * h * w, self.cnn_critic_hidden_1_size)),
            activation,
            init_(nn.Linear(self.cnn_critic_hidden_1_size, self.cnn_critic_hidden_2_size)),
            init_(nn.Linear(self.cnn_critic_hidden_2_size, 1))
        )

        self.layers_info = {'actor': self.actor, 'critic': self.critic}

        self.train()

    def forward(self, inputs):
        inputs = inputs / 255.0
        if len(inputs.size()) == 3:
            inputs = inputs.unsqueeze(0)

        hidden_actor = self.actor(inputs)
        hidden_critic = self.critic(inputs)

        return hidden_critic, hidden_actor

    @property
    def output_size(self):
        return self.cnn_critic_hidden_1_size


if __name__ == "__main__":
    cnnBase = CNNBase(input_channels=2, input_height=5, input_width=5, continuous=False)
    print(cnnBase)
