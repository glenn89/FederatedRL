import math
import random
from collections import namedtuple, deque

import torch.optim as optim
import torch.nn.functional as F

from rl_main.main_constants import *

from rl_main import rl_utils
from rl_main.utils import print_torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'adjusted_reward'))

TARGET_UPDATE_PERIOD = 10


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, adjusted_reward):
        state = torch.tensor(state, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        adjusted_reward = torch.tensor([[adjusted_reward]]).to(device)
        self.memory.append(Transition(state, action, next_state, adjusted_reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_v0:
    def __init__(self, env, worker_id, gamma, env_render, logger, verbose):
        self.env = env

        self.worker_id = worker_id

        # discount rate
        self.gamma = gamma

        self.trajectory = []

        # learning rate
        self.learning_rate = LEARNING_RATE

        self.env_render = env_render
        self.logger = logger
        self.verbose = verbose

        self.policy_model = rl_utils.get_rl_model(self.env).to(device)
        self.target_model = rl_utils.get_rl_model(self.env).to(device)

        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        self.optimizer = rl_utils.get_optimizer(
            parameters=self.policy_model.parameters(),
            learning_rate=self.learning_rate
        )

        self.memory = ReplayMemory(10000)
        self.steps_done = 0

        self.model = self.policy_model

    def on_episode(self, episode):
        state = self.env.reset()

        done = False
        score = 0.0

        while not done:
            if self.env_render:
                self.env.render()

            action = self.select_epsilon_greedy_action(state)

            next_state, reward, adjusted_reward, done, info = self.env.step(action)

            # Store the transition in memory
            self.memory.push(state, action, next_state, adjusted_reward)

            # Move to the next state
            state = next_state
            score += reward

        gradients, loss = self.train_net()

        # Update the target network, copying all weights and biases in DQN
        if episode % TARGET_UPDATE_PERIOD == 0:
            self.target_model.load_state_dict(self.policy_model.state_dict())

        return gradients, loss, score

    # epsilon greedy policy
    def select_epsilon_greedy_action(self, state):
        sample = random.random()
        epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.steps_done / EPSILON_DECAY_RATE)
        self.steps_done += 1
        if sample > epsilon_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action, _ = self.policy_model.act(state)
                return action.unsqueeze(0)
        else:
            return torch.tensor([[random.randrange(self.env.n_actions)]], device=device, dtype=torch.long)

    def train_net(self):
        if len(self.memory) < DQN_BATCH_SIZE:
            return

        transitions = self.memory.sample(DQN_BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool
        )

        non_final_next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).unsqueeze(dim=1)
        state_batch = torch.cat(batch.state).unsqueeze(dim=1)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.adjusted_reward)

        # print("non_final_next_state_batch.size():", non_final_next_state_batch.size())
        # print("state_batch.size():", state_batch.size())
        # print("action_batch.size():", action_batch.size())
        # print("reward_batch.size()", reward_batch.size())

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        critic_values, actor_values = self.policy_model.evaluate(state_batch)
        q_values = actor_values.gather(1, action_batch)
        # print_torch("critic_values", critic_values)
        # print_torch("q_values", q_values)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_critic_value, next_action_probs = self.target_model.evaluate(non_final_next_state_batch)
        next_state_values = torch.zeros([DQN_BATCH_SIZE, 1], device=device)
        next_state_values[non_final_mask] = next_action_probs.max(dim=1)[0].unsqueeze(dim=1).detach()

        # print_torch("next_state_values", next_state_values)
        # print_torch("reward_batch", reward_batch)

        # Compute the target Q values
        target_q_values = (next_state_values * self.gamma) + reward_batch
        # print_torch("target_q_values", target_q_values)

        # Compute the target critic values (advantage)
        critic_target_values = torch.zeros([DQN_BATCH_SIZE, 1], device=device)
        critic_target_values[non_final_mask] = next_critic_value.detach()
        target_critic_values = (critic_target_values * self.gamma) + reward_batch
        # print_torch("critic_target_values", critic_target_values)

        delta = target_critic_values - critic_values
        # print_torch("delta", delta)
        # print_torch("delta_0", delta[0])

        advantage_batch = torch.zeros([DQN_BATCH_SIZE, 1], device=device)
        advantage = 0.0
        reverse_index_list = list(range(delta.size()[0]))[::-1]
        for idx_t in reverse_index_list:
            advantage = self.gamma * GAE_LAMBDA * advantage + delta[idx_t]
            advantage_batch[idx_t] = advantage
        advantage_batch = (advantage_batch - advantage_batch.mean()) / torch.max(advantage_batch.std(), torch.tensor(1e-6, dtype=torch.float))
        # print_torch("advantage_batch", advantage_batch)

        advantage_loss = advantage_batch.pow(2).mean()
        # print("advantage_loss.requires_grad:", advantage_loss.requires_grad)
        # print_torch("advantage_loss", advantage_loss)

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values, target_q_values) + advantage_loss
        # print(loss.requires_grad)
        # print_torch("loss", loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for name, param in self.policy_model.named_parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        gradients = self.policy_model.get_gradients_for_current_parameters()

        return gradients, loss.item()

    def get_parameters(self):
        return self.policy_model.get_parameters()

    def transfer_process(self, parameters, soft_transfer, soft_transfer_tau):
        self.policy_model.transfer_process(parameters, soft_transfer, soft_transfer_tau)