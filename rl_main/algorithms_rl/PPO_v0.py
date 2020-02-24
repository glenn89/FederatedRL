# -*- coding: utf-8 -*-
import datetime
import sys
import time

import numpy as np
import random
import torch
import torch.nn.functional as F

from rl_main import rl_utils
from rl_main.main_constants import device, PPO_K_EPOCH, GAE_LAMBDA, PPO_EPSILON_CLIP, \
    PPO_VALUE_LOSS_WEIGHT, PPO_ENTROPY_WEIGHT, TRAJECTORY_SAMPLING, TRAJECTORY_LIMIT_SIZE, TRAJECTORY_BATCH_SIZE, \
    LEARNING_RATE


class PPO_v0:
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

        self.model = rl_utils.get_rl_model(self.env, self.worker_id)

        self.optimizer = rl_utils.get_optimizer(
            parameters=self.model.parameters(),
            learning_rate=self.learning_rate
        )

    def put_data(self, transition):
        self.trajectory.append(transition)

    def get_trajectory_data(self, sampling=False):
        # print("Before - Trajectory Size: {0}".format(len(self.trajectory)))

        state_lst, action_lst, reward_lst, next_state_lst, prob_action_lst, done_mask_lst = [], [], [], [], [], []
        if sampling:
            sampling_index = random.randrange(0, len(self.trajectory) - TRAJECTORY_BATCH_SIZE + 1)
            trajectory = self.trajectory[sampling_index : sampling_index + TRAJECTORY_BATCH_SIZE]
        else:
            trajectory = self.trajectory

        for transition in trajectory:
            s, a, r, s_prime, prob_a, done = transition

            if type(s) is np.ndarray:
                state_lst.append(s)
            else:
                state_lst.append(s.numpy())

            action_lst.append(a)
            reward_lst.append([r])

            if type(s) is np.ndarray:
                next_state_lst.append(s_prime)
            else:
                next_state_lst.append(s_prime.numpy())

            prob_action_lst.append([prob_a])

            done_mask = 0 if done else 1
            done_mask_lst.append([done_mask])

        state_lst = torch.tensor(state_lst, dtype=torch.float).to(device)
        # action_lst = torch.tensor(action_lst).to(device)
        action_lst = torch.cat(action_lst, 0).to(device)
        reward_lst = torch.tensor(reward_lst).to(device)
        next_state_lst = torch.tensor(next_state_lst, dtype=torch.float).to(device)
        done_mask_lst = torch.tensor(done_mask_lst, dtype=torch.float).to(device)
        prob_action_lst = torch.tensor(prob_action_lst).to(device)

        # print("After - Trajectory Size: {0}".format(len(self.trajectory)))

        # print("state_lst.size()", state_lst.size())
        # print("action_lst.size()", action_lst.size())
        # print("reward_lst.size()", reward_lst.size())
        # print("next_state_lst.size()", next_state_lst.size())
        # print("done_mask_lst.size()", done_mask_lst.size())
        # print("prob_action_lst.size()", prob_action_lst.size())
        return state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst, prob_action_lst

    def train_net(self):

        state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst, prob_action_lst = self.get_trajectory_data()

        loss_sum = 0.0
        for i in range(PPO_K_EPOCH):
            if TRAJECTORY_SAMPLING:
                state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst, prob_action_lst = self.get_trajectory_data(sampling=True)
            else:
                pass
            # print("WORKER: {0} - PPO_K_EPOCH: {1}/{2} - state_lst: {3}".format(self.worker_id, i+1, PPO_K_EPOCH, state_lst.size()))


            state_values = self.model.get_critic_value(state_lst)

            # discount_r_lst = []
            # discounted_reward = 0
            # for r in reversed(reward_lst):
            #     discounted_reward = r + (self.gamma * discounted_reward)
            #     discount_r_lst.insert(0, discounted_reward)
            # discount_r = torch.tensor(discount_r_lst, dtype=torch.float).to(device)
            #
            # # Normalizing the rewards:
            # discount_r = (discount_r - discount_r.mean()) / (discount_r.std() + 1e-5)
            # discount_r = discount_r.unsqueeze(dim=1)
            #
            # advantage = (discount_r - state_values).detach()

            v_target = reward_lst + self.gamma * self.model.get_critic_value(next_state_lst) * done_mask_lst

            delta = v_target - state_values
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for i, delta_t in enumerate(delta[::-1]):
                advantage = self.gamma * GAE_LAMBDA * done_mask_lst[i] * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage_lst = torch.tensor(advantage_lst, device=device, dtype=torch.float)
            advantage_lst = (advantage_lst - advantage.mean() + torch.tensor(1e-6, dtype=torch.float)) / torch.max(
                advantage_lst.std(),
                torch.tensor(1e-6, dtype=torch.float)
            )

            critic_loss = PPO_VALUE_LOSS_WEIGHT * F.smooth_l1_loss(input=state_values, target=v_target.detach())
            # critic_loss = PPO_VALUE_LOSS_WEIGHT * F.smooth_l1_loss(input=state_values, target=discount_r.detach())

            self.optimizer.zero_grad()
            critic_loss.mean().backward()
            self.optimizer.step()

            _, new_prob_action_lst, dist_entropy = self.model.evaluate_for_other_actions(state_lst, action_lst)

            ratio = torch.exp(new_prob_action_lst - prob_action_lst)  # a/b == exp(log(a)-log(b))
            surr1 = ratio * advantage_lst
            surr2 = torch.clamp(ratio, 1 - PPO_EPSILON_CLIP, 1 + PPO_EPSILON_CLIP) * advantage_lst

            # loss = -torch.mean(torch.min(surr1, surr2)) + PPO_VALUE_LOSS_WEIGHT * torch.mean(
            #     torch.mul(advantage_lst, advantage_lst)) - PPO_ENTROPY_WEIGHT * dist_entropy

            actor_loss = - torch.min(surr1, surr2).to(device) - PPO_ENTROPY_WEIGHT * dist_entropy

            self.optimizer.zero_grad()
            actor_loss.mean().backward()
            self.optimizer.step()

            loss = critic_loss + actor_loss

            # print("state_lst_mean: {0}".format(state_lst.mean()))
            # print("next_state_lst_mean: {0}".format(next_state_lst.mean()))
            # print("advantage_lst: {0}".format(advantage_lst[:3]))
            # print("pi: {0}".format(pi[:3]))
            # print("prob: {0}".format(new_prob_action_lst[:3]))
            # print("prob_action_lst: {0}".format(prob_action_lst[:3]))
            # print("new_prob_action_lst: {0}".format(new_prob_action_lst[:3]))
            # print("ratio: {0}".format(ratio[:3]))
            # print("surr1: {0}".format(surr1[:3]))
            # print("surr2: {0}".format(surr2[:3]))
            # print("entropy: {0}".format(entropy[:3]))
            # print("self.model.v(state_lst): {0}".format(self.model.v(state_lst)[:3]))
            # print("v_target: {0}".format(v_target[:3]))
            # print("F.smooth_l1_loss(self.model.v(state_lst), v_target.detach()): {0}".format(F.smooth_l1_loss(self.model.v(state_lst), v_target.detach())))
            # print("loss: {0}".format(loss[:3]))

            # params = self.model.get_parameters()
            # for layer in params:
            #     for name in params[layer]:
            #         print(layer, name, "params[layer][name]", params[layer][name])
            #         break
            #     break
            #
            # print("GRADIENT!!!")


            # actor_fc_named_parameters = self.model.actor_fc_layer.named_parameters()
            # critic_fc_named_parameters = self.model.critic_fc_layer.named_parameters()
            # for name, param in actor_fc_named_parameters:
            #     print("!!!!!!!!!!!!!! - 1 - actor", name)
            #     print(param.grad)
            # for name, param in critic_fc_named_parameters:
            #     print("!!!!!!!!!!!!!! - 2 - critic", name)
            #     print(param.grad)

            # self.optimizer.zero_grad()
            # loss.mean().backward()
            # self.optimizer.step()

            # actor_fc_named_parameters = self.model.actor_fc_layer.named_parameters()
            # critic_fc_named_parameters = self.model.critic_fc_layer.named_parameters()
            # for name, param in actor_fc_named_parameters:
            #     print("!!!!!!!!!!!!!! - 3 - actor", name)
            #     print(param.grad)
            # for name, param in critic_fc_named_parameters:
            #     print("!!!!!!!!!!!!!! - 4 - critic", name)
            #     print(param.grad)

            # self.optimizer.zero_grad()
            # loss.mean().backward()
            # self.optimize_step()

            # grads = self.model.get_gradients_for_current_parameters()
            # for layer in params:
            #     for name in params[layer]:
            #         print(layer, name, "grads[layer][name]", grads[layer][name])
            #         break
            #     break
            #
            #
            #
            # params = self.model.get_parameters()
            # for layer in params:
            #     for name in params[layer]:
            #         print(layer, name, "params[layer][name]", params[layer][name])
            #         break
            #     break

            loss_sum += loss.mean().item()

        self.trajectory.clear()

        gradients = self.model.get_gradients_for_current_parameters()
        return gradients, loss_sum / PPO_K_EPOCH

    def on_episode(self, episode):

        score = 0.0
        number_of_reset_call = 0.0

        if TRAJECTORY_SAMPLING:
            max_trajectory_len = TRAJECTORY_LIMIT_SIZE
        else:
            max_trajectory_len = 0

        while not len(self.trajectory) >= max_trajectory_len:
            done = False
            state = self.env.reset()
            number_of_reset_call += 1.0
            while not done:
                #start_time = datetime.datetime.now()
                if self.env_render:
                    self.env.render()
                action, prob = self.model.act(state)

                next_state, reward, adjusted_reward, done, info = self.env.step(action)
                if "dead" in info.keys():
                    if info["dead"]:
                        self.put_data((state, action, adjusted_reward, next_state, prob, info["dead"]))
                else:
                    self.put_data((state, action, adjusted_reward, next_state, prob, done))

                state = next_state
                score += reward
                #elapsed_time = datetime.datetime.now() - start_time

                #print(elapsed_time, " !!!")

        avrg_score = score / number_of_reset_call
        gradients, loss = self.train_net()
        #print("episode", episode, action)
        return gradients, loss, avrg_score

    def get_parameters(self):
        return self.model.get_parameters()

    def transfer_process(self, parameters, soft_transfer, soft_transfer_tau):
        self.model.transfer_process(parameters, soft_transfer, soft_transfer_tau)
