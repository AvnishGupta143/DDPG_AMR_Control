#!/usr/bin/env python

import rospy
import os
import sys

from std_msgs.msg import Float32
import torch
import torch.nn.functional as F

# Local imports
from models import Actor, Critic
import config

# ---Functions to make network updates---#

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DDPGAgent:

    def __init__(self, state_dim, action_dim, action_v_max, action_w_max, memory_buffer, path_save = "models", path_load = "models"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_v_max = action_v_max
        self.action_w_max = action_w_max
        # print('w',self.action_w_max)
        self.memory_buffer = memory_buffer
        # self.iter = 0

        self.actor = Actor(self.state_dim, self.action_dim, self.action_v_max, self.action_w_max)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_v_max, self.action_w_max)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config.LEARNING_RATE)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config.LEARNING_RATE)
        
        self.pub_qvalue = rospy.Publisher('qvalue', Float32, queue_size=5)
        
        self.qvalue = Float32()
        self.path_save = path_save
        self.path_load = path_load

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def get_action(self, state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        # print('actionploi', action)
        return action.data.numpy()

    def optimizer(self):
        s_sample, a_sample, r_sample, new_s_sample, done_sample = self.memory_buffer.sample(config.BATCH_SIZE)

        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample)
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample)
        done_sample = torch.from_numpy(done_sample)

        # -------------- optimize critic

        a_target = self.target_actor.forward(new_s_sample).detach()
        next_value = torch.squeeze(self.target_critic.forward(new_s_sample, a_target).detach())
        # y_exp = r _ gamma*Q'(s', P'(s'))
        y_expected = r_sample + (1 - done_sample) * config.GAMMA * next_value
        # y_pred = Q(s,a)
        y_predicted = torch.squeeze(self.critic.forward(s_sample, a_sample))
        # -------Publisher of Vs------
        self.qvalue = y_predicted.detach()
        self.pub_qvalue.publish(torch.max(self.qvalue))
        # print(self.qvalue, torch.max(self.qvalue))
        # ----------------------------
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ------------ optimize actor
        pred_a_sample = self.actor.forward(s_sample)
        loss_actor = -1 * torch.sum(self.critic.forward(s_sample, pred_a_sample))

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
    def update_target(self):
        soft_update(self.target_actor, self.actor, config.TAU)
        soft_update(self.target_critic, self.critic, config.TAU)

    def save_models(self, episode):
        if not os.path.isdir(f"{self.path_save}/save_agent_{episode}"):
            os.makedirs(f"{self.path_save}/save_agent_{episode}")
        
        torch.save(self.target_actor.state_dict(), f"{self.path_save}/save_agent_{episode}/actor.pt")
        torch.save(self.target_critic.state_dict(), f"{self.path_save}/save_agent_{episode}/critic.pt")
        print('****Models saved***')

    def load_models(self, episode):
        self.actor.load_state_dict(torch.load(f"{self.path_load}/save_agent_{episode}/actor.pt"))
        self.critic.load_state_dict(torch.load(f"{self.path_load}/save_agent_{episode}/critic.pt"))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('***Models load***')

# ---Mish Activation Function---#
def mish(x):
    """
        Mish: A Self Regularized Non-Monotonic Neural Activation Function
        https://arxiv.org/abs/1908.08681v1
        implemented for PyTorch / FastAI by lessw2020
        https://github.com/lessw2020/mish
        param:
            x: output of a layer of a neural network
        return: mish activation function
    """
    return x * (torch.tanh(F.softplus(x)))
