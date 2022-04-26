#!/usr/bin/env python

import rospy
import os
import sys

from std_msgs.msg import Float32
import torch
import torch.nn.functional as F

# Local imports
import config
if config.MODEL == 0:
    from models import Actor, Critic
elif config.MODEL == 1:
    print("-----------------------USING BIG MODEL --------------------------------")
    from models_big import Actor, Critic

# ---Functions to make network updates---#

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DDPGAgent:

    def __init__(self, state_dim, action_dim, action_v_max, action_w_max, memory_buffer = None, path_save = "models", path_load = "models"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_v_max = action_v_max
        self.action_w_max = action_w_max
        self.memory_buffer = memory_buffer
        self.is_cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        

        self.actor = Actor(self.state_dim, self.action_dim, self.action_v_max, self.action_w_max).to(self.device)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_v_max, self.action_w_max).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config.ACTOR_LR)

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config.CRITIC_LR)
        
        self.pub_qvalue = rospy.Publisher('qvalue', Float32, queue_size=5)
        
        self.qvalue = Float32()
        self.path_save = path_save
        self.path_load = path_load
        self.critic_loss = -1
        self.actor_loss = -1

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def get_action(self, state):
        state = torch.from_numpy(state).to(self.device)
        action = self.actor.forward(state).detach()
        return action.data.cpu().numpy()

    def learn(self):
        s_sample, a_sample, r_sample, new_s_sample, done_sample = self.memory_buffer.sample(config.BATCH_SIZE)

        s_sample = torch.from_numpy(s_sample).to(self.device)
        a_sample = torch.from_numpy(a_sample).to(self.device)
        r_sample = torch.from_numpy(r_sample).to(self.device)
        new_s_sample = torch.from_numpy(new_s_sample).to(self.device)
        done_sample = torch.from_numpy(done_sample).to(self.device)

        # -------------- optimize critic ------------------
        target_actions = self.target_actor.forward(new_s_sample).detach()
        target_critic_values = torch.squeeze(self.target_critic.forward(new_s_sample, target_actions).detach())
        critic_value = torch.squeeze(self.critic.forward(s_sample, a_sample))

        # self.qvalue = critic_value.detach()
        # self.pub_qvalue.publish(torch.max(self.qvalue))
        
        target = r_sample + (1 - done_sample) * config.GAMMA * target_critic_values
        critic_loss = F.smooth_l1_loss(critic_value, target)
        
        self.critic_loss = critic_loss.data.cpu().numpy()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        

        print(torch.norm(torch.cat([p.grad.view(-1) for p in self.critic.parameters()])))
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 2)
        print(torch.norm(torch.cat([p.grad.view(-1) for p in self.critic.parameters()])))

        self.critic_optimizer.step()


        # ------------ optimize actor ------------------
        policy_actions = self.actor.forward(s_sample)
        actor_loss = -1 * torch.sum(self.critic.forward(s_sample, policy_actions))
        self.actor_loss = actor_loss.data.cpu().numpy()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        

        print(torch.norm(torch.cat([p.grad.view(-1) for p in self.actor.parameters()])))

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 2)
        
        print(torch.norm(torch.cat([p.grad.view(-1) for p in self.actor.parameters()])))


        self.actor_optimizer.step()
        
    def get_critic_loss(self):
        return self.critic_loss
        
    def get_actor_loss(self):
        return self.actor_loss
        
    def update_target(self):
        soft_update(self.target_actor, self.actor, config.TAU)
        soft_update(self.target_critic, self.critic, config.TAU)

    def save_models(self, steps):
        if not os.path.isdir(f"{self.path_save}/save_agent_{steps}"):
            os.makedirs(f"{self.path_save}/save_agent_{steps}")
        
        torch.save(self.target_actor.state_dict(), f"{self.path_save}/save_agent_{steps}/actor.pt")
        torch.save(self.target_critic.state_dict(), f"{self.path_save}/save_agent_{steps}/critic.pt")
        print('****Models saved***')

    def load_models(self, steps):
        self.actor.load_state_dict(torch.load(f"{self.path_load}/save_agent_{steps}/actor.pt"))
        self.critic.load_state_dict(torch.load(f"{self.path_load}/save_agent_{steps}/critic.pt"))
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
