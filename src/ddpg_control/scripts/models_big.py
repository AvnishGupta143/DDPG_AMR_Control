import torch
import torch.nn as nn
import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Critic(nn.Module):
    """
    # ---Critic--
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 128)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.fa1 = nn.Linear(action_dim, 128)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)

        self.fca1 = nn.Linear(256, 256)
        nn.init.xavier_uniform_(self.fca1.weight)
        self.fca1.bias.data.fill_(0.01)
        
        self.fca2 = nn.Linear(256, 128)
        nn.init.xavier_uniform_(self.fca2.weight)
        self.fca2.bias.data.fill_(0.01)
        
        self.fca3 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.fca3.weight)
        self.fca3.bias.data.fill_(0.01)

        self.fca4 = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.fca4.weight)
        self.fca4.bias.data.fill_(0.01)

    def forward(self, state, action):
        xs = torch.relu(self.fc1(state))
        xa = torch.relu(self.fa1(action))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.fca1(x))
        x = torch.relu(self.fca2(x))
        x = torch.relu(self.fca3(x))
        vs = self.fca4(x)
        return vs


class Actor(nn.Module):
    """
    # ---Actor---
    """

    def __init__(self, state_dim, action_dim, action_v_max, action_w_max):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_v_max = action_v_max
        self.action_w_max = action_w_max

        self.fa1 = nn.Linear(state_dim, 128)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)

        self.fa2 = nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)
        
        self.fa3 = nn.Linear(256, 256)
        nn.init.xavier_uniform_(self.fa3.weight)
        self.fa3.bias.data.fill_(0.01)
        
        self.fa4 = nn.Linear(256, 128)
        nn.init.xavier_uniform_(self.fa4.weight)
        self.fa4.bias.data.fill_(0.01)
        
        self.fa5 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.fa5.weight)
        self.fa5.bias.data.fill_(0.01)

        self.fa6 = nn.Linear(64, action_dim)
        nn.init.xavier_uniform_(self.fa6.weight)
        self.fa6.bias.data.fill_(0.01)

    def forward(self, state):
        print(state.shape)
        x = torch.relu(self.fa1(state))
        x = torch.relu(self.fa2(x))
        x = torch.relu(self.fa3(x))
        x = torch.relu(self.fa4(x))
        x = torch.relu(self.fa5(x))
        action = self.fa6(x)
        if state.shape <= torch.Size([self.state_dim]):
            action[0] = torch.sigmoid(action[0]) * self.action_v_max
            action[1] = torch.tanh(action[1]) * self.action_w_max
        else:
            action[:, 0] = torch.sigmoid(action[:, 0]) * self.action_v_max
            action[:, 1] = torch.tanh(action[:, 1]) * self.action_w_max
        return action
