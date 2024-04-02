import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
import numpy as np


# Concatenates the state and one-hot version of an action
def _join_state_action(state, action, action_size):
    return torch.cat([state, F.one_hot(action, action_size).to(dtype=torch.float32)], dim=1)

class FeatureExtractor(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.ft_extractor = nn.Sequential(nn.Conv2d(9, 16, 5), nn.MaxPool2d(5, stride=5), nn.Conv2d(16, 32, 5), nn.MaxPool2d(5, stride=5), nn.Flatten(start_dim=1))

    def forward(self, state):
        if(len(state) == 2):
            state = torch.from_numpy(state[0]['rgb'].astype(np.float32))
        elif(torch.is_tensor(state)):
            state = state
        else:
            # print(state)
            state = torch.Tensor(state['rgb'])
        # state = state.view(3, 200, 200)
        state_n = state.type(torch.FloatTensor)
        if(len(state_n.shape) == 4):
            features = self.ft_extractor(state_n) #.squeeze(dim=1)
        else:
            features = self.ft_extractor(state_n.unsqueeze(0)) #.squeeze(dim=1)
        # print(value)
        # sffsd
        return features

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, dropout=0):
        super().__init__()
        # print(state_size, action_size, hidden_size)
        if dropout > 0:
            self.actor = nn.Sequential(nn.LazyLinear(128), nn.Dropout(p=dropout), nn.ReLU(), nn.Linear(128, 32), nn.Dropout(p=dropout), nn.ReLU(), nn.Linear(32, action_size), nn.Softmax(0))
        else:
            self.actor = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, action_size), nn.Softmax(0))

    def forward(self, state):
        policy = Categorical(self.actor(state))
        return policy

    # Calculates the log probability of an action a with the policy π(·|s) given state s
    def log_prob(self, state, action):
        return self.forward(state).log_prob(action)
    
    def entorpy(self, state):
        return self.forward(state).entropy()

class Critic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, state):
        value = self.critic(state) #.squeeze(dim=1)
        return value


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.ft_ext = FeatureExtractor(state_size, hidden_size)
        self.actor = Actor(state_size, action_size, hidden_size)
        self.critic = Critic(state_size, hidden_size)

    def forward(self, state):
        features = self.ft_ext(state)
        policy, value = self.actor(features), self.critic(features)
        return policy, value

    # Calculates the log probability of an action a with the policy π(·|s) given state s
    def log_prob(self, state, action):
        return self.actor.log_prob(state[0]['rgb'], action)
    
    def entropy(self, state):
        return self.actor.entropy(state[0]['rgb'])


def convert_traj(trajs):
    new_traj = []
    for i in trajs:
        # print(i[0])
        new_traj.append(i[0])
    return torch.tensor(new_traj)

class GAILDiscriminator(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, state_only=False):
        super().__init__()
        self.action_size, self.state_only = action_size, state_only
        # flatten_input = nn.Flatten(state_size if state_only else state_size + action_size)
        # input_layer = nn.Linear(state_size if state_only else state_size + action_size, hidden_size)
        self.ft_ext = FeatureExtractor(state_size, hidden_size)
        self.discriminator = nn.Sequential(nn.LazyLinear(128), nn.Tanh(), nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, state, action):
        # return self.discriminator(state if self.state_only else
        #                           _join_state_action(state, action, self.action_size)).squeeze(dim=1)
        print("started..........")
        features = self.ft_ext(state)
        return self.discriminator(features)

    def predict_reward(self, state, action):
        # action = F.one_hot(action, num_classes=4)
        state = convert_traj(state)
        D = self.forward(state.type(torch.FloatTensor), action) #state.type(torch.FloatTensor)
        h = torch.log(D) - torch.log1p(-D)
        return h
