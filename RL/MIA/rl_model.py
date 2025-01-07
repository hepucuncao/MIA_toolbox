import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

class RL_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_rate=0.2):
        super(RL_model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.fc_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for layer in self.fc_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action_probs = self.forward(state)
            action_probs = torch.softmax(action_probs, dim=1)
            action = torch.argmax(action_probs).item()
            return action

    def get_action_probs(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action_probs = self.forward(state)
            action_probs = torch.softmax(action_probs, dim=1)
            return action_probs

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            value = self.forward(state)
            return value
