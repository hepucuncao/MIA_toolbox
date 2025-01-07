import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):

    def __init__(self, 
                 dim_in = 10, dim_hidden = 20, dim_out = 2,
                 batch_size = 100, rtn_layer = True):
        super(FCNet, self).__init__()
       
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)

        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

