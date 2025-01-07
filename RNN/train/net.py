import torch
from torch import nn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        #set initial hidden and cell states
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)

        #forward propagate LSTM
        out,_=self.lstm(x,(h0,c0))#out：tensor of shape(batch_size,seq_length,hidden_size)
        #decode the hidden state of the last time step
        out=self.fc(out[:,-1,:])
        return out
