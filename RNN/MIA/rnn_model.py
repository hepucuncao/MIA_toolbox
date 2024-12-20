import torch as torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRNN(nn.Module):
    def __init__(self, height, width, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        input_size = height * width  # calculate input_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)  # flatten to (batch_size, seq_length, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # out shape: (batch_size, output_size)
        return out

# Example usage:
height = 28
width = 28
hidden_size = 128
output_size = 10

rnn_model = SimpleRNN(height, width, hidden_size, output_size)

example_input = torch.randn(4, 5, 28, 28)  # Assuming input shape is (batch_size, seq_length, height, width)

output = rnn_model(example_input)
