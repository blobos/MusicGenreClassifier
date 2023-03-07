import torch
import torch.nn as nn
from torchsummary import summary

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn1 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn1(x, h0)
        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn2(out, h1)
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    cnn = SimpleRNN(1,4096)
    summary(cnn.cuda(), (1, 128, 44))
