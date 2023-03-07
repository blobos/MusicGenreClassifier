import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary as summary


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
        )
        self.flatten = nn.Flatten()
        #FIXME: find dimensions of flatten to find input size for LSTM
        self.rnn = nn.LSTM(input_size=256, hidden_size=4096, num_layers=2, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(512 * 9 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, input_data):
        x = self.cnn(input_data)
        x = self.flatten(x)
        # x = self.dense


        # RNN
        # h0 = torch.zeros(2, batch_size, 256).to(x.device)  # 2 because num_layers=2
        # c0 = torch.zeros(2, batch_size, 256).to(x.device)
        # x, _ = self.rnn(x, (h0, c0))
        # x = self.fc(x[:, -1, :])  # last timestep

        return x


if __name__ == "__main__":
    num_classes = 12
    crnn = CRNN(num_classes)
    summary(crnn.cuda(), (1, 128, 44))  # mel spectrogram dim (ch, freq axis(mel bins), time axis) = input dimensions
