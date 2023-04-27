import torch
import torch.nn as nn
from torchsummary import summary

# Define the audio processing parameters
sr = 44100
duration = 30
hop_length = 512
n_fft = 1024
n_mels = 64
# fmax = 8000
# n_frames = 1 + int((sr * duration - n_fft + hop_length) / hop_length)


class NetworkModel(nn.Module):
    def __init__(self):
        super().__init__()  # ????
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # since downsampled2mono
                out_channels=32,  # 32 filters
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

        # self.reshape = torch.reshape()

        self.lstm = nn.LSTM(input_size=20 * n_mels, hidden_size=1024, num_layers=1, batch_first=True)
        # self.lstm = nn.Sequential(
        #     nn.LSTM(input_size=20 * n_mels, hidden_size=1024, num_layers=1, batch_first=True),
        #     nn.LSTM(input_size=20 * n_mels, hidden_size=1024, num_layers=1, batch_first=False)
        #     # nn.ReLU()
        # )

        self.dense = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 12)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        print("input:", input_data.shape)
        x = self.conv1(input_data)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv4(x)
        print("conv4:", x.shape)
        # print(type(x))
        x = torch.permute(x, (0, 2, 1, 3))
        print("permute:", x.shape)
        # print(type(x))
        x = self.flatten(x)
        # print(x)
        print("flatten:", x.size())
        # print(type(x))
        h0 = torch.zeros(1, x.size(0), 1024).requires_grad_().to('cuda')
        c0 = torch.zeros(1, x.size(0), 1024).requires_grad_().to('cuda')
        x, _ = self.lstm(x, (h0, c0))
        # print("lstm:", type(x), x.shape)
        # print("lstm:", np.shape(x))
        logits = self.dense(x)
        predictions = logits
        return predictions


if __name__ == "__main__":
    # print("Is cuda available?", torch.cuda.is_available())
    #
    # print("Is cuDNN version:", torch.backends.cudnn.version())
    #
    # print("cuDNN enabled? ", torch.backends.cudnn.enabled)
    #
    # print("Device count?", torch.cuda.device_count())
    #
    # torch.cuda.set_device(1)
    #
    # print("Current device?", torch.cuda.current_device())
    #
    # print("Device name? ", torch.cuda.get_device_name(torch.cuda.current_device()))

    crnn = NetworkModel().cuda()
    summary(crnn, (1, 128, 44))
