import torch
import torch.nn as nn
from torchsummary import summary

# Define the audio processing parameters
sr = 44100
duration = 30
n_fft = 2048
hop_length = n_fft/2
n_mels = 128
n_frames = 1 + int((sr * duration - n_fft + hop_length) / hop_length) # number of frames(i.e. Length)


class NetworkModel(nn.Module):
    def __init__(self):
        super().__init__()  # ????
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # since downsampled2mono
                out_channels=16,  # 32 filters
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)


        #variable input size = x.flatten[2] == x.permute.size(2) * x.permute.size(3)) ==

        self.lstm = nn.LSTM(input_size=10240, hidden_size=1024, num_layers=3, bidirectional=True,
                             batch_first= True)


        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d((1)) #1D not 2D since [Batch, C, Length] not [B, C, H, W]
        # self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.global_avg_pool = nn.AvgPool2d(kernel_size=2)
        # self.lstm2 = nn.
        # self.lstm = nn.Sequential(
        #     nn.LSTM(input_size=20 * n_mels, hidden_size=1024, num_layers=1, batch_first=True),
        #     nn.LSTM(input_size=20 * n_mels, hidden_size=1024, num_layers=1, batch_first=False)
        #     # nn.ReLU()
        # )

        self.dense = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 12)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # print("input:", input_data.shape)
        x = self.conv1(input_data)
        # print("conv1:", x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print("conv4:", x.shape)
        # print(type(x))
        x = torch.permute(x, (0, 2, 1, 3))
        # print("permute:", x.shape)
        # print(type(x))
        # print("flattened=", x.size(0), x.size(1), x.size(2)*x.size(3))
        x = self.flatten(x)
        # print(x)
        # print("flatten:", x.size())
        # print(type(x))

        #(D*num_layers, Batch, H(out))
        #D = 2 if Bidirectional, else 1
        # h_0 = torch.zeros(2, x.size(0), 1024).requires_grad_().to('cuda') #hidden state
        # print("h_0:", h_0.size())
        # c_0 = torch.zeros(2, x.size(0), 1024).requires_grad_().to('cuda') #cell state
        #for LSTM, input=(Batch,Length,Hidden(input)) when batch_first=True
        # x, _ = self.lstm1(x, (h_0, c_0)) #Defaults to zeros if (h_0, c_0) is not provided.
        # x, (h_n, c_n) = self.lstm(x)
        x, _ = self.lstm(x)
        #lstm output =  output, (h_n, c_n)
        # print("x:", x.size(), "h_n:", h_n.size(), "c_n:", c_n.size())
        # x, _ = self.lstm(x, (h_n, c_n))
        # print("lstm:", x.shape)
        # print("lstm:", type(x), x.shape)
        # print("lstm:", np.shape(x))
        x =x.transpose(1, 2) # swap 2nd and 3rd(features) dim, since avg pool is on second dimension
        # print("transpose:", x.shape)
        x = self.adaptive_avg_pool(x).squeeze() #remove middle dimension

        # print("pooling:", x.size())
        # x = self.adaptive_avg_pool(x)
        # x = x.view(x.size(0), -1)
        # print("pool collapsed:", x.size())

        # x = x[:, -1, :] #get last element to get many-to-many to get many-to-one
        # print(x.size())
        # print(x)
        logits = self.dense(x)
        # print(logits.size())
        # print(x)
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
    input_size = (1, n_mels, n_frames)
    print("input:", input_size)
    summary(crnn, (1, n_mels, n_frames))# mel spectrogram dim (ch, freq axis(mel bins), time axis)
