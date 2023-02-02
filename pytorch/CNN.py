from torch import nn
# from torchinfo import summary
from torchsummary import summary


class CNNNetwork(nn.Module):  # inherit for nn.Module (pytorch NN)

    def __init__(self):
        super().__init__()  # ????
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # since downsam pled2mono
                out_channels=16,  # 16 filters
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # since downsampled2mono
                out_channels=32,  # 32 filters
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,  # since downsampled2mono
                out_channels=64,  # 64 filters
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,  # since downsampled2mono
                out_channels=128,  # 128 filters
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        #why is flatten
        self.linear = nn.Linear(128 * 5 * 4, 14)
        # dense layer output shape: (ch * freq axis * time axis, classes) or (in, out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # print(input_data.shape)
        x = self.conv1(input_data)
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape) #torch.Size([128, 20])
        #NOT FLATTENED
        # mat1 and mat2 shapes cannot be multiplied (128x20 and 2560x10)
        # where does 2560 come from? 128 * 20
        # logits = self.linear(x)
        # predictions = self.softmax(logits)
        # return predictions
        # !!! torchinfo summary not working properly? gives errors shown above


if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn, (1, 64, 44))  # mel spectrogram dim (ch, freq axis(mel bands), time axis)
    # summary(cnn.cuda(), (1, 64, 44))