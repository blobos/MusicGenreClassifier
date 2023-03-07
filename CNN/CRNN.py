from cnn_vgg16 import CNNNetwork
#like transfer learning

from torch import nn
# from torchinfo import summary
from torchsummary import summary


class CNNNetwork(nn.Module):  # inherit for nn.Module (pytorch NN)

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
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.flatten = nn.Flatten()
        # self.dense = nn.Sequential(
        #     nn.Linear(512 * 9 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 14),
        # )
        # # dense layer output shape: (ouput ch * freq axis * time axis, classes) = flatten or (in, out)
        # self.softmax = nn.Softmax(dim=1)
        # self.rnn1 = nn.RNN()

    def forward(self, input_data):
        # print(input_data.shape)
        x = self.conv1(input_data)
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        # x = self.rnn1(x, 4096, 1, batch_first = True)
        # logits = self.dense(x)
        # predictions = logits
        # return predictions

if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn.cuda(), (1, 128, 44))  # mel spectrogram dim (ch, freq axis(mel bins), time axis) = input dimensions

