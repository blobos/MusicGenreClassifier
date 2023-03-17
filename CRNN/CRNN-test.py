from torch import nn
# from torchinfo import summary
from torchsummary import summary


class CRNN(nn.Module):  # inherit for nn.Module (pytorch NN)

    def __init__(self):
        super().__init__()  # ????
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
        self.dense = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 14),
        )
        # dense layer output shape: (ouput ch * freq axis * time axis, classes) = flatten or (in, out)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # print(input_data.shape)
        x = self.cnn(input_data)
        x = self.flatten(x)
        print(x.shape)
        x = self.dense(x)
        # logits = self.dense(x)
        # predictions = logits
        # return predictions

if __name__ == "__main__":
    crnn = CRNN().to('cpu')
    summary(crnn, (1, 128, 44))  # mel spectrogram dim (ch, freq axis(mel bins), time axis) = input dimensions

