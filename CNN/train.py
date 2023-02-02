import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, random_split

from datasetmelspecprep import DatasetMelSpecPrep
from cnn import CNNNetwork

from sklearn.metrics import accuracy_score

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "/FYP/data/train_annotations.csv"
AUDIO_DIR = "/home/student/Music/1/FYP/data/train/chunks"

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loaders(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = DatasetMelSpecPrep(ANNOTATIONS_FILE,
                             AUDIO_DIR,
                             mel_spectrogram,
                             SAMPLE_RATE,
                             NUM_SAMPLES,
                             device,
                             test=False)
    
    train_dataloader = create_data_loaders(usd, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "test.pth")
    print("Trained feed forward net saved at test.pth")