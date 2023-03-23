import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from DataPreprocessing.Preprocess import Preprocess
from CNN import CNNNetwork1

BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 0.001
checkpointDirectory = "./checkpoint/"

ANNOTATIONS_FILE = r"/FYP/data/train/train_annotations.csv"
AUDIO_DIR = r"/FYP/data/train/chunks/"

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


# train on individual segments

def create_data_loader(train_data, batch_size):
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    return train_data_loader


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        print(inputs)
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropgate loss and updates weights
        optimiser.zero_grad()  # reset gradient after each batch
        loss.backward()  # backprop
        optimiser.step()  # update weights

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("------------------")
        torch.save(cnn.state_dict(), checkpointDirectory + "latest.pth")
        if i % 5 == 0:
            torch.save(cnn.state_dict(), checkpointDirectory + "epoch{i + 1}.pth")
    print("Training is done")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    # instantiate dataset obj and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,  # samples per sec
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )  # callable object (self.transformation)

    # constructor
    chunk = Preprocess(ANNOTATIONS_FILE,
                       AUDIO_DIR,
                       mel_spectrogram,
                       SAMPLE_RATE,
                       NUM_SAMPLES,
                       device)
    print(f"There are {len(chunk)} samples in the dataset.")
    # train_data_loader
    train_data_loader = create_data_loader(chunk, batch_size=BATCH_SIZE)
    # validation_data_loader

    cnn = CNNNetwork1().to(device)
    print(cnn)

    # instantiate loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.network_parameters(),
                                 lr=LEARNING_RATE)
    # train model
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "cnn.pth")
    print("Model trained and store at cnn.pth")
