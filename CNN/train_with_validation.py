import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, random_split

from datasetmelspecprep import DatasetMelSpecPrep
from cnn import CNNNetwork

from sklearn.metrics import accuracy_score

BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/train_annotations.csv"
AUDIO_DIR = "/home/student/Music/1/FYP/data/train/chunks"

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


# def create_data_loaders(train_data, batch_size):
#     train_dataloader = DataLoader(train_data, batch_size=batch_size)
#     return train_dataloader


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

    return loss.item()




def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_loss = train_single_epoch(model, data_loader, loss_fn, optimiser, device)


        # add validation loop
        val_loss, val_acc = validate(model, val_dataloader, loss_fn, device)
        print(f"Epoch {i + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        if (i + 1) % 5 == 0:
            torch.save(cnn.state_dict(), f"model_{i + 1}.pth")
            print(f"Model saved at model_{i + 1}.pth")
        print("---------------------------")


        with open("training_log.txt", "a") as f:
            f.write(
                f"Epoch {i + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")
    print("Finished training")

def validate(model, data_loader, loss_fn, device):
    model.eval()  # set model to evaluation mode

    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)

            prediction = model(input)
            val_loss += loss_fn(prediction, target).item()

            # calculate accuracy
            _, predicted_classes = torch.max(prediction, dim=1)
            val_acc += accuracy_score(target.cpu(), predicted_classes.cpu())

    # average over the number of batches
    val_loss /= len(data_loader)
    val_acc /= len(data_loader)

    model.train()  # set model back to training mode

    return val_loss, val_acc

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

    dmsp = DatasetMelSpecPrep(ANNOTATIONS_FILE,
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              device,
                              test=False)

    train_data, val_data = random_split(dmsp, [len(dmsp) - int(0.2 * len(dmsp)), int(0.2 * len(dmsp))])
    train_dataloader = DataLoader(train_data, BATCH_SIZE)
    val_dataloader = DataLoader(val_data, BATCH_SIZE, shuffle=True)


    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "CNN.pth")
    print("Trained feed forward net saved at CNN.pth")