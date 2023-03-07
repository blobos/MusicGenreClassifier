import os

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, random_split

from datasetmelspecprep import DatasetMelSpecPrep
# from cnn import CNNNetwork
from cnn_vgg16 import CNNNetwork
# from cnn_2 import CNNNetwork
from sklearn.metrics import accuracy_score





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


def train(model, train_dataloader, loss_fn, optimiser, device, epochs, patience):
    highest_validation_accuracy = 0
    lowest_training_loss =100
    lowest_validation_loss = 100
    val_wait = 0
    train_wait = 0
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_loss = train_single_epoch(model, train_dataloader, loss_fn, optimiser, device)

        # add validation loop
        val_loss, val_acc = validate(model, val_dataloader, loss_fn, device)
        print(
            f"Epoch {i + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        if val_acc > highest_validation_accuracy:
            torch.save(cnn.state_dict(), CHECKPOINTS_DIR + "highest_val_acc.pth")
            highest_validation_accuracy = val_acc
            print(f"Highest Validation Accuracy")
        if train_loss < lowest_training_loss:
            train_wait = 0
        else:
            train_wait +=1
            if train_wait == patience:
                with open(CHECKPOINTS_DIR + "training_log.txt", "a") as f:
                    f.write(
                        f"Epoch {i + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}: "
                        f"Stopping training after {patience} epochs without improvement in training loss"
                    )
                print(f"Stopping training after {patience} epochs without improvement in training loss")
                break

        if val_loss < lowest_validation_loss:
            torch.save(cnn.state_dict(), CHECKPOINTS_DIR + "lowest_val_loss.pth")
            lowest_validation_loss = val_loss
            print(f"Lowest Validation Loss")
            val_wait = 0
        else:
            val_wait += 1
            if val_wait == patience:
                with open(CHECKPOINTS_DIR + "training_log.txt", "a") as f:
                    f.write(
                        f"Epoch {i + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}: "
                        f"Stopping training after {patience} epochs without improvement in validation loss"
                    )
                print(f"Stopping training after {patience} epochs without improvement in validation loss")
                break

        torch.save(cnn.state_dict(), CHECKPOINTS_DIR + "latest.pth")
        # print(f"Model saved as model_{i + 1}.pth")
        print("---------------------------")

        with open(CHECKPOINTS_DIR + "training_log.txt", "a") as f:
            f.write(
                f"Epoch {i + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            if val_acc > highest_validation_accuracy:
                f.write(", Highest Validation Accuracy ")
            if val_loss < lowest_validation_loss:
                f.write(", Lowest Validation Loss")
            f.write("\n")
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
    CHECKPOINTS_DIR = "/home/student/Music/1/FYP/MusicGenreClassifier/CNN/checkpoints/"
    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)
    ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/train_annotations.csv"
    AUDIO_DIR = "/home/student/Music/1/FYP/data/train/chunks"
    # ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/mini_train_annotations.csv"
    # AUDIO_DIR = "/home/student/Music/1/FYP/data/miniDataset/chunks"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        print(f"Using {device}")

    SAMPLE_RATE = 44100
    NUM_SAMPLES = 44100
    N_FFT = 2048
    HOP_LENGTH = 1024
    N_MELS = 128

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    dmsp = DatasetMelSpecPrep(ANNOTATIONS_FILE,
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              device,
                              labelled=True)

    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.001

    train_data, val_data = random_split(dmsp, [len(dmsp) - int(0.2 * len(dmsp)), int(0.2 * len(dmsp))])
    train_dataloader = DataLoader(train_data, BATCH_SIZE)
    val_dataloader = DataLoader(val_data, BATCH_SIZE, shuffle=True)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    # cnn = CNNNetwork1().to(device)
    print(cnn)

    with open(CHECKPOINTS_DIR + "parameters.txt", 'a') as f:
        f.write("Parameters\n"
                f"Batch size: {BATCH_SIZE}\n"
                f"Epochs: {EPOCHS}\n"
                f"Learning rate: {LEARNING_RATE}\n"
                "Mel Spectrogram\n"
                f"Sample rate: {SAMPLE_RATE}\n"
                f"n_fft(frequency resolution): {N_FFT}\n"
                f"hop_length: {HOP_LENGTH}\n"
                f"Mel bins(n_mels): {N_MELS}\n")

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimizer, device, EPOCHS, patience=50)
