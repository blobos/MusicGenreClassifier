import os
from os.path import exists

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from FYP.MusicGenreClassifier.DataPreprocessing.datasetmelspecprep import DatasetMelSpecPrep
# from cnn import CNNNetwork
# from FYP.MusicGenreClassifier.CRNN.CRNN.CRNN import CRNN
# from cnn_2 import CNNNetwork
from FYP.MusicGenreClassifier.CRNN.CRNN_biLSTM import network_Model
from sklearn.metrics import accuracy_score

from torchsummary import summary


#TODO: add load checkpoint (missing training accuracy)
def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    progress_bar = tqdm(data_loader, desc='Training', unit='batch')
    # for input, target in data_loader:
    for input, target in progress_bar:
        # print("target:", target.shape)
        target = F.one_hot(target, 12)
        # print("target one-hot:", target.shape)
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        progress_bar.set_postfix({'loss': loss.item()})
    progress_bar.close()

    return loss.item()


def train(model, train_dataloader, loss_fn, optimiser, device, epochs, patience):
    highest_validation_accuracy = 0
    lowest_training_loss = 100
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
            torch.save(model.state_dict(), CHECKPOINTS_DIR + "highest_val_acc.pth")
            highest_validation_accuracy = val_acc
            print(f"Highest Validation Accuracy")
        if train_loss < lowest_training_loss:
            train_wait = 0
        else:
            train_wait += 1
            if train_wait == patience:
                with open(CHECKPOINTS_DIR + "training_log.txt", "w") as f:
                    f.write(
                        f"Epoch {i + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}: "
                        f"Stopping training after {patience} epochs without improvement in training loss"
                    )
                print(f"Stopping training after {patience} epochs without improvement in training loss")
                break

        if val_loss < lowest_validation_loss:
            torch.save(model.state_dict(), CHECKPOINTS_DIR + "lowest_val_loss.pth")
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

        torch.save(model.state_dict(), CHECKPOINTS_DIR + "latest.pth")
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
        for input, target in tqdm(data_loader, total=len(data_loader)):
            target = F.one_hot(target, 12)
            input, target = input.to(device), target.to(device)

            prediction = model(input)
            val_loss += loss_fn(prediction, target).item()

            # calculate accuracy
            _, predicted_classes = torch.max(prediction, dim=1)
            # print("target:", target.shape)
            target = np.argmax(target.cpu(), axis=1)
            # print("target:", target.shape)
            # print(target)
            # print("predicted:", predicted_classes.shape)
            predicted_classes = np.argmax(predicted_classes.cpu(), axis=1)
            val_acc += accuracy_score(target, predicted_classes)

    # average over the number of batches
    val_loss /= len(data_loader)
    val_acc /= len(data_loader)

    model.train()  # set model back to training mode

    return val_loss, val_acc


if __name__ == "__main__":
    CHECKPOINTS_DIR = "/home/student/Music/1/FYP/MusicGenreClassifier/CRNN/checkpoints/"
    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)
    ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/train_annotations.csv"
    AUDIO_DIR = "/home/student/Music/1/FYP/data/train/chunks"
    # ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/mini_annotations.csv"
    # AUDIO_DIR = "/home/student/Music/1/FYP/data/mini/chunks"

    if torch.cuda.is_available():
        device = "cuda:1"
    else:
        device = "cpu"
        print(f"Using {device}")

    SAMPLE_RATE = 44100
    DURATION = 30
    NUM_SAMPLES = 44100 * 30 # = 1323000
    N_FFT = 2048
    HOP_LENGTH = 1024
    N_MELS = 128
    # NUM_FRAMES = 1 + (NUM_SAMPLES - N_FFT) // HOP_LENGTH
    # (NUM_FRAMES, N_MELS) = (1 + (NUM_SAMPLES - N_FFT) // HOP_LENGTH, N_MELS)
    # (NUM_FRAMES, N_MELS) = (1 + (1323000 - 2048) // 1024, 128) = (1292, 128)

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # mel_spectrogram = torch.nn.functional.normalize(mel_spectrogram, dim=1)

    dmsp = DatasetMelSpecPrep(ANNOTATIONS_FILE,
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              device,
                              labelled=True)

    BATCH_SIZE = 4
    EPOCHS = 200
    LEARNING_RATE = 0.0001
    PATIENCE = 50

    train_data, val_data = random_split(dmsp, [len(dmsp) - int(0.2 * len(dmsp)), int(0.2 * len(dmsp))])
    train_dataloader = DataLoader(train_data, BATCH_SIZE)
    val_dataloader = DataLoader(val_data, BATCH_SIZE, shuffle=True)

    CHECKPOINT_LOAD_PATH = "/home/student/Music/1/FYP/MusicGenreClassifier/CRNN/checkpoints/latest.pth"

    if exists(CHECKPOINT_LOAD_PATH):
        network_Model = torch.load(CHECKPOINT_LOAD_PATH)
        print("checkpoint loaded")

    # construct model and assign it to device
    model = network_Model().to(device)
    print(model)

    with open(CHECKPOINTS_DIR + "parameters.txt", 'w') as f:
        f.write("Parameters\n"
                f"Batch size: {BATCH_SIZE}\n"
                f"Epochs: {EPOCHS}\n"
                f"Patience: {PATIENCE}\n"
                f"Learning rate: {LEARNING_RATE}\n"
                "Mel Spectrogram:\n"
                f"Sample rate: {SAMPLE_RATE}\n"
                f"Number of samples: {NUM_SAMPLES}\n"
                f"n_fft(frequency resolution): {N_FFT}\n"
                f"hop_length: {HOP_LENGTH}\n"
                f"Mel bins(n_mels): {N_MELS}\n"
                f"Network:\n {model}"
                )

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)




    # train model
    train(model, train_dataloader, loss_fn, optimizer, device, EPOCHS, patience=PATIENCE)
