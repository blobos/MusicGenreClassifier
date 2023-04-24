import os

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from FYP.MusicGenreClassifier.DataPreprocessing.datasetmelspecprep import DatasetMelSpecPrep
# from cnn import CNNNetwork
# from FYP.MusicGenreClassifier.CRNN.CRNN_39.CRNN_biLSTM import NetworkModel
from FYP.MusicGenreClassifier.CRNN.CRNN_biLSTM import NetworkModel
from sklearn.metrics import accuracy_score

from torchsummary import summary


#CNN vs CRNN: CRNN requires target to be one hot

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    progress_bar = tqdm(data_loader, desc='Training', unit='batch')
    # for input, target in data_loader:
    for input, target in progress_bar:
        # print("input:", input)
        # print("input shape:", input.shape, "type:", type(target))
        # print("target:", target)
        # print("target shape:", target.shape, "type:", type(target))
        target = F.one_hot(target, 12)
        target = target.float()
        # print("target one-hot:", target)
        # print("target one-hot shape:", target.shape, "type:", type(target))
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        # print("target:", target)
        # print("prediction:", prediction)
        # print("predictions:", prediction)
        # print("prediction shape: ", prediction.shape)
        # print(prediction.ndim)
        if prediction.ndim < 2:
            prediction = prediction.unsqueeze(0)
        #
        # prediction = torch.argmax(prediction, dim=1)
        # print("prediction", prediction)
        # print("prediction shape: ", prediction.shape)
        # print("target:", target)
        # target = torch.argmax(target, dim=1)
        # print("target:", target)
        loss = loss_fn(prediction, target)



        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        progress_bar.set_postfix({'loss': loss.item()})

    progress_bar.close()

#training accuracy
    total_correct = 0
    total_samples = 0
    progress_bar = tqdm(data_loader, desc='Training accuracy', unit='batch')
    with torch.no_grad():
        for input, target in progress_bar:
            input, target = input.to(device), target.to(device)
            prediction = model(input)
            if prediction.ndim < 2:
                prediction = prediction.unsqueeze(0)
            _, predicted_labels = torch.max(prediction, 1)

            # print(target, predicted_labels)
            total_correct += (predicted_labels == target).sum().item()
            total_samples += input.size(0)
    progress_bar.close()

    training_accuracy = total_correct / total_samples
    # print(f'Training accuracy: {training_accuracy:.4f}')
    return loss.item(), training_accuracy


def train(model, train_dataloader, loss_fn, optimiser, device, epochs, patience):
    highest_validation_accuracy = 0
    lowest_training_loss = 100
    lowest_validation_loss = 100
    wait = 0
    wait = 0
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_loss, training_accuracy = train_single_epoch(model, train_dataloader, loss_fn, optimiser, device)

        # add validation loop
        val_loss, val_acc = validate(model, val_dataloader, loss_fn, device)
        print(
            f"Epoch {i + 1}, Training Loss: {train_loss:.4f}, Training Accuracy: {training_accuracy:.4f}, "
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        if val_acc > highest_validation_accuracy:
            torch.save(cnn.state_dict(), CHECKPOINTS_DIR + "highest_val_acc.pth")
            highest_validation_accuracy = val_acc
            print(f"Highest Validation Accuracy")
        if train_loss < lowest_training_loss:
            wait = 0
        else:
            wait += 1
            if wait == patience:
                with open(CHECKPOINTS_DIR + "training_log.txt", "w") as f:
                    f.write(
                        f"Epoch {i + 1}, Training Loss: {train_loss}, Training Accuracy: {training_accuracy:.4f}, "
                        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}:"
                        f"Stopping training after {patience} epochs without improvement in training loss"
                    )
                print(f"Stopping training after {patience} epochs without improvement in training loss")
                break

        if val_loss < lowest_validation_loss:
            torch.save(cnn.state_dict(), CHECKPOINTS_DIR + "lowest_val_loss.pth")
            lowest_validation_loss = val_loss
            print(f"Lowest Validation Loss")
            wait = 0
        else:
            wait += 1
            if wait == patience:
                with open(CHECKPOINTS_DIR + "training_log.txt", "a") as f:
                    f.write(
                        f"Epoch {i + 1}, Training Loss: {train_loss}, Training Accuracy: {training_accuracy:.4f}, "
                        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}: "
                        f"Stopping training after {patience} epochs without improvement in validation loss"
                    )
                print(f"Stopping training after {patience} epochs without improvement in validation loss")
                break

        torch.save(cnn.state_dict(), CHECKPOINTS_DIR + "latest.pth")
        # print(f"Model saved as model_{i + 1}.pth")
        print("---------------------------")

        with open(CHECKPOINTS_DIR + "training_log.txt", "a") as f:
            f.write(
                f"Epoch {i + 1}, Training Loss: {train_loss}, Training Accuracy: {training_accuracy:.4f}, "
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
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
        for input, target in tqdm(data_loader, total=len(data_loader), desc='Validation', unit='batch'):
            target = F.one_hot(target, 12)
            target = target.float()
            input, target = input.to(device), target.to(device)

            prediction = model(input)
            # print("predictions:", prediction)
            # print(torch.max(prediction, dim=1))
            if prediction.ndim < 2:
                prediction = prediction.unsqueeze(0)
            val_loss += loss_fn(prediction, target).item()

            # calculate accuracy
            predicted_classes = torch.nn.functional.one_hot(torch.argmax(prediction, dim=1), num_classes=12).float()
            # print("target:",target, target.shape)
            # print("prediction:", prediction, prediction.shape)
            # target = np.argmax(target.cpu(), axis=1)
            # print("target:", target.shape)
            # print(target)
            predicted_classes = predicted_classes.cpu()
            # print("predicted_classes:", predicted_classes, predicted_classes.shape)
            # print("target:", target, target.shape)
            # predicted_classes = np.argmax(predicted_classes.cpu(), axis=1)

            # val_acc += accuracy_score(target, predicted_classes)
            val_acc += accuracy_score(target.cpu(), predicted_classes)

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

    BATCH_SIZE = 8
    EPOCHS = 200
    LEARNING_RATE = 0.0001
    PATIENCE = 50

    # print(f"There are {len(dmsp)} samples in the dataset.")
    # signal, label = dmsp[0]
    # print(dmsp[0][0])
    # print(dmsp[0][0].size())

    train_data, val_data = random_split(dmsp, [len(dmsp) - int(0.2 * len(dmsp)), int(0.2 * len(dmsp))])
    train_dataloader = DataLoader(train_data, BATCH_SIZE)
    val_dataloader = DataLoader(val_data, BATCH_SIZE, shuffle=True)

    # construct model and assign it to device
    cnn = NetworkModel().to(device)
    # cnn = CNNNetwork1().to(device)
    print(cnn)

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
                f"Network:\n {cnn}"
                )

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimizer, device, EPOCHS, patience=PATIENCE)
