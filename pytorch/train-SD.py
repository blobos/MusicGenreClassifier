import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, random_split

from SubgenreDataset import SubgenreDataset
from CNN import CNNNetwork

from sklearn.metrics import accuracy_score

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "/FYP/data/train_annotations.csv"
AUDIO_DIR = "/home/student/Music/1/FYP/data/train/chunks"

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loaders(train_data, validation_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, validation_dataloader


def train_single_epoch(model, train_dataloader, loss_fn, optimizer, device):
    model.train() #train mode (model.eval() turns off dropout, batch norm uses mean and var of whole dataset, instead of batch)
    train_loss = 0
    for input, target in train_dataloader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad() #grad not accumulated from prev batch, else overfit
        output = model(input)
        loss = loss_fn(output, target)
        train_loss += loss.item() #update current loss
        loss.backward() #calculate gradients
        optimizer.step() #update parameters using gradients
    return train_loss / len(train_dataloader)

# Calculate validation loss
def validation_loss(model, validation_dataloader, loss_fn, device):
    model.eval()
    validation_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): #don't track grad
        for input, target in validation_dataloader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            validation_loss += loss_fn(output, target).item()
            _, predicted = torch.max(output.data, 1) #_ = max value (discarded), predicted (class prediction)= index of _
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return validation_loss / len(validation_dataloader), accuracy

def train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device, epochs, PATIENCE):
    best_validation_loss = float('inf')
    best_validation_acc = 0
    no_improvement = 0
    for epoch in range(1, epochs + 1):
        # train the model for one epoch
        train_loss = train_single_epoch(model, train_dataloader, loss_fn, optimizer, device)
        # evaluate the model on the validation dataset
        validation_loss_val, validation_acc = validation_loss(model, validation_dataloader, loss_fn, device)
        # log the loss and accuracy
        print(
            f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss_val:.4f}, Validation Acc: {validation_acc:.4f}")
        # check if this is the best validation loss and save the model if it is
        if validation_loss_val < best_validation_loss:
            best_validation_loss = validation_loss_val
            best_validation_acc = validation_acc
            no_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improvement += 1
            if no_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Best validation accuracy: {best_validation_acc:.4f}")

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

    train_SD = SubgenreDataset(ANNOTATIONS_FILE,
                               AUDIO_DIR,
                               mel_spectrogram,
                               SAMPLE_RATE,
                               NUM_SAMPLES,
                               device)


    valid_SD = SubgenreDataset(ANNOTATIONS_FILE,
                               AUDIO_DIR,
                               mel_spectrogram,
                               SAMPLE_RATE,
                               NUM_SAMPLES,
                               device)

    train_dataloader = create_data_loaders(train_SD, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    # train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device, epochs, PATIENCE):
    train(cnn, train_dataloader, loss_fn, optimizer, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "test.pth")
    print("Trained feed forward net saved at test.pth")