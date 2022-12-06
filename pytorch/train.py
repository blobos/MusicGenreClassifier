import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
checkpointDirectory = "./checkpoint/"


class FeedForwardNet(nn.Module):  # class inherits from module

    def __init__(self):
        # invoke constructor of base class
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28 * 28, 256),  # linear==dense layer, (input features == dimension of img,output features)
            nn.ReLU(),
            nn.Linear(256, 10),  # 10 genres
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions


def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,  # if not dled, then del
        train=True,  # let Pytorch know trainset
        transform=ToTensor()  # image to tensor
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,  # if not dled, then dl
        train=False,  # let Pytorch know trainset
        transform=ToTensor()  # image to tensor
    )
    return train_data, validation_data


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
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
        torch.save(feed_forward_net.state_dict(), checkpointDirectory+ "lastest.pth")
        if i % 5 == 0:
            torch.save(feed_forward_net.state_dict(), checkpointDirectory + "epoch{i + 1}.pth")
    print("Training is done")


if __name__ == "__main__":
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded")

    # train_data_loader
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    # validation_data_loader

    # build model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device} device")
    feed_forward_net = FeedForwardNet().to(device)

    # instantiate loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)
    # train model
    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    print("Model trained and store at feedforwardnet.pth")
