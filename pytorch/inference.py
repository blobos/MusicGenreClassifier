import torch
from train import FeedForwardNet, download_mnist_datasets

# genre_mapping = [
#     "rock",
#     "classical",
#     "jazz"
# ]

##subgenre_mapping = ["black",

class_mapping = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def predict(model, input, target, class_mapping):
    model.eval()  # turns off batch norm, dropout vs model.train()
    with torch.no_grad():  # context manager: model doesn't calculate gradient
        predictions = model(input)
        # Tensor (#number of samples passed, # of classes)
        # Tensor (1,10)
        predicted_index = predictions[0].argmax(0)  # argmax argument??? 0 axis???
        # map index to class
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

        return predicted, expected


if __name__ == "__main__":
    # load model back
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("lastest.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load MNIST validation dataset
    _, validation_data = download_mnist_datasets()  # _ since not intersted in train component

    # get sample from validation dataset for inference
    input, target = validation_data[1][0], validation_data[1][1]
    # for own project since genre, subgenre have 2 targets?

    # make inference
    predicted, expected = predict(feed_forward_net, input, target,
                                  class_mapping)  # map integers to class(genre)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
