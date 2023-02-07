import torch
import torchaudio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from cnn import CNNNetwork
from datasetmelspecprep import DatasetMelSpecPrep
from train import SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "alternative_rock",
    "black_Metal",
    "death_Metal",
    "dreampop_rock",
    "heavy_Metal",
    "house_electronic",
    "indie_rock",
    "post_rock",
    "progressive_rock",
    "punk_rock",
    "synthwave_electronic",
    "techno_electronic",
    "thrash_Metal",
    "trance_electronic"
]


def predict(model, input, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input[0].unsqueeze_(0))
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[input[1]]
    return predicted, expected


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/test_annotations.csv"
    AUDIO_DIR = "/home/student/Music/1/FYP/data/test/chunks"
    model = "/home/student/Music/1/FYP/MusicGenreClassifier/CNN/trained/checkpoints_14_Epoch_no_val_improvement_in_10/lowest_val_Loss_.pth"

    title = model.split("/")[-1]
    print(type(title))
    # ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/mini_train_annotations.csv"
    # AUDIO_DIR = "/home/student/Music/1/FYP/data/miniDataset/chunks"

    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load(model)
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
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
                              "cpu",
                              True)

    num_classes = len(class_mapping)
    true_labels = []
    predicted_labels = []

    interval = 0.05 * len(dmsp)
    for i in range(len(dmsp)):
        input = dmsp[i]
        predicted, expected = predict(cnn, input, class_mapping)
        # print(predicted, expected)
        true_labels.append(expected)
        predicted_labels.append(predicted)
        if i % interval == 0:
            percent_complete = (i / len(dmsp)) * 100
            print(f"{percent_complete:.2f}% complete")

    # print(predicted_labels)
    # print(len(predicted_labels))
    # print(true_labels)
    # print(len(true_labels))
    #
    # print(class_mapping)
    # print(len(class_mapping))

    cm = confusion_matrix(true_labels, predicted_labels, labels=class_mapping)
    print(cm)
    df_cm = pd.DataFrame(cm, index=class_mapping, columns=class_mapping)

    sns.heatmap(df_cm, annot=True, cmap="Blues")

    plt.title(title, y=1.05)
    plt.show()
