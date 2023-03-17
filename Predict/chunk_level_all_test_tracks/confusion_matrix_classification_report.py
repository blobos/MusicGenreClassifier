import torch
import torchaudio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# from cnn import CNNNetwork
from cnn_vgg16 import CNNNetwork
from datasetmelspecprep import DatasetMelSpecPrep
# from train import SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "alternative_rock",
    "black_metal",
    "death_metal",
    "dreampop_rock",
    "heavy_metal",
    "house_electronic",
    "indie_rock",
    "post_rock",
    "progressive_rock",
    "punk_rock",
    "synthwave_electronic",
    "techno_electronic",
    "thrash_metal",
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
    ANNOTATIONS_FILE = "/FYP/data/test_annotations.csv"
    AUDIO_DIR = "/FYP/data/test/chunks"
    model = "/home/student/Music/1/FYP/MusicGenreClassifier/CNN/checkpoints/lowest_val_loss.pth"

     # ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/mini_train_annotations.csv"
    # AUDIO_DIR = "/home/student/Music/1/FYP/data/mini/chunks"

    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load(model)
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        n_fft=2048,
        hop_length=1024,
        n_mels=128
    )

    dmsp = DatasetMelSpecPrep(ANNOTATIONS_FILE,
                              AUDIO_DIR,
                              mel_spectrogram,
                              44100,
                              44100,
                              "cpu",
                              True)

    num_classes = len(class_mapping)
    true_labels = []
    predicted_labels = []

    interval = int(0.1 * len(dmsp))
    for i in range(len(dmsp)):
        input = dmsp[i]
        predicted, expected = predict(cnn, input, class_mapping)
        # print(predicted, expected)
        true_labels.append(expected)
        predicted_labels.append(predicted)
        if i % interval == 0:
            percent_complete = (i / len(dmsp)) * 100
            print(f"{percent_complete:.2f}% complete")


    cm = confusion_matrix(true_labels, predicted_labels, labels=class_mapping)
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm * 100, 2)
    df_cm = pd.DataFrame(cm, index=class_mapping, columns=class_mapping)

    plt.figure(figsize=(1080 / 96, 810 / 96), dpi=96)
    sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='.2f')

    plt.subplots_adjust(left=0.17, bottom=0.19, right=1, top=0.92)
    plt.xlabel("Predicted")
    plt.ylabel("Truth")

    title = model.split("/")[-1]
    plt.title(title, y=1.05)
    plt.savefig(model + "_confusion_matrix.png")
    # plt.show()
    plt.close()

    classification_report = classification_report(true_labels, predicted_labels, labels=class_mapping)
    with open(model + "_classification_report.txt", "a") as f:
        f.write(classification_report)

    print(classification_report)

