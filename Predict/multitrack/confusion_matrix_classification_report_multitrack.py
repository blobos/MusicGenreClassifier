import torch
import torchaudio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class_mapping = [
    "black_metal",
    "death_metal",
    "dreampop_rock",
    "heavy_metal",
    "house_electronic",
    "post_rock",
    "progressive_rock",
    "punk_rock",
    "synthwave_electronic",
    "techno_electronic",
    "thrash_metal",
    "trance_electronic"
]





if __name__ == "__main__":


     # ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/mini_train_annotations.csv"
    # AUDIO_DIR = "/home/student/Music/1/FYP/data/miniDataset/chunks"

    num_classes = len(class_mapping)
    song_predictions_csv = "/home/student/Music/1/FYP/MusicGenreClassifier/CNN/checkpoints/song_predictions.csv"
    df = pd.read_csv(song_predictions_csv)
    print(df.head())

    print(class_mapping)


    cm = confusion_matrix(df['expected'].map(lambda x: class_mapping[x]), df['prediction'].map(lambda x: class_mapping[x]), labels=class_mapping)
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm * 100, 2)
    df_cm = pd.DataFrame(cm, index=class_mapping, columns=class_mapping)

    plt.figure(figsize=(1080 / 96, 810 / 96), dpi=96)
    sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='.2f')

    plt.subplots_adjust(left=0.17, bottom=0.19, right=1, top=0.92)
    plt.xlabel("Predicted")
    plt.ylabel("Truth")

    title = 'song level prediction'
    plt.title(title, y=1.05)
    plt.savefig(title + "_confusion_matrix_voting.png")
    # plt.show()
    plt.close()

    classification_report = classification_report(df['expected'].map(lambda x: class_mapping[x]), df['prediction'].map(lambda x: class_mapping[x]), labels=class_mapping)
    with open(title + "_classification_report_voting.txt", "a") as f:
        f.write(classification_report)

    print(classification_report)

