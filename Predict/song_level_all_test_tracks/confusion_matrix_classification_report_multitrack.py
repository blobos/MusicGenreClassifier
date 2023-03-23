import torch
import torchaudio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from inference_voting_multitrack import inference_voting_multitrack

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
    checkpoint_dir = "../../CNN/Model_Weights_Logs/vgg19/"
    model_path = checkpoint_dir + "lowest_val_loss.pth"
    parameters = checkpoint_dir + "parameters.txt"

    audiofile_annotations = "/home/student/Music/1/FYP/data/test_annotations.csv"
    audiofile_dir = "/home/student/Music/1/FYP/data/test/chunks"

    song_predictions_csv = inference_voting_multitrack(model_path=model_path,
                                                       parameters=parameters,
                                                       prediction_output_csv_dir=checkpoint_dir,
                                                       audiofile_dir=audiofile_dir,
                                                       audiofile_annotations=audiofile_annotations)

    # ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/mini_train_annotations.csv"
    # AUDIO_DIR = "/home/student/Music/1/FYP/data/mini/chunks"

    num_classes = len(class_mapping)
    df = pd.read_csv(song_predictions_csv)
    print(df.head())

    print(class_mapping)

    cm = confusion_matrix(df['expected'].map(lambda x: class_mapping[x]),
                          df['prediction'].map(lambda x: class_mapping[x]), labels=class_mapping)
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
    plt.savefig(checkpoint_dir + title + "_confusion_matrix_voting.png")
    # plt.show()
    plt.close()

    classification_report = classification_report(df['expected'].map(lambda x: class_mapping[x]),
                                                  df['prediction'].map(lambda x: class_mapping[x]),
                                                  labels=class_mapping)
    with open(checkpoint_dir + title + "_classification_report_voting.txt", "a") as f:
        f.write(classification_report)

    print(classification_report)
