import os
import sys
import torch
import torchaudio.transforms
import pandas as pd
from scipy.stats import mode
from collections import Counter
import csv

sys.path.append("")
from FYP.MusicGenreClassifier.CNN.cnn_vgg16 import CNNNetwork
# from FYP.MusicGenreClassifier.CNN.train_with_validation_binary import SAMPLE_RATE, NUM_SAMPLES, N_FFT, HOP_LENGTH, N_MELS
from datasetmelspecprep_for_multitrack import DatasetMelSpecPrep


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


def predict(model, input):
    # print("prediction for ")
    model.eval()
    with torch.no_grad():
        predictions = model(input[0].unsqueeze_(0))
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]

        # print(predictions[0])

        expected_label = input[1]
        path = input[2]
        # print(expected_label, path)
    return path.split("/")[-1], expected_label, predictions[0]

def prediction2df(predict_list):
    #for each prediction of input/dmsp
    for i in range(0, len(dmsp)):
        input = dmsp[i]
        path, expected, prediction = predict(cnn, input)
        # row = pd.Series({'path': path, 'expected': expected, 'predict': prediction})
        # print("row", row)
        row = [path, expected, prediction.argmax(0).item()]
        predict_list.append(row)
    return predict_list

def prediction_vote(group):
    # Use the mode function to vote for the prediction
    voted_prediction = mode(group["prediction"])[0][0]
    return voted_prediction

# make sure predictions are probabilities instead of integer


if __name__ == "__main__":


    model_path = "/FYP/MusicGenreClassifier/CNN/trained/vgg16/lowest_val_loss.pth"
    parameters = "/home/student/Music/1/FYP/MusicGenreClassifier/CNN/trained/vgg16/parameters.txt"

    #Get audio sample parameters from parameters.txt for mel spectrogram transformations
    with open(parameters, "r") as f:
        line = f.readlines()
        SAMPLE_RATE = int(line[6].split()[-1])
        NUM_SAMPLES = SAMPLE_RATE
        N_FFT = int(line[7].split()[-1])
        HOP_LENGTH = int(line[8].split()[-1])
        N_MELS = int(line[9].split()[-1])
        print(f"Sample Rate: {SAMPLE_RATE}\n"
              f"N_FFT: {N_FFT}\n"
              f"Hop length: {HOP_LENGTH}\n"
              f"N_MELS: {N_MELS}")


    #mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    ANNOTATIONS_FILE = "/FYP/data/test_annotations.csv"
    AUDIO_DIR = "/FYP/data/test/chunks"
    cnn = CNNNetwork()
    state_dict = torch.load(model_path)
    cnn.load_state_dict(state_dict)

    dmsp = DatasetMelSpecPrep(ANNOTATIONS_FILE,
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              "cpu")

    chunk_prediction_list = []
    prediction2df(chunk_prediction_list)

    chunk_predictions = pd.DataFrame(chunk_prediction_list, columns=['path', 'expected', 'prediction'])
    predictions_csv_path = '../../CNN/trained/vgg16/chunk_predictions.csv'
    chunk_predictions.to_csv(predictions_csv_path)
    print("df.head", chunk_predictions.head())
    grouped = chunk_predictions.groupby(chunk_predictions["path"].str[:-17]).apply(prediction_vote)
    final_predictions_list = []
    for group_name, voted_prediction in grouped.items():
        #from first row in group get:
        # chunk_prediction = chunk_predictions.loc[chunk_predictions["path"].str[:-17] == group_name, "prediction"].iloc[0]
        group_expected = chunk_predictions.loc[chunk_predictions["path"].str[:-17] == group_name, "expected"].iloc[0]
        final_predictions_list.append((group_name, group_expected, voted_prediction))

    pd.DataFrame(final_predictions_list, columns=['name', 'expected', 'prediction']).to_csv(
        '../../CNN/trained/vgg16/song_predictions.csv')
    # Print the new list
    print(final_predictions_list)



    # print(dmsp[0][1])
