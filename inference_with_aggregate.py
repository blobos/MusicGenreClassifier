import os
import sys
import torch
import torchaudio.transforms
from collections import Counter
import csv

sys.path.append("")
from FYP.MusicGenreClassifier.CNN.cnn_vgg16 import CNNNetwork
# from FYP.MusicGenreClassifier.CNN.train_with_validation_binary import SAMPLE_RATE, NUM_SAMPLES, N_FFT, HOP_LENGTH, N_MELS
from CNN.datasetmelspecprep import DatasetMelSpecPrep
from DataPreprocessing.chunks_to_CSV import chunks_to_CSV
from DataPreprocessing.track_to_chunks import split_audio

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


def file_prep(file, output_directory='./predict_track'):
    # load file
    # print(file)
    # print(output_directory)
    if not os.path.exists(output_directory):
        print("no exist directory")
        os.makedirs(output_directory)
    file_name = file.split("/")[-1]
    test_directory = output_directory + f"/{file_name}/"
    csv_path = output_directory + file_name + ".csv"
    if not os.path.exists(test_directory):
        # split_audio(file, output_directory, False)
        test_directory = split_audio(file, output_directory, False)
        print(test_directory)
        file_name = file.split("/")[-1]
        csv_path = output_directory + file_name + ".csv"
        print(csv_path)
        chunks_to_CSV(test_directory, csv_path, False)
    return test_directory, csv_path


def predict(model, input):
    print("prediction for ")
    model.eval()
    with torch.no_grad():
        predictions = model(input[0].unsqueeze_(0))
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]

        print(predictions[0])

        path = input[1]
    return predictions[0]


def predict_vote():
    predictions = []
    i = 0
    for i in range(0, len(dmsp)):
        print(i)
        input = dmsp[i]
        predicted = predict(cnn, input)
        print("chunk prediction:", predicted)
        # get index of arg max
        # add to prediction list
        # get highest occurence
        # get classmapping
        predictions.append(predicted.argmax(0))
        i += 1
    final_prediction = class_mapping[max(predictions, key=predictions.count)]
    print("final class prediction:", final_prediction)


# make sure predictions are probabilities instead of integer


if __name__ == "__main__":
    test_directory, csv_path = file_prep(
        "/home/student/Music/1/FYP/MusicGenreClassifier/aggregate_prediction/predict_track/005_punk_rock_02 - Forward To Death.mp3")
    model_path = "/home/student/Music/1/FYP/MusicGenreClassifier/CNN/trained/vgg16/lowest_val_loss.pth"
    parameters = "/home/student/Music/1/FYP/MusicGenreClassifier/CNN/trained/vgg16/parameters.txt"

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

    ANNOTATIONS_FILE = csv_path
    AUDIO_DIR = test_directory
    cnn = CNNNetwork()
    state_dict = torch.load(model_path)
    cnn.load_state_dict(state_dict)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    dmsp = DatasetMelSpecPrep(ANNOTATIONS_FILE,
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              "cpu",
                              labelled=False)

    predict_vote()

    # print(dmsp[0][1])
