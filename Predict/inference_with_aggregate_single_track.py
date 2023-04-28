import os
import sys
import torch
import torchaudio.transforms
from tqdm import tqdm
from collections import Counter

sys.path.append("")
# from FYP.MusicGenreClassifier.CNN.Models.cnn_vgg19_leaky_relu_batchnorm_dropout import CNNNetwork
from FYP.MusicGenreClassifier.CRNN.CRNN.CRNN_biLSTM import NetworkModel
from FYP.MusicGenreClassifier.DataPreprocessing.datasetmelspecprep import DatasetMelSpecPrep
from FYP.MusicGenreClassifier.DataPreprocessing.chunks_to_CSV import chunks_to_CSV
from FYP.MusicGenreClassifier.DataPreprocessing.track_to_chunks_single import split_audio

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


def file_prep(file, output_directory='./single_track'):
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
        test_directory = split_audio(file, output_directory, df='', labelled=False)
        print(test_directory)
        file_name = file.split("/")[-1]
        csv_path = output_directory + file_name + ".csv"
        print(csv_path)
        chunks_to_CSV(test_directory, csv_path, False)
    return test_directory, csv_path


def predict(model, input):
    # print("prediction for ")
    # print("input", input)
    model.eval()
    with torch.no_grad():
        predictions = model(input[0].unsqueeze_(0))
        # print("prediction:", predictions)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
    return predictions


def predict_vote(dmsp):
    predictions = []

    progress_bar = tqdm(dmsp, desc="Predicting", unit='chunks')
    for input in progress_bar:
        # print("chunk #:", i)
        # input = dmsp[i]
        networkModel = NetworkModel()
        predicted = predict(networkModel, input)
        # print(predicted)
        # get index of arg max
        # add to prediction list
        # get highest occurence
        # get classmapping
        predictions.append(class_mapping[predicted.argmax(0)])
        # i += 1

    top_predictions = Counter(predictions).most_common(3)
    for i in range(len(top_predictions)):
        top_predictions[i] = top_predictions[i][0].split("_")[0]
        top_predictions[i] = top_predictions[i][0].capitalize() + top_predictions[i][1:]

    print("final class prediction:", top_predictions[0], top_predictions[1], top_predictions[2])
    return(top_predictions[0], top_predictions[1], top_predictions[2])



# make sure predictions are probabilities instead of integer

def combined(file_path, model_dir):
    test_directory, csv_path = file_prep(file_path)
    model_path = model_dir + "highest_val_acc.pth"
    parameters = model_dir + "parameters.txt"

    with open(parameters, "r") as f:
        line = f.readlines()
        SAMPLE_RATE = int(line[6].split()[-1])
        NUM_SAMPLES = int(line[7].split()[-1])
        N_FFT = int(line[8].split()[-1])
        HOP_LENGTH = int(line[9].split()[-1])
        N_MELS = int(line[10].split()[-1])
        # print(f"Sample Rate: {SAMPLE_RATE}\n"
        #       f"N_FFT: {N_FFT}\n"
        #       f"Hop length: {HOP_LENGTH}\n"
        #       f"N_MELS: {N_MELS}")

    ANNOTATIONS_FILE = csv_path
    AUDIO_DIR = test_directory
    networkModel = NetworkModel()
    state_dict = torch.load(model_path)
    networkModel.load_state_dict(state_dict)

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
    prediction = predict_vote(dmsp)
    return prediction

if __name__ == "__main__":
    file_path = "/home/student/Music/1/FYP/data/test/testing/learned_subgenres/synthwave_electronic/018_synthwave_electronic_018_house_electronic_09 - Last Dance XX.mp3"
    model_dir = "/home/student/Music/1/FYP/MusicGenreClassifier/CRNN/CRNN_test/"
    combined(file_path, model_dir)

    # test_directory, csv_path = file_prep(
    #     "/FYP/MusicGenreClassifier/aggregate_prediction/single_track/005_punk_rock_02 - Forward To Death.mp3")
    # model_path = "/FYP/MusicGenreClassifier/CNN/trained/vgg16/lowest_val_loss.pth"
    # parameters = "/home/student/Music/1/FYP/MusicGenreClassifier/CNN/trained/vgg16/parameters.txt"

    #
    # with open(parameters, "r") as f:
    #     line = f.readlines()
    #     SAMPLE_RATE = int(line[6].split()[-1])
    #     NUM_SAMPLES = SAMPLE_RATE
    #     N_FFT = int(line[7].split()[-1])
    #     HOP_LENGTH = int(line[8].split()[-1])
    #     N_MELS = int(line[9].split()[-1])
    #     print(f"Sample Rate: {SAMPLE_RATE}\n"
    #           f"N_FFT: {N_FFT}\n"
    #           f"Hop length: {HOP_LENGTH}\n"
    #           f"N_MELS: {N_MELS}")
    #
    # ANNOTATIONS_FILE = csv_path
    # AUDIO_DIR = test_directory
    # cnn = CNNNetwork()
    # state_dict = torch.load(model_path)
    # cnn.load_state_dict(state_dict)
    #
    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=SAMPLE_RATE,
    #     n_fft=N_FFT,
    #     hop_length=HOP_LENGTH,
    #     n_mels=N_MELS
    # )
    #
    # dmsp = DatasetMelSpecPrep(ANNOTATIONS_FILE,
    #                           AUDIO_DIR,
    #                           mel_spectrogram,
    #                           SAMPLE_RATE,
    #                           NUM_SAMPLES,
    #                           "cpu",
    #                           labelled=False)
    #
    # predict_vote()
    #
    # # print(dmsp[0][1])
