import os
import sys
import torch
import torchaudio.transforms
import pandas as pd
from scipy.stats import mode

from tqdm import tqdm
sys.path.append("")
from FYP.MusicGenreClassifier.CNN.Models.cnn_vgg19_leaky_relu_batchnorm_dropout import CNNNetwork
from FYP.MusicGenreClassifier.CRNN.CRNN_test.CRNN_biLSTM import NetworkModel
# from FYP.MusicGenreClassifier.CNN.train_with_validation_binary import SAMPLE_RATE, NUM_SAMPLES, N_FFT, HOP_LENGTH, N_MELS
# from FYP.MusicGenreClassifier.CNN.Models.cnn_vgg16 import CNNNetwork
from FYP.MusicGenreClassifier.Predict.song_level_all_test_tracks.datasetmelspecprep_for_multitrack import DatasetMelSpecPrep

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


def predict(model, input):
    # print("prediction for ")
    model.eval()
    with torch.no_grad():
        predictions = model(input[0].unsqueeze_(0)) #why unsqueeze
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        # print("predict() predictions:", predictions)
        # print(predictions[0])

        expected_label = input[1]
        path = input[2]
        # print(expected_label, path)
    return path.split("/")[-1], expected_label, predictions


def prediction2df(predict_list, dmsp, network_model):
    # for each prediction of input/dmsp
    # print("dmsp length:", len(dmsp))
    # for i in range(0, len(dmsp)):
    #     input = dmsp[i] #(tensor, expected, path)
    for input in tqdm(dmsp, desc="Predicting chunks", unit='chunks'):
        # print("input:", input, type(input))
        # print("waveform:", input[0].shape)
        path, expected, prediction = predict(network_model, input)
        # print(path, expected, prediction)
        #prediction = tensor of 10 tensors
        # print("prediction:", prediction, prediction.size(), type(prediction))
        # print("prediction argmax:", prediction.argmax(0).item())
        # row = pd.Series({'path': path, 'expected': expected, 'predict': prediction})
        # print("row", row)
        row = [path, expected, prediction.argmax(0).item()]
        predict_list.append(row)
    return predict_list



def prediction_vote(group):
    # Use the mode function to vote for the prediction
    voted_prediction = mode(group["prediction"])[0]
    return voted_prediction


# make sure predictions are probabilities instead of integer

def inference_voting_multitrack(model_path, parameters, prediction_output_csv_dir, audiofile_dir,
                                audiofile_annotations):
    if not os.path.exists(prediction_output_csv_dir + 'song_predictions.csv'):
        print('Creating:',  prediction_output_csv_dir + 'song_predictions.csv')
        # model_path = "../../CNN/checkpoints/lowest_val_loss.pth"
        # parameters = "../../CNN/checkpoints/parameters.txt"
        # prediction_output_csv_dir = "../../CNN/checkpoints/"

        # Get audio sample parameters from parameters.txt for mel spectrogram transformations
        with open(parameters, "r") as f:
            line = f.readlines()
            print(line[5])
            SAMPLE_RATE = int(line[6].split()[-1])
            NUM_SAMPLES = int(line[7].split()[-1])
            N_FFT = int(line[8].split()[-1])
            HOP_LENGTH = int(line[9].split()[-1])
            N_MELS = int(line[10].split()[-1])
            print(f"Sample Rate: {SAMPLE_RATE}\n"
                  f"Number of samples: {NUM_SAMPLES}"
                  f"N_FFT: {N_FFT}\n"
                  f"Hop length: {HOP_LENGTH}\n"
                  f"N_MELS: {N_MELS}")

            # mel spectrogram
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            )

            # audiofile_annotations = "/home/student/Music/1/FYP/data/test_annotations.csv"
            # audiofile_dir = "/home/student/Music/1/FYP/data/test/chunks"
            network_model = NetworkModel()
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            network_model.load_state_dict(state_dict)

            dmsp = DatasetMelSpecPrep(audiofile_annotations,
                                      audiofile_dir,
                                      mel_spectrogram,
                                      SAMPLE_RATE,
                                      NUM_SAMPLES,
                                      "cpu")

            chunk_prediction_list = []
            prediction2df(chunk_prediction_list, dmsp, network_model)

            chunk_predictions = pd.DataFrame(chunk_prediction_list, columns=['path', 'expected', 'prediction'])
            chunk_predictions_csv_path = prediction_output_csv_dir + 'chunk_predictions.csv'
            chunk_predictions.to_csv(chunk_predictions_csv_path)
            print("df.head", chunk_predictions.head())
            grouped = chunk_predictions.groupby(chunk_predictions["path"].str[:-17]).apply(prediction_vote)
            final_predictions_list = []

            for group_name, voted_prediction in grouped.items():
                # from first row in group get:
                # chunk_prediction = chunk_predictions.loc[chunk_predictions["path"].str[:-17] == group_name, "prediction"].iloc[0]
                group_expected = chunk_predictions.loc[chunk_predictions["path"].str[:-17] == group_name, "expected"].iloc[
                    0]
                final_predictions_list.append((group_name, group_expected, voted_prediction))

            pd.DataFrame(final_predictions_list, columns=['name', 'expected', 'prediction']).to_csv(
                prediction_output_csv_dir + 'song_predictions.csv')
            print("Song Prediction.csv:", prediction_output_csv_dir + 'song_predictions.csv')
    return prediction_output_csv_dir + 'song_predictions.csv'


if __name__ == "__main__":

    # Replace model_path and parameters.txt
    checkpoint_dir = "../../CRNN/CRNN_test/"
    model_path = checkpoint_dir + "lowest_val_loss.pth"
    parameters = checkpoint_dir + "parameters.txt"

    # Get audio sample parameters from parameters.txt for mel spectrogram transformations
    with open(parameters, "r") as f:
        line = f.readlines()
        print(line[5])
        SAMPLE_RATE = int(line[6].split()[-1])
        NUM_SAMPLES = int(line[7].split()[-1])
        N_FFT = int(line[8].split()[-1])
        HOP_LENGTH = int(line[9].split()[-1])
        N_MELS = int(line[10].split()[-1])
        print(f"Sample Rate: {SAMPLE_RATE}\n"
              f"Number of samples: {NUM_SAMPLES}\n"
              f"N_FFT: {N_FFT}\n"
              f"Hop length: {HOP_LENGTH}\n"
              f"N_MELS: {N_MELS}")

    # mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    audiofile_annotations = "/home/student/Music/1/FYP/data/test_annotations.csv"
    audiofile_dir = "/home/student/Music/1/FYP/data/test/chunks"
    network_model = NetworkModel()
    state_dict = torch.load(model_path)
    network_model.load_state_dict(state_dict)

    dmsp = DatasetMelSpecPrep(audiofile_annotations,
                              audiofile_dir,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              "cpu")

    chunk_prediction_list = []
    prediction2df(chunk_prediction_list, dmsp, network_model)

    chunk_predictions = pd.DataFrame(chunk_prediction_list, columns=['path', 'expected', 'prediction'])
    chunk_predictions_csv_path = checkpoint_dir + 'chunk_predictions.csv'
    chunk_predictions.to_csv(chunk_predictions_csv_path)
    print("df.head", chunk_predictions.head())
    grouped = chunk_predictions.groupby(chunk_predictions["path"].str[:-17]).apply(prediction_vote)
    final_predictions_list = []

    for group_name, voted_prediction in grouped.items():
        # from first row in group get:
        # chunk_prediction = chunk_predictions.loc[chunk_predictions["path"].str[:-17] == group_name, "prediction"].iloc[0]
        group_expected = chunk_predictions.loc[chunk_predictions["path"].str[:-17] == group_name, "expected"].iloc[0]
        final_predictions_list.append((group_name, group_expected, voted_prediction))

    pd.DataFrame(final_predictions_list, columns=['name', 'expected', 'prediction']).to_csv(
        checkpoint_dir + 'song_predictions.csv')
    # Print the new list
    print(final_predictions_list)

    # print(dmsp[0][1])
