import os
import sys
import torch
import torchaudio.transforms

sys.path.append("../CNN")
from FYP.MusicGenreClassifier.CNN.cnn_vgg16 import CNNNetwork
from FYP.MusicGenreClassifier.CNN.train_with_validation_binary import SAMPLE_RATE, NUM_SAMPLES, N_FFT, HOP_LENGTH, N_MELS
from FYP.MusicGenreClassifier.CNN.datasetmelspecprep import DatasetMelSpecPrep
from FYP.MusicGenreClassifier.DataPreprocessing.chunks_to_CSV import chunks_to_CSV
from FYP.MusicGenreClassifier.DataPreprocessing.track_to_chunks import split_audio

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
def file_prep(file, output_directory='./predict_track/'):
    #load file
    print(file)
    print(output_directory)
    if not os.path.exists(output_directory):
        # print("no exist directory")
        os.makedirs(output_directory)
    # split_audio(file, output_directory, False)
    test_directory = split_audio(file, output_directory, False)
    print(test_directory)
    file_name = file.split("/")[-1]
    csv_path = output_directory + file_name + ".csv"
    print(csv_path)
    chunks_to_CSV(output_directory, csv_path, False)
    return test_directory, csv_path

def predict(model, input):
    model.eval()
    with torch.no_grad():
        predictions = model(input[0].unsqueeze_(0))
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]

        print(predictions[0])


        path = input[1]
    return predictions[0]


def predict_vote(csv_path):

    # open up csv
    # for row in csv get prediction and add to list

    #make sure predictions are probabilities instead of integer
    #

    return



if __name__ == "__main__":
    test_directory, csv_path = file_prep(
        "/FYP/data/test/testing/learned_subgenres/black_metal/003_black_metal_02. Cradle of Filth - The Forest "
        "Whispers My Name.mp3")
    model_path = "/home/student/Music/1/FYP/MusicGenreClassifier/CNN/trained/66s epoch resampler/model_55.pth"
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



    input = dmsp[0]

