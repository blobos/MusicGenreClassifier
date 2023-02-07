import torch
import torchaudio
import csv

from cnn import CNNNetwork
from datasetmelspecprep import DatasetMelSpecPrep
from train import SAMPLE_RATE, NUM_SAMPLES



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
    model.eval()
    with torch.no_grad():
        predictions = model(input[0].unsqueeze_(0))
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]

        print(predictions[0])


        path = input[1]
    return predictions[0]


if __name__ == "__main__":

    ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/mini_train_annotations.csv"
    AUDIO_DIR = "/home/student/Music/1/FYP/data/miniDataset/chunks/"

    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("/home/student/Music/1/FYP/MusicGenreClassifier/CNN/trained/66s epoch "
                            "resampler/model_55.pth")
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


    # get a sample from the urban sound dataset for inference
    input = dmsp[0] # [batch size, num_channels, fr, time]
    # print(input)
    # print("len: " + str(len(dmsp[0])))
    # print(dmsp[0])


    # make an inference
    predicted = predict(cnn, input)

    with open("/home/student/Music/1/FYP/data/mini_train_annotations.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        prediction = []
        i = 0
        for row in reader:
            input = dmsp[i]
            predicted = predict(cnn, input)
            prediction += predicted
            i += 1
        predicted_index = prediction.argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[input[1]]
