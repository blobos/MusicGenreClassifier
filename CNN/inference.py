import torch
import torchaudio

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


def predict(model, input_tensor, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        # expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model

    cnn = CNNNetwork()
    cnn = cnn.to("cuda")
    state_dict = torch.load("test.pth")
    cnn.load_state_dict(state_dict)



    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/test_annotations.csv"
    AUDIO_DIR = "/home/student/Music/1/FYP/data/test/chunks"

    DMSP = DatasetMelSpecPrep(ANNOTATIONS_FILE,
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              "cuda",
                              test=True)

    # get a sample from the urban sound dataset for inference
    print("usd[0]", DMSP[0])
    print(len(DMSP))
    print(DMSP[0][0])
    input_tensor, target = DMSP[0][0], DMSP[0][1]  # [batch size, num_channels, fr, time]
    input_tensor.unsqueeze_(0)
    input_tensor = input_tensor.to("cuda")

    # make an inference
    predicted, expected = predict(cnn, input_tensor, target,
                                  class_mapping)
    # print(f"Predicted: '{predicted}', expected: '{expected}'")
    print(f"Predicted: '{predicted}'")
