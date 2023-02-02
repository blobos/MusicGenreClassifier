import torch
import torchaudio

from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES


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


ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/test_annotations.csv"
AUDIO_DIR = "/home/student/Music/1/FYP/data/test/chunks"

def predict(model, input, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input[0].unsqueeze_(0))
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        print(predictions[0])
        predicted = class_mapping[predicted_index]
        path = input[1]
    return predicted, path


if __name__ == "__main__":

    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("/home/student/Music/1/FYP/MusicGenreClassifier/CNN/test.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu",
                            True)


    # get a sample from the urban sound dataset for inference
    input = usd[5] # [batch size, num_channels, fr, time]
    print(input)
    print("len: " + str(len(usd[0])))
    print(usd[0])


    # make an inference
    predicted,  path = predict(cnn, input, class_mapping)
    print(f"Predicted: '{predicted}', Path '{path}'")