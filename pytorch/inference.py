import torch
import torchaudio
from CNN import CNNNetwork
from UrbanSoundDataset import UrbanSoundDataset
from train import AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES

# genre_mapping = [
#     "rock",
#     "classical",
#     "jazz"
# ]

##subgenre_mapping = ["black",
# infer on multiple segments
ANNOTATIONS_FILE = "/FYP/data/predict/predict_annotations.csv"

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

subgenre_map = {0: "alternative_rock", 1: "black_metal", 2: "death_metal", 3: "dreampop_rock", 4: "heavy_metal",
                5: "house_electronic", 6: "indie_rock", 7: "post_rock", 8: "progressive_rock", 9: "punk_rock",
                10: "synthwave_electronic", 11: "techno_electronic", 12: "thrash_metal", 13: "trance_electronic"}


def predict(model, input, class_mapping):
    model.eval()  # turns off batch norm, dropout vs model.train()
    with torch.no_grad():  # context manager: model doesn't calculate gradient
        predictions = model(input)
        # Tensor (#number of samples passed, # of classes)
        # Tensor (1,10)
        predicted_index = predictions[0].argmax(0)  # argmax argument??? 0 axis???
        # map index to class
        predicted = class_mapping[predicted_index]
        # expected = class_mapping[target]

        return predicted


if __name__ == "__main__":
    # load model back
    cnn = CNNNetwork()
    state_dict = torch.load("latest.pth", map_location=torch.device('cpu'))
    cnn.load_state_dict(state_dict)

    # load urban sound dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,  # samples per sec
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )  # callable object (self.transformation)

    # constructor
    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu")  # no need to run on GPU

    # get sample from urban sound dataset for inference

    #load track
    #For loop trackID or track chunk count
    for track in usd[]
    input = usd[1][0]  # tensor is 3 dim, but we need 4 since batch_size --> [batch_size , num_channels, fr, time]
    print(usd[1])
    print(usd[1][0])
    print(input.shape)
    input.unsqueeze_(0)  # underscore helps put in extra dim(batch_size)
    print(input.shape)
    print(usd[1][1])

    # make inference
    predicted = predict(cnn, input, class_mapping)  # map integers to class(genre)
    print(f"Predicted: '{predicted}'")
