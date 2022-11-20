import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def chunks2spectogram():
    destination_directory = "/media/aaron/My Passport/FYP/spectograms/"
    input_directory = '/media/aaron/My Passport/FYP/chunks/'

    counter = 1
    for filename in os.listdir(input_directory):
        f = os.path.join(input_directory, filename)
        if os.path.isfile(f):
            y, sr = librosa.load(f)

            n_mels = 128  # entire frequency spectrum split into 128
            n_fft = 2048  # The amount of samples we are shifting after each fft (window size
            hop_length = 512  # Short-time Fourier Transformation on our audio data

            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_DB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
            plt.axis('off')
            plt.savefig(destination_directory + filename + '.png', bbox_inches='tight', pad_inches=0)
            print(str(counter) + ", " + filename)
            counter = counter + 1
