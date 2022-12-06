import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

#torchaudio MelSpectogram instead of librosa?

def chunk2spectogram():
    output_directory = "/media/aaron/My Passport/FYP/spectograms/"
    chunk_directory = '/media/aaron/My Passport/FYP/chunks/'

    counter = 1
    for genre in os.listdir(chunk_directory):
        chunk_genre_directory = os.path.join(chunk_directory, genre)
        for subgenre in os.listdir(chunk_genre_directory):
            chunk_subgenre_directory = os.path.join(chunk_genre_directory, subgenre)
            for track in os.listdir(chunk_subgenre_directory):
                chunk_track_directory = os.path.join(chunk_subgenre_directory, track)
                for track_chunk in os.listdir(chunk_track_directory):
                    track_chunk_path = os.path.join(chunk_track_directory, track_chunk)
                    if os.path.isfile(track_chunk_path):
                        y, sr = librosa.load(track_chunk_path)

                        n_mels = 128  # entire frequency spectrum split into 128
                        n_fft = 2048  # The amount of samples we are shifting after each fft (window size
                        hop_length = 512  # Short-time Fourier Transformation on our audio data

                        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                           n_mels=n_mels)
                        ## log spectogram to percieved by humans
                        S_DB = librosa.power_to_db(S, ref=np.max)

                        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
                        plt.axis('off')

                        spec_directory = output_directory + genre + "/" + subgenre + "/" + track + "/"
                        if not os.path.exists(spec_directory):
                            os.makedirs(spec_directory)
                        plt.savefig(spec_directory + track_chunk + '.png', bbox_inches='tight', pad_inches=0)
                        print(str(counter) + ", " + track_chunk)
                        counter = counter + 1
