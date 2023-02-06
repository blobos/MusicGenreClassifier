import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class DatasetMelSpecPrep(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 labelled):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.labelled = labelled

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        if self.labelled:
            return signal, label
        else:
            return signal, audio_sample_path

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            # print("cut")
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # print("padded")
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            # print("resampled")
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler = resampler.to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            # print("Mono-ed")
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        if self.labelled:
            subgenre = self.annotations.iloc[index, 2]
            filename = self.annotations.iloc[index, 0].split("_")
            filename = filename[0] + "_" + filename[1] + "_" + filename[2]
            fold = subgenre + "/" + filename  # folder of audiofile in SD
            path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        else:
            fold = self.annotations.iloc[index, 0][:-17]
            # print(self.audio_dir)
            # print(fold)
            # print(self.annotations.iloc[index, 0])
            path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        # print(path)
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 3]


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/home/student/Music/1/FYP/data/test_annotations.csv"
    AUDIO_DIR = "/home/student/Music/1/FYP/data/test/chunks"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = DatasetMelSpecPrep(ANNOTATIONS_FILE,
                             AUDIO_DIR,
                             mel_spectrogram,
                             SAMPLE_RATE,
                             NUM_SAMPLES,
                             device,
                             labelled=True)
    # broken when labelled = False


    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
    print(usd[0][1])
    print(usd)
