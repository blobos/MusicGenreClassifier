import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os


class UrbanSoundDataset(Dataset):



    # constructor
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)  # return # of samples in dataset

    def __getitem__(self, index):  # load waveform at index and label
        # a_list[1] == a_list.__getitem__(1)
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        # signal (a tensor) -> (num_channels, samples) -> (2, 160000)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr) #uniform sample rate
        signal = self._mix_down_if_necessary(signal) #mix to Mono
        signal = self._cut_if_necessary(signal) #too many samples, truncate
        signal = self._right_pad_if_necessary(signal) #if samples less pad right with 0's
        signal = self.transformation(signal) #transform into melSpec
        return signal, label

    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (# of channels (1), num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples] #all channel, up to num_samples
        return signal

    def _right_pad_if_necessary(self, signal):
        # 0 padding to the right
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples) #(left_pad, right_pad)
            #functional.pad padding:
            #(left pad for last dim in tensor, right pad ..., left pad for 2nd last dim, right ...)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"  # folder of audiofile in USD
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6] #6 = Class ID


if __name__ == "__main__":
    ANNOTATIONS_FILE = r"Y:\MusicGenreClassifier\pytorch\data\UrbanSound8K\metadata\UrbanSound8K.csv"
    AUDIO_DIR = r"Y:\MusicGenreClassifier\pytorch\data\UrbanSound8K\audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, #samples per sec
        n_fft=1024,
        hop_length=512,
        n_mels=64
    ) #callable object (self.transformation)

    # constructor
    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    print(f"There are {len(usd)} samples in the dataset.")

    signal, label = usd[0]

    a = 1
