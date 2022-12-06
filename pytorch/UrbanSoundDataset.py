from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir):
        self.__annotations__ = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)  # return # of samples in dataset

    def __getitem__(self, index):  # load waveform at index and label
        # a_list[1] == a_list.__getitem__(1)
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal,label

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}" #folder of audiofile in USD
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == "__main__":
    ANNOTATIONS_FILE = "pytorch/data/UrbanSound8k/metadata/UrbanSound8k.csv"
    AUDIO_DIR = "pytorch/data/UrbanSound8k/audio"
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)

    print(f"There are {len(usd)} samples in the dtaset.")

    signal, label = usd[0]

    a = 1