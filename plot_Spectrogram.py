import torch
import torchaudio
import matplotlib.pyplot as plt

waveform, sample_rate = torchaudio.load("path/to/your/audiofile.wav")

sample_rate = 44100
n_fft = 2048
n_mels = 128
# Construct the MelSpectrogram transform
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)

# Apply the MelSpectrogram transform to the waveform
melspectrogram = mel_transform(waveform)

# Plot the spectrogram
plt.imshow(melspectrogram.log2()[0, :, :].detach().numpy(), cmap="inferno", origin="lower", aspect="auto")
plt.colorbar()
plt.title("Mel Spectrogram")
plt.xlabel("Frame")
plt.ylabel("Mel Frequency Bin")
plt.tight_layout()
plt.show()