import librosa
import urllib3

y, sr = librosa.load(('/media/aaron/My Passport/FYP/a1.ogg'))
# melspec = librosa.feature.melspectrogram(y=y, sr=sr)

# Displays are built with matplotlib
import matplotlib.pyplot as plt

# Let's make plots pretty
# import matplotlib.style as ms
# ms.use('seaborn-muted')

# Render figures interactively in the notebook
# %matplotlib nbagg

# IPython gives us an audio widget for playback
from IPython.display import Audio
