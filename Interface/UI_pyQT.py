from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
import sys
from Predict.inference_with_aggregate_single_track import combined
import librosa

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Music Genre Classifier')

        self.setMinimumSize(QSize(600, 600))

        self.file_path = ''
        self.prediction = ''

        self.file_label = QLabel('Select audio file for classification')
        self.prediction_label = QLabel('')

        self.select_file_button = QPushButton('Select File')
        self.select_file_button.clicked.connect(self.select_file)
        self.select_file_button.setFixedHeight(100)

        layout = QVBoxLayout()
        layout.addWidget(self.file_label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.select_file_button)

        self.setLayout(layout)

    def select_file(self):
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, 'Open file', '', 'MP3 Files (*.mp3);; '
                                                                               'WAV Files (*.wav);;'
                                                                               'FLAC files (*.flac);; '
                                                                               'OGG files (*.ogg);; '
                                                                               'OPUS files (*.opus'
                                                                               ';;All Files (*)')
        if self.file_path:
            file_name = self.file_path.split("/")[-1]
            self.file_label.setText(f'Predicting: {file_name}')

        MODEL_DIR = "/home/cihe/Music/MusicGenreClassifier/CNN/Model_Weights_Logs" \
                    "/checkpoints_14_Epoch_no_val_improvement_in_10/"
        self.prediction = combined(self.file_path, MODEL_DIR)  # call your classifier function here
        self.prediction_label.setText(f'Prediction: {self.prediction}')

    def load_default_spectrogram(self):
        # Load default audio file
        default_file = 'path/to/default/file.wav'
        y, sr = librosa.load(default_file)

        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_S = librosa.power_to_db(S, ref=np.max)

        # Display spectrogram in figure canvas
        self.display_spectrogram(log_S)

    def load_spectrogram(self, file_path):
        # Load selected audio file
        y, sr = librosa.load(file_path)

        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_S = librosa.power_to_db(S, ref=np.max)

        # Display spectrogram in figure canvas
        self.display_spectrogram(log_S)

#TODO:
#display mel spectrogram and waveform after loading audio file
    # https://ask.csdn.net/questions/7543071
    def display_mel_spectrogram(self):
        y, sr = librosa.load(self.file_path, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        self.figure.clear()
        self.spec_image = plt.imshow(log_S, origin='lower', aspect='auto', cmap='jet')
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
