from PyQt5.QtCore import QSize, QThread, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QProgressBar
import sys
from FYP.MusicGenreClassifier.Predict.inference_with_aggregate_single_track import combined
import librosa
import music_tag

import numpy as np
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
        self.processing = False

        # self.figure = plt.figure(figsize=(12, 8))
        # self.canvas = FigureCanvas(self.figure)

        self.file_label = QLabel('Select audio file for classification')
        self.prediction_label = QLabel('')
        self.status_label = QLabel('')
        self.progress_bar = QProgressBar()

        self.select_file_button = QPushButton('Select File')
        self.select_file_button.clicked.connect(self.select_file)
        self.select_file_button.setFixedHeight(100)

        layout = QVBoxLayout()
        # layout.addWidget(self.canvas)
        layout.addWidget(self.file_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.select_file_button)

        self.setLayout(layout)

    def select_file(self):
        if self.processing:
            return

        file_dialog = QFileDialog()
        audio_path = '/home/student/Music/1/FYP/data/one_track/'
        self.file_path, _ = file_dialog.getOpenFileName(self, 'Open Audio', audio_path, 'Audio Files (*.mp3 *.wav *.flac *.ogg *.opus *.wma *.aac *.m4a)'
                                                                                ';;All Files (*)')
        #TODO: 30 second clip of song or full song?
        if self.file_path:
            file_name = self.file_path.split("/")[-1]
            self.file_label.setText(f'Predicting: {file_name}')
            self.prediction_label.setText('')
            self.status_label.setText('Processing...')
            self.progress_bar.setValue(0)


            MODEL_DIR = "/home/student/Music/1/FYP/MusicGenreClassifier/CRNN/checkpoints/"
            self.processing = True

            # Start a new thread to run the combined method
            thread = QThread()
            worker = Worker(self.file_path, MODEL_DIR)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.progress.connect(self.update_progress)
            worker.result.connect(self.display_result)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.start()
        else:
            self.file_label.setText('Select audio file for classification')

            # self.prediction = combined(self.file_path, MODEL_DIR)
            # self.prediction_label.setText(f'Prediction: {self.prediction}')



        # y, sr = librosa.load(self.file_path)
        # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        # log_S = librosa.power_to_db(S, ref=np.max)

        # plot a wavform
        # plt.subplot(2, 1, 1)
        # librosa.display.waveplot(y, sr)
        # plt.title('Waveform')
        # plot mel spectrogram
        # plt.subplot(2, 1, 2)
        # librosa.display.specshow(
        #     log_S, sr=sr, x_axis='time', y_axis='mel')
        # plt.title('Mel spectrogram')
        # plt.tight_layout()  # 保证图不重叠
        # self.canvas.draw()

class Worker(QObject):
    class Worker(QThread):
        progress = pyqtSignal(int)
        finished = pyqtSignal(str)

        def __init__(self, file_path, model_dir):
            super().__init__()
            self.file_path = file_path
            self.model_dir = model_dir

        def run(self):
            prediction = combined(self.file_path, self.model_dir)
            self.finished.emit(prediction)

        # def predict_vote(self, dmsp):
        #     predictions = []
        #     i = 0
        #     for i in range(0, len(dmsp)):
        #         input = dmsp[i]
        #         networkModel = NetworkModel()
        #         predicted = predict(networkModel, input)
        #         progress = int((i + 1) / len(dmsp) * 100)
        #         self.progress.emit(progress)
        #         predictions.append(predicted.argmax(0))
        #         i += 1
        #     final_prediction = class_mapping[max(predictions, key=predictions.count)]
        #     return (prediction)

    # def load_default_spectrogram(self):
    #     # Load default audio file
    #     default_file = 'path/to/default/file.wav'
    #     y, sr = librosa.load(default_file)
    #
    #     # Compute mel spectrogram
    #     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    #     log_S = librosa.power_to_db(S, ref=np.max)
    #
    #     # Display spectrogram in figure canvas
    #     self.display_spectrogram(log_S)

    # def load_spectrogram(self, file_path):
    #     # Load selected audio file
    #     y, sr = librosa.load(file_path)
    #
    #     # Compute mel spectrogram
    #     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    #     log_S = librosa.power_to_db(S, ref=np.max)
    #
    #     # Display spectrogram in figure canvas
    #     self.display_spectrogram(log_S)

# TODO:
#display mel spectrogram and waveform after loading audio file
    # https://ask.csdn.net/questions/7543071
    # def display_mel_spectrogram(self):
    #     y, sr = librosa.load(self.file_path)
    #     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    #     log_S = librosa.power_to_db(S, ref=np.max)
    #
    #     # plot a wavform
    #     plt.subplot(2, 1, 1)
    #     librosa.display.waveplot(y, sr)
    #     plt.title('Waveform')
    #     # plot mel spectrogram
    #     plt.subplot(2, 1, 2)
    #     librosa.display.specshow(
    #         log_S, sr=sr, x_axis='time', y_axis='mel')
    #     plt.title('Mel spectrogram')
    #     plt.tight_layout()  # 保证图不重叠
    #     self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
