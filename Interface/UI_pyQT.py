from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
import sys
from Predict.inference_with_aggregate_single_track import combined


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


        # self.predict_button = QPushButton('Predict')
        # self.predict_button.clicked.connect(self.predicting_label)
        # self.predict_button.clicked.connect(self.predict)

        layout = QVBoxLayout()
        layout.addWidget(self.file_label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.select_file_button)
        # layout.addWidget(self.predict_button)


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

        MODEL_DIR = "/home/cihe/Music/MusicGenreClassifier/CNN/Model_Weights_Logs/checkpoints_14_Epoch_no_val_improvement_in_10/"
        self.prediction = combined(self.file_path, MODEL_DIR)  # call your classifier function here
        self.prediction_label.setText(f'Prediction: {self.prediction}')

    # def predicting_label(self):
    #     # TODO: add "predicting"
    #     self.prediction_label.setText('Predicting...')
    #
    # def predict(self):
    #     MODEL_DIR = "/home/cihe/Music/MusicGenreClassifier/CNN/Model_Weights_Logs/checkpoints_14_Epoch_no_val_improvement_in_10/"
    #     if self.file_path:
    #         self.prediction = combined(self.file_path, MODEL_DIR)  # call your classifier function here
    #         self.prediction_label.setText(f'Prediction: {self.prediction}')
    #     else:
    #         self.prediction_label.setText('Please select a file first.')



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
