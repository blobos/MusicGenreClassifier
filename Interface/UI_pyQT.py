from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
import sys
from FYP.MusicGenreClassifier.Predict.inference_with_aggregate_single_track import combined


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Classifier Interface')

        self.file_path = ''
        self.prediction = ''

        self.file_label = QLabel('No file selected.')
        self.prediction_label = QLabel('')

        self.select_file_button = QPushButton('Select File')
        self.select_file_button.clicked.connect(self.select_file)

        self.predict_button = QPushButton('Predict')
        self.predict_button.clicked.connect(self.predicting_label)
        self.predict_button.clicked.connect(self.predict)

        layout = QVBoxLayout()
        layout.addWidget(self.file_label)
        layout.addWidget(self.select_file_button)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.prediction_label)

        self.setLayout(layout)

    def select_file(self):
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, 'Open file', '', 'CSV Files (*.csv);;All Files (*)')
        if self.file_path:
            self.file_label.setText(f'Selected file: {self.file_path}')

    def predicting_label(self):
        # TODO: add "predicting"
        self.prediction_label.setText('Predicting...')
    def predict(self):
        MODEL_DIR = "/home/student/Music/1/FYP/MusicGenreClassifier/CNN/Model_Weights_Logs/vgg19/"
        if self.file_path:
            self.prediction = combined(self.file_path, MODEL_DIR)  # call your classifier function here
            self.prediction_label.setText(f'Prediction: {self.prediction}')
        else:
            self.prediction_label.setText('Please select a file first.')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
