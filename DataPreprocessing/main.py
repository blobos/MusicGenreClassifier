from splitter import track2chunks
from chunks2spectogram import chunk2spectogram
import multiprocessing

TRAIN_INPUT_DIRECTORY = "/FYP/data/train/tracks/"
TRAIN_OUTPUT_DIRECTORY = "/FYP/data/train/chunks/"
TRAIN_CSV_DIRECTORY = "/FYP/data/train/train_annotations.csv"
PREDICT_INPUT_DIRECTORY = "/FYP/data/predict/tracks/"
PREDICT_OUTPUT_DIRECTORY = "/FYP/data/predict/chunks/"
PREDICT_CSV_DIRECTORY = "/FYP/data/predict/predict_annotations.csv"

if __name__ == "__main__":

    track2chunks(TRAIN_INPUT_DIRECTORY, TRAIN_OUTPUT_DIRECTORY, TRAIN_CSV_DIRECTORY)
    track2chunks(PREDICT_INPUT_DIRECTORY, PREDICT_OUTPUT_DIRECTORY, PREDICT_CSV_DIRECTORY)
    # chunk2spectogram()
