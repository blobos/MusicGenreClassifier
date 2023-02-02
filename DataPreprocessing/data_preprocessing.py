from chunks_to_CSV import chunks_to_CSV
from track_to_chunks import audio_split_pooling


TRAIN_INPUT_DIRECTORY = "/home/student/Music/1/FYP/data/train/tracks/"  # "/FYP/data/train/tracks/"
TRAIN_OUTPUT_DIRECTORY = "/home/student/Music/1/FYP/data/train/chunks/"  # "data/train/chunks/"
TRAIN_CSV_DIRECTORY = "/home/student/Music/1/FYP/data/train_annotations.csv"

PREDICT_INPUT_DIRECTORY = "/home/student/Music/1/FYP/data/test/testing"
PREDICT_OUTPUT_DIRECTORY = "/home/student/Music/1/FYP/data/test/chunks"
PREDICT_CSV_DIRECTORY = "/home/student/Music/1/FYP/data/test_annotations.csv"

# split chunks
# audio_split_pooling(TRAIN_INPUT_DIRECTORY, TRAIN_OUTPUT_DIRECTORY, test=False)
# chunks_to_CSV(TRAIN_OUTPUT_DIRECTORY, TRAIN_CSV_DIRECTORY, test=False)

audio_split_pooling(PREDICT_INPUT_DIRECTORY, PREDICT_OUTPUT_DIRECTORY, test=True)
chunks_to_CSV(PREDICT_OUTPUT_DIRECTORY, PREDICT_CSV_DIRECTORY, test=True)
