from chunks_to_CSV import chunks_to_CSV
from track_to_chunks import audio_split_pooling


# TRAIN_INPUT_DIRECTORY = "/home/student/Music/1/FYP/data/train/tracks/"  # "/FYP/data/train/tracks/"
# TRAIN_OUTPUT_DIRECTORY = "/home/student/Music/1/FYP/data/train/chunks/"  # "data/train/chunks/"
# TRAIN_CSV_DIRECTORY = "/home/student/Music/1/FYP/data/train_annotations.csv"

PREDICT_INPUT_DIRECTORY = "/home/student/Music/1/FYP/data/test/testing/learned_subgenres"
PREDICT_OUTPUT_DIRECTORY = "/home/student/Music/1/FYP/data/test/chunks/"
PREDICT_CSV_DIRECTORY = "/home/student/Music/1/FYP/data/test_annotations.csv"

TRAIN_INPUT_DIRECTORY = "/home/student/Music/1/FYP/data/miniDataset/tracks/"  # "/FYP/data/train/tracks/"
TRAIN_OUTPUT_DIRECTORY = "/home/student/Music/1/FYP/data/miniDataset/chunks/"  # "data/train/chunks/"
TRAIN_CSV_DIRECTORY = "/home/student/Music/1/FYP/data/mini_train_annotations.csv"

# split chunks
# audio_split_pooling(TRAIN_INPUT_DIRECTORY, TRAIN_OUTPUT_DIRECTORY, labelled=False)
# chunks_to_CSV(TRAIN_OUTPUT_DIRECTORY, TRAIN_CSV_DIRECTORY, labelled=False)

audio_split_pooling(PREDICT_INPUT_DIRECTORY, PREDICT_OUTPUT_DIRECTORY, labelled=True)
chunks_to_CSV(PREDICT_OUTPUT_DIRECTORY, PREDICT_CSV_DIRECTORY, labelled=True)
