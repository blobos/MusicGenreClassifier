from pydub import AudioSegment
from multiprocessing import Pool
import os
from chunks_to_CSV import chunks2CSV
import csv

chunk_length = 30 * 1000  # 30 seconds pydub calculates in millisec
overlap = 15 * 1000  # 15 seconds
bitrate = "128k"  # 128kbps


def split_audio(file, output_directory):
    # print(file)
    try:
        path_split = file.split("\\")
        file_path = path_split[0].split("/")[-1]
        file_name = file_path.split("_")
        subgenre = file_name[1] + "_" + file_name[2]
        # print("subgenre: ", subgenre)

        audio_track = AudioSegment.from_file(file)
        subgenre_track_counter = file_name[0]

        start = 0
        max_track_length = 4 * 60 * 1000
        track_length = len(audio_track)

        if track_length > max_track_length:
            track_length = len(audio_track[:max_track_length])
        else:
            track_length = track_length - (track_length % (30 * 1000))

        chunk_counter = 1
        total_subgenre_track_chunks = len(audio_track) // (chunk_length - overlap)  # 30 sec chunks with 15 overlap
        # print(file_name, start, track_length)
        # print("-------------------")
        while start < track_length:
            end = start + chunk_length
            chunk = audio_track[start:end]

            sort_name = subgenre + "_" + subgenre_track_counter
            # black_metal_001
            # print("sortname: ",sort_name)
            output_filename = sort_name + "_chunk_" + str("%02d" % (chunk_counter,)) + 'of' + str(
                total_subgenre_track_chunks) + '.mp3'
            # print("filename: ", output_filename)
            # black_metal_0001_chunk_01_25.mp3
            chunk_directory = output_directory + subgenre + "/" + sort_name + "/"
            # chunks/black_metal/black_metal_001/
            # print("directory: ", chunk_directory)
            if not os.path.exists(chunk_directory):
                # print("no exist directory")
                os.makedirs(chunk_directory)
            if not os.path.exists(chunk_directory + output_filename):
                # print(file)
                chunk.export(chunk_directory + output_filename, format="mp3", bitrate=bitrate)
                # print("Processing chunk " + output_filename + ". Start = " + str(start) + " end = " + str(end))
                chunk_counter = chunk_counter + 1
                start += chunk_length - overlap
    except Exception as e:
        print("Error processing file: ", file)
        print(e)


if __name__ == "__main__":
    TRAIN_INPUT_DIRECTORY = "/home/student/Music/1/FYP/data/train/tracks/"  # "/FYP/data/train/tracks/"
    TRAIN_OUTPUT_DIRECTORY = "/home/student/Music/1/FYP/data/train/chunks/"  # "data/train/chunks/"
    # TRAIN_CSV_DIRECTORY = "data/train/train_annotations.csv"
    # PREDICT_INPUT_DIRECTORY = "/FYP/data/predict/tracks/"
    # PREDICT_OUTPUT_DIRECTORY = "/FYP/data/predict/chunks/"
    # PREDICT_CSV_DIRECTORY = "/FYP/data/predict/predict_annotations.csv"

    input_directory = TRAIN_INPUT_DIRECTORY
    output_directory = TRAIN_OUTPUT_DIRECTORY
    # csv_directory = TRAIN_CSV_DIRECTORY

    # split chunks
    audio_files = []
    # print(audio_files)
    # Get a list of all the subdirectories
    subdirectories = [x[0] for x in os.walk(input_directory)]
    # Get a list of all the audio files in the folder and subfolders
    # print(subdirectories)
    AUDIO_EXTENSIONS = ['.mp3', '.flac', '.wav', '.aac', '.m4a']

    for subdir in subdirectories:
        files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(tuple(AUDIO_EXTENSIONS))]
        audio_files.extend(files)
    # print(audio_files)
    # for file in audio_files:
    #     split_audio(file,output_directory)
    with Pool(processes=16) as pool:
        pool.starmap(split_audio, [(file, output_directory) for file in audio_files])

    chunks2CSV()
