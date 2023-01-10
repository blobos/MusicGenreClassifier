from pydub import AudioSegment
from multiprocessing import Pool
import os
import csv

chunk_length = 30 * 1000  # 30 seconds pydub calculates in millisec
overlap = 15 * 1000  # 15 seconds
bitrate = "128k"  # 128kbps


def split_audio(file, output_directory):
    # print(file)
    path_split = file.split("\\")
    subgenre = path_split[0].split("/")[-1]
    print("subgenre: ", subgenre)


    audio_track = AudioSegment.from_file(file)
    subgenre_track_counter = path_split[1][:3]
    start = 0
    max_track_length = 4 * 60 * 1000
    track_length = len(audio_track)

    if track_length > max_track_length:
        audio_track = audio_track[:max_track_length]

    chunk_counter = 0
    while start < track_length:
        end = start + chunk_length
        chunk = audio_track[start:end]

        sort_name = subgenre + "_" + subgenre_track_counter
        # black_metal_001
        print("sortname: ",sort_name)
        output_filename = sort_name + "_chunk_" + "%02d" % (chunk_counter,) + '.mp3'
        print("filename: ", output_filename)
        # black_metal_0001_chunk_01.mp3
        chunk_directory = output_directory + subgenre + "/" + sort_name + "/"
        # chunks/black_metal/black_metal_001/
        print("directory: ",chunk_directory)
        if not os.path.exists(chunk_directory):
            os.makedirs(chunk_directory)

        chunk.export(chunk_directory + output_filename, format="mp3", bitrate=bitrate)
        # print("Processing chunk " + str(chunk_counter) + ". Start = " + str(start) + " end = " + str(end))
        chunk_counter = chunk_counter + 1
        start += chunk_length - overlap


def add_to_CSV(input_directory, csv_directory):
    # CSV
    header = ["Subgenre", "Subgenre Index", "Subgenre Track Number", "Total chunks for track", "Filename"]
    with open(csv_directory, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for subgenre in os.listdir(input_directory):
            # f = os.path.join(input_directory, filename)
            # if os.path.isfile(f):
            # print(f)
            print("subgenre: ", subgenre)
            subgenre_directory = os.path.join(input_directory, subgenre)
            subgenre_map = {0: "alternative_rock", 1: "black_metal", 2: "death_metal", 3: "dreampop", 4: "heavy_metal",
                            5: "house", 6: "indie_rock", 7: "post_rock", 8: "progressive_rock", 9: "punk_rock",
                            10: "synthwave", 11: "techno", 12: "thrash_metal", 13: "trance"}

            for file in os.listdir(subgenre_directory):
                # print("track: ", file)
                track_path = os.path.join(subgenre_directory, file)
                # load audio track to get total chunks
                audio_track = AudioSegment.from_file(track_path)
                track_length = len(audio_track)
                total_subgenre_track_chunks = track_length // (chunk_length - overlap)  # 30 sec chunks with 15 overlap
                # get subgenre id and subgenre track counter from filename
                subgenre_id = get_key(subgenre_map, subgenre)
                # print(subgenre_id, subgenre)
                subgenre_track_counter = file[:3]
                # write to csv
                entry = [subgenre, subgenre_id, subgenre_track_counter, total_subgenre_track_chunks, file]
                # print(entry)
                writer.writerow(entry)

def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None

if __name__ == "__main__":
    TRAIN_INPUT_DIRECTORY = "/FYP/data/train/tracks/"
    TRAIN_OUTPUT_DIRECTORY = "/FYP/data/train/chunks/"
    TRAIN_CSV_DIRECTORY = "/FYP/data/train/train_annotations.csv"
    PREDICT_INPUT_DIRECTORY = "/FYP/data/predict/tracks/"
    PREDICT_OUTPUT_DIRECTORY = "/FYP/data/predict/chunks/"
    PREDICT_CSV_DIRECTORY = "/FYP/data/predict/predict_annotations.csv"

    input_directory = TRAIN_INPUT_DIRECTORY
    output_directory = TRAIN_OUTPUT_DIRECTORY
    csv_directory = TRAIN_CSV_DIRECTORY

    # make csv file
    add_to_CSV(input_directory, csv_directory)

    # split chunks
    audio_files = []
    # Get a list of all the subdirectories
    subdirectories = [x[0] for x in os.walk(input_directory)]
    # Get a list of all the audio files in the folder and subfolders
    for subdir in subdirectories:
        files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(".mp3")]
        audio_files.extend(files)

    # files = [f for f in os.listdir(input_directory)]
    with Pool(processes=4) as pool:
        pool.starmap(split_audio, [(file, output_directory) for file in audio_files])
