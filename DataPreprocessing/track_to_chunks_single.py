import pandas as pd
from pydub import AudioSegment
from multiprocessing import Pool
import os
import traceback
import logging  # added import for logging
# from tqdm import tqdm  # added import for progress bar

#check if in csv rather than iterate through folder
logging.basicConfig(filename='track_to_chunks_error.log', level=logging.ERROR)

chunk_length = 30 * 1000  # 30 seconds pydub calculates in millisec
overlap = 0.5 * chunk_length  # 15 seconds
bitrate = "128k"  # 128kbps


def split_audio(file, output_directory, df, labelled):
    # print(file)
    #combine subgenre name + subgenre track counter into new column to check


    try:
        path_split = file.split("\\")
        file_name = path_split[0].split("/")[-1]
        if labelled:  # if test track, subgenre from the filename is not used.
            file_name = file_name.split("_")
            subgenre = file_name[1] + "_" + file_name[2]
            subgenre_track_counter = file_name[0]
        # print("subgenre: ", subgenre)

        audio_track = AudioSegment.from_file(file)

        start = 0
        max_track_length = 4 * 60 * 1000 #4 minutes
        track_length = len(audio_track)

        #if >4 minutes, trim to 4 minutes
        if track_length > max_track_length:
            track_length = len(audio_track[:max_track_length])
        else:
            #trim to nearest 30 seconds
            track_length = track_length - (track_length % (30 * 1000))


        chunk_counter = 1
        total_subgenre_track_chunks = track_length / (chunk_length - overlap) - 1
        # 30 sec chunks with 15 overlap, -1 because chunks are 30 sec not 15, so cannot fit chunk into last 15 sec

        # print(file_name, start, track_length)
        # print("-------------------")
        while chunk_counter < total_subgenre_track_chunks+1:
            end = start + chunk_length
            chunk = audio_track[start:end]

            sort_name = file_name.replace(" ", "_")
            chunk_directory = output_directory + "/" + sort_name + "/"


            # chunks/black_metal/black_metal_001/
            # print("directory: ", chunk_directory)
            output_filename = sort_name + "_chunk_" + str("%02d" % (chunk_counter,)) + 'of' + str(
                "%02d" % (total_subgenre_track_chunks,)) + '.wav'
            # print("filename: ", output_filename)
            # black_metal_0001_chunk_01of25.mp3



            if not os.path.exists(chunk_directory):
                print(f"creating directory: f{chunk_directory}")
                os.makedirs(chunk_directory)
            print("creating: ", output_filename)
            chunk.export(chunk_directory + output_filename, format="wav", bitrate=bitrate)
            # print("Processing chunk " + output_filename + ". Start = " + str(start) + " end = " + str(end))
            chunk_counter = chunk_counter + 1
            start += chunk_length - overlap
            # else:
                # print(f"file f{output_filename} in directory")

        return chunk_directory

    except Exception as e:
        logging.error(f"Error processing file {file}: {traceback.format_exc()}")
        print(f"Error processing file {file}: {e}")


def audio_split_pooling(input_directory, output_directory, pool_processes=128, labelled=False):
    # Run split_audio with pooling for files in input_directory


    audio_files = []
    # print(audio_files)

    # Get a list of all the subdirectories
    subdirectories = [x[0] for x in os.walk(input_directory)]

    # Get a list of all the audio files in the folder and subfolders
    # print(subdirectories)
    AUDIO_EXTENSIONS = ['.mp3', '.flac', '.wav', '.aac', '.m4a', '.ogg']

    for subdir in subdirectories:
        files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(tuple(AUDIO_EXTENSIONS))]
        audio_files.extend(files)
    # print(audio_files)
    # for file in audio_files:
    #     split_audio(file,output_directory)
    # with Pool(pool_processes) as pool:
    #     for _ in tqdm(pool.starmap(split_audio, [(file, output_directory, df, labelled) for file in audio_files]), total=len(audio_files)):
    #         pass
    with Pool(pool_processes) as pool:
        pool.starmap(split_audio, [(file, output_directory, df, labelled) for file in audio_files])


if __name__ == "__main__":

    TRAIN_INPUT_DIRECTORY = "/home/student/Music/1/FYP/data/train/tracks"
    TRAIN_OUTPUT_DIRECTORY = "/home/student/Music/1/FYP/data/train/chunks/"  # "data/train/chunks/"
    TRAIN_ANNOTATIONS = "/home/student/Music/1/FYP/data/train_annotations.csv"

    # print(df.columns)
    # print(df.head)
    audio_split_pooling(TRAIN_INPUT_DIRECTORY, TRAIN_OUTPUT_DIRECTORY, TRAIN_ANNOTATIONS, pool_processes=128, labelled=True)