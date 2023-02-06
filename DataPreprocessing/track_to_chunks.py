from pydub import AudioSegment
from multiprocessing import Pool
import os
from chunks_to_CSV import chunks_to_CSV
import csv

chunk_length = 30 * 1000  # 30 seconds pydub calculates in millisec
overlap = 15 * 1000  # 15 seconds
bitrate = "128k"  # 128kbps


def _split_audio(file, output_directory, labelled):
    # print(file)
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
        max_track_length = 4 * 60 * 1000
        track_length = len(audio_track)

        if track_length > max_track_length:
            track_length = len(audio_track[:max_track_length])
        else:
            track_length = track_length - (track_length % (30 * 1000))

        chunk_counter = 1
        total_subgenre_track_chunks = track_length // (chunk_length - overlap)  # 30 sec chunks with 15 overlap
        # print(file_name, start, track_length)
        # print("-------------------")
        while start < track_length:
            end = start + chunk_length
            chunk = audio_track[start:end]

            if labelled:
                sort_name = subgenre + "_" + subgenre_track_counter
                # chunks/black_metal/black_metal_001/
                # print("directory: ", chunk_directory)
                # chunks/black_metal/black_metal_001/
                # print("directory: ", chunk_directory)
                chunk_directory = output_directory + subgenre + "/" + sort_name + "/"

            else:
                sort_name = file_name.replace(" ", "_")
                chunk_directory = output_directory + "/" + sort_name + "/"

            # chunks/black_metal/black_metal_001/
            # print("directory: ", chunk_directory)
            output_filename = sort_name + "_chunk_" + str("%02d" % (chunk_counter,)) + 'of' + str(
                "%02d" % (total_subgenre_track_chunks,)) + '.wav'
            # print("filename: ", output_filename)
            # black_metal_0001_chunk_01_25.mp3




            if not os.path.exists(chunk_directory):
                # print("no exist directory")
                os.makedirs(chunk_directory)
            if not os.path.exists(chunk_directory + output_filename):
                # print(file)
                chunk.export(chunk_directory + output_filename, format="wav", bitrate=bitrate)
                # print("Processing chunk " + output_filename + ". Start = " + str(start) + " end = " + str(end))
                chunk_counter = chunk_counter + 1
                start += chunk_length - overlap

    except Exception as e:
        print("Error processing file: ", file)
        print(e)


def audio_split_pooling(input_directory, output_directory, labelled):
    # Run split_audio with pooling for files in input_directory

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
        pool.starmap(_split_audio, [(file, output_directory, labelled) for file in audio_files])


