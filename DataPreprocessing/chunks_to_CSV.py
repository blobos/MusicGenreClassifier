import os
import csv


def chunks2CSV():
    # Define the top level directory to start the search from
    chunk_directory = '/home/student/Music/1/FYP/data/train/chunks/'
    csv_path = "/home/student/Music/1/FYP/data/train/train_annotations.csv"

    subgenre_map = {0: "alternative_rock", 1: "black_metal", 2: "death_metal", 3: "dreampop_rock", 4: "heavy_metal",
                    5: "house_electronic", 6: "indie_rock", 7: "post_rock", 8: "progressive_rock", 9: "punk_rock",
                    10: "synthwave_electronic", 11: "techno_electronic", 12: "thrash_metal", 13: "trance_electronic"}

    # Create the CSV file and write the header

    # os.chmod("/media/student/s280016Aaron/FYP/data", 0o755)

    with open(csv_path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['chunk_file_name', 'subgenre_track_counter', 'subgenre', 'subgenre_id', 'chunk_number',
                         'total_chunk_number'])

        # Recursively search through all subdirectories and add each file to the CSV
        for subdir, dirs, files in os.walk(chunk_directory):
            for file in files:
                if file.endswith('.mp3'):
                    chunk_file_name = file
                    subgenre_track_counter = 'unknown'
                    subgenre_id = 'unknown'
                    subgenre = 'unknown'
                    chunk_number = 'unknown'
                    total_chunk_number = 'unknown'
                    filename = file.split("_")
                    # black_metal_002_chunk_00of21.mp3
                    subgenre = filename[0] + "_" + filename[1]
                    subgenre_track_counter = filename[2]
                    chunk_counters = filename[-1].split(".")
                    chunk_number = chunk_counters[0][0:2]
                    total_chunk_number = chunk_counters[0][4:]
                    for id, subgenre_name in subgenre_map.items():
                        if subgenre == subgenre_name:
                            subgenre_id = id
                            break
                    writer.writerow([chunk_file_name, subgenre_track_counter, subgenre, subgenre_id, chunk_number,
                                     total_chunk_number])
