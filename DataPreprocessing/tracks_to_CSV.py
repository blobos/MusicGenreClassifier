import os
import csv

#Not used

def chunks_to_CSV(chunk_directory, csv_path):
    subgenre_map = {0: "black_metal",
                    1: "death_metal",
                    2: "dreampop_rock",
                    3: "heavy_metal",
                    4: "house_electronic",
                    5: "post_rock",
                    6: "progressive_rock",
                    7: "punk_rock",
                    8: "synthwave_electronic",
                    9: "techno_electronic",
                    10: "thrash_metal",
                    11: "trance_electronic"}

    # Create the CSV file and write the header

    # os.chmod("/media/student/s280016Aaron/FYP/data", 0o755)

    with open(csv_path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['subgenre_id','subgenre', 'subgenre_track_counter', 'filename'])
        data = []
        # Recursively search through all subdirectories and add each file to the list `data`
        for subdir, dirs, files in os.walk(chunk_directory):
            for file in files:
                # if file
                track_filename = file
                filename = file.split("_")
                subgenre = filename[1] + "_" + filename[2]
                subgenre_track_counter = filename[0]
                album = filename[3][1:]
                album_track_number = filename[4]
                track_name,_ = os.path.splitext(filename[-1])
                track_name = track_name[:-1]
                for id, subgenre_name in subgenre_map.items():
                    if subgenre == subgenre_name:
                        subgenre_id = id
                        break
                    if album == "fma":
                        album_track_number = ""
                data.append([subgenre_id, subgenre, subgenre_track_counter, album, album_track_number, track_name])

        # Sort the data by the first column (chunk_file_name)
        data.sort(key=lambda x: (x[1], x[2]))

        # Write the sorted data to the CSV file
        writer.writerows(data)




if __name__ == "__main__":
    chunk_directory = "/home/student/Music/1/FYP/data/train/tracks/"
    csv_path = "/home/student/Music/1/FYP/data/tracks.csv"
    chunks_to_CSV(chunk_directory, csv_path)
