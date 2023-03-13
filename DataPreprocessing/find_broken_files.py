# file size should be 5.3MB (5,292,044 bytes) at  and 44.1khz
# find, log and delete files
import os
import csv
import shutil
from chunks_to_CSV import chunks_to_CSV


def find_files(path, size):
    files_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path) and os.path.getsize(file_path) < size:
                files_list.append([file_path, os.path.getsize(file_path), size - os.path.getsize(file_path)])
    # print(files_list)
    return files_list


def delete(files_list, dir=False):
    if dir:
        for file_path in files_list:
            #remove entire directory
            dir_to_remove = os.path.dirname(file_path[0])
            if os.path.exists(dir_to_remove):
                shutil.rmtree(dir_to_remove)
                print(f"Deleted directory of: {file_path[0]}")
    else:
        for file_path in files_list:
            #remove offending file
            # os.remove(file_path[0])
            print(f"Deleted: {file_path[0]}")



path = "/home/student/Music/1/FYP/data/train/chunks"
size = 5292044  # size in bytes, change to desired size
csv_file = "files.csv"

files = find_files(path, size)

# write files list to CSV file
with open(csv_file, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["File Path", "Size (Bytes)", "Size Difference (Bytes)"])
    for row in files:
        writer.writerow(row)

print(f"Found {len(files)} files smaller than {size} bytes, saved to {csv_file}")

delete(files)
#update csv
chunk_directory = "/home/student/Music/1/FYP/data/train/chunks"
csv_path = "/home/student/Music/1/FYP/data/train_annotations.csv"
chunks_to_CSV(chunk_directory, csv_path)