#file size should be 5.3MB (5,292,044 bytes) at 30secs
#find, log and delete files
import os
import csv

def find_files(path, size):
    files_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path) and os.path.getsize(file_path) < size:
                files_list.append([file_path, os.path.getsize(file_path), size - os.path.getsize(file_path)])
    return files_list


def delete_files(files_list):
    for file_path in files_list:
        # print(file_path[0])
        # os.remove(file_path[0])
        print(f"Deleted file: {file_path}")


path = "/home/student/Music/1/FYP/data/train/chunks"
size = 5292044 # size in bytes, change to desired size
csv_file = "files.csv"

files = find_files(path, size)

# write files list to CSV file
with open(csv_file, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["File Path", "Size (Bytes)", "Size Difference (Bytes)"])
    for row in files:
        writer.writerow(row)

print(f"Found {len(files)} files smaller than {size} bytes, saved to {csv_file}")

delete_files(files)
