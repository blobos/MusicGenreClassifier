import os
import shutil

import pandas as pd
import ast

csv_path = "/FYP/data/FMA/tracks.csv"

fma_subgenres = {167: "black_metal", 101: "death_metal", 182: "house_electronic", 181: "techno_electronic",
                 359: "dreampop", 98: "progressive_rock", 26: "post_rock", 25: "punk"}

df = pd.read_csv(csv_path)
# print(df.head())
# print(df.columns)
# print(type(df))


# clean up metadata
temp = df.dropna(subset=['genres'])
temp = temp[['trackid', 'genres']]
temp = temp[temp['genres'].str.startswith('[')]
# print("temp head: \n", temp.head())
# print(temp['genres'][0])
# print(type(temp['genres'][0]))

# string to list from cell
temp['genres'] = temp['genres'].apply(lambda x: ast.literal_eval(x))


# print(temp['genres'].iloc[1][0])
# print(temp['genres'].iloc[1][0] == 21)
# print(type(temp['genres'].iloc[1][0]))


def extract_genre(genre_num, track_directory='/home/student/Music/1/FYP/data/FMA/fma_medium/fma_medium'):
    output_dir = "/home/student/Music/1/FYP/data/train/FMA_extracted/" + fma_subgenres.get(genre_num)
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # get tracks
    # tracks repeated/shared amongst genres
    # track_series = temp[temp['genres'].apply(lambda x: genre_num in x)]['trackid']
    # get track if subgenre is in list, but not other keys in the wanted subgenres
    check_dict = lambda lst: genre_num in lst and not any(key in fma_subgenres and key != genre_num for key in lst)
    track_series = temp[temp['genres'].apply(check_dict)]['trackid']
    # add leading 0's
    track_series = track_series.apply(lambda x: '{:06d}'.format(x))
    # ouput_dir = /home/student/Music/1/FYP/data/train/chunks/alternative_rock

    for track in track_series:
        folder = track[:3]
        filename = track[3:] + '.mp3'
        # print("/" + folder + "/" + folder + filename)
        # print("test", track)
        full_path = track_directory + "/" + folder + "/" + folder + filename
        if os.path.isfile(full_path):
            shutil.copy(full_path, output_dir)
        else:
            print(f"{folder + filename} doesn't exist: {fma_subgenres.get(genre_num)}")

if __name__ == "__main__":
    extract_genre(167)  # black
    extract_genre(101)  # death
    extract_genre(182)  # house
    extract_genre(181)  # techno
    extract_genre(25)   # punk
    extract_genre(359)  # shoegaze as dreampop
    extract_genre(98)   # prog
    extract_genre(26)   # post

# FMA folder structure
# ---fma_medium/
# ------/000/
# ----------/000002.mp3
# ...
# ----------/000998.mp3
# ------/001/
# ...
# ------/155/
# ----------/155064.mp3
