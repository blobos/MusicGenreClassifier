import pandas as pd

csv_path = "/home/student/Music/1/FYP/data/FMA/tracks.csv"
# trackid, genre_top

df = pd.read_csv(csv_path)
print(df.head)

techno_tracks = []
for line in df:
    if df.loc("genre_top") == "techno":
        techno_tracks += df.loc("trackid")

print(techno_tracks)

# if genre_top == techno
# add to list

# for track in list:
# copy to other folder
