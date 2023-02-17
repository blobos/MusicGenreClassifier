import pandas as pd
import ast

csv_path = "/home/student/Music/1/FYP/data/FMA/tracks.csv"
# trackid, genre_top
# techno:181, house 182, punk 25, shoegaze 359
df = pd.read_csv(csv_path)
# print(df.head())
# print(df.columns)
techno_tracks = []
house_tracks = []
punk_tracks = []
shoegaze_tracks = []

#clean up metadata
temp = df.dropna(subset=['genres'])
temp = temp[['trackid', 'genres']]
temp = temp[temp['genres'].str.startswith('[')]
# print("temp head: \n", temp.head())
# print(temp['genres'][0])
# print(type(temp['genres'][0]))

#string to list from cell
temp['genres'] = temp['genres'].apply(lambda x: ast.literal_eval(x))
# print(temp.head())
# print(temp['genres'][0])
# print(type(temp['genres'][0]))


techno_tracks = temp[temp['genres'].apply(lambda x: 181 in x)]['trackid']
house_tracks = temp[temp['genres'].apply(lambda x: 182 in x)]
shoegaze_tracks = temp[temp['genres'].apply(lambda x: 359 in x)]
punk_tracks = temp[temp['genres'].apply(lambda x: 25 in x)]


# print("techno tracks:\n", techno_tracks)

#for tracks in techno copy to other folder

track_directory = '/home/student/Music/1/FYP/data/FMA/fma_medium/'
#for trackid in tracks
print(type(techno_tracks))
print(techno_tracks['0'])

for trackid in techno_tracks:
    trackid = '{:06d}'.format(trackid)

#append leading 0's to track to get 6 digits
#get first 3 digits for folder
#copy track to other folder

#folder structure
#---fma_medium/
#------/000/
#----------/000002.mp3
#...
#----------/000998.mp3
#------/001/
#...
#------/155/
#----------/155064.mp3