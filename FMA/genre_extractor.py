import pandas as pd
import ast

csv_path = "/FYP/data/FMA/tracks.csv"
# trackid, genre_top
# techno:181, house 182, punk 25, shoegaze 359
df = pd.read_csv(csv_path)
# print(df.head())
# print(df.columns)
techno_tracks = []
house_tracks = []
punk = []
shoegaze = []
# for line in df:
# techno_tracks = df[df['genres'].apply(lambda x: 181 in x)['trackid']]
# techno_tracks = df.loc[ast.literal_eval(df['genres']).apply(lambda x: 181 in x), 'trackid']
# test = (ast.literal_eval(df.at[4, 'genres']))
# print(test)
# print(type(test))
# print( 103 in test)
temp = df['genres']
print(df.columns.get_loc('genres'))
# print(temp)
# try:
#     temp = df['genres'].apply(lambda x: ast.literal_eval(x))
# except ValueError as e:
temp = df[['trackid','genres']]

print(temp)
print(temp['genres'])
def safe_eval(x):
    try:
        if isinstance(x, str) and x.startswith('['):
            return ast.literal_eval(x)
        else:
            return x
    except (ValueError, SyntaxError) as e:
        print(f"Error: {e} (value: {x})")
        return None
temp['genres'] = temp['genres'].apply(lambda x: safe_eval(x))
# techno_tracks = temp.loc[mask, 'trackid']


# for index, row in df.iterrows():
#     # if df.loc("genre_top") == "techno":
#     if row["genre"] == "Techno":
#         techno_tracks.append(row["trackid"])
#         print(row["trackid"])
#         # techno_tracks += df.loc("trackid")

print(techno_tracks)

# if genre_top == techno
# add to list

# for track in list:
# copy to other folder
