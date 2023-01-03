from pydub import AudioSegment
import os
import csv

# GENRES = [
#     "Metal",
#     "Rock",
#     "Electronic",
#     "Jazz",
#     "Classical",
#     "Pop"
# ]
#
# SUBGENRES = ["Black", "Death", "Thrash",
#              "Sludge", "Alternative", "Dreampop", "Indie", "Post", "Progressive", "Psychedelic",
#              "Synthwave", "Techno", "House", "Trance"]

SUBGENRES = ["Black Metal",
             "Death Metal",
             "Thrash Metal",
             "Doom Metal",
             "Alternative Rock",
             "Dreampop (Rock)",
             "Indie Rock",
             "Post Rock",
             "Progressive Rock",
             "Psychedelic Rock",
             "Folk Rock",
             "Synthwave (Electronic)",
             "Techno (Electronic)",
             "House (Electronic)",
             "Trance (Electronic)",
             "Classical",
             "Pop",
             "Indie Pop"]
#add Classical subgenres? Experimental subgenre? Classic Rock? Glam Rock?

INTERVAL = 30 * 1000  # pydub calculates in millisec
OVERLAP = 15 * 1000
MAX_TRACK_LENGTH = 4 * 60 * 1000

# to do:
# audio adjustments?
def track2chunks(input_directory, output_directory, csv_directory):

    # CSV
    header = ["Genre", "Genre Index", "Subgenre", "Subgenre Index", "Subgenre Track Number", "Filename"]
    with open(csv_directory, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for genre in os.listdir(input_directory):
            # f = os.path.join(input_directory, filename)
            # if os.path.isfile(f):
            # print(f)
            print("genre: ", genre)
            genre_directory = os.path.join(input_directory, genre)
            for subgenre in os.listdir(genre_directory):
                print("subgenre: ", subgenre)
                subgenre_directory = os.path.join(genre_directory, subgenre)
                #dictionary/mapping??
                if subgenre == "black":
                    subgenre_id = 0
                elif subgenre == "death":
                    subgenre_id = 1
                elif subgenre == "thrash":
                    subgenre_id = 2
                elif subgenre == "doom":
                    subgenre_id = 3
                elif subgenre == "alternative":
                    subgenre_id = 4
                elif subgenre == "dreampop":
                    subgenre_id = 5
                elif subgenre == "indie":
                    if genre == "rock":
                        subgenre_id = 6
                    elif genre == "pop":
                        subgenre_id = 17
                elif subgenre == "post":
                    subgenre_id = 7
                elif subgenre == "progressive":
                    subgenre_id = 8
                elif subgenre == "psychedelic":
                    subgenre_id = 9
                elif subgenre == "folk":
                    subgenre_id = 10
                elif subgenre == "synthwave": #drop microgenre?
                    subgenre_id = 11
                elif subgenre == "techno":
                    subgenre_id = 12
                elif subgenre == "house":
                    subgenre_id = 13
                elif subgenre == "trance":
                    subgenre_id = 14
                elif subgenre == "classical":
                    subgenre_id = 15
                elif subgenre == "pop":
                    subgenre_id = 16

                subgenre_track_counter = 1
                for track in os.listdir(subgenre_directory):
                    print("track: ", track)
                    track_path = os.path.join(subgenre_directory, track)
                    #load audio track
                    audio_track = AudioSegment.from_file(track_path)
                    # Split to Chunk
                    track_length = len(audio_track)
                    print(track_length)

                    # Make chunks of one sec
                    chunk_counter = 1
                    #write to csv
                    total_subgenre_track_chunks = track_length/(INTERVAL-OVERLAP)#30 sec chunks with 15 overlap
                    entry = [genre, subgenre, subgenre_id, subgenre_track_counter, total_subgenre_track_chunks, track]
                    print(entry)
                    writer.writerow(entry)

                    # Iterate from 0 to end of the file,
                    # with increment = interval
                    # why 2 n
                    # for i in range(0, 2 * n, interval):
                    if track_length > MAX_TRACK_LENGTH:
                        track_length = MAX_TRACK_LENGTH
                    for i in range(0, track_length * 2, INTERVAL):
                        if i == 0:
                            start = 0
                            end = INTERVAL
                            chunk = audio_track[start:end]
                        else:
                            start = end - OVERLAP
                            end = start + INTERVAL
                            chunk = audio_track[start:end]

                            if end >= track_length:
                                continue

                        sort_name = subgenre + "_" + genre + "_" + str("%04d" % (subgenre_track_counter,))
                        # print(sort_name)
                        output_filename = sort_name + "_chunk_" + "%02d" % (chunk_counter,) + '.ogg'
                        # print(output_filename)
                        chunk_directory = output_directory + genre + "/" + subgenre + "/" + sort_name + "/"
                        # print(chunk_directory)
                        # mode = 0o666
                        if not os.path.exists(chunk_directory):
                            os.makedirs(chunk_directory)

                        chunk.export(chunk_directory + output_filename, format="ogg")
                        # print("Processing chunk " + str(chunk_counter) + ". Start = " + str(start) + " end = " + str(end))
                        chunk_counter = chunk_counter + 1

                    subgenre_track_counter = subgenre_track_counter + 1
