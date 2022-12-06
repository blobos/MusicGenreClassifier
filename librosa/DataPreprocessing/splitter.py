from pydub import AudioSegment
import os
import csv

GENRES = [
    "Metal",
    "Rock",
    "Electronic",
    "Jazz",
    "Classical",
    "Pop"
]

SUBGENRES = ["Black", "Death", "Thrash",
             "Sludge", "Alternative", "Dreampop", "Indie", "Post", "Progressive", "Psychedelic",
             "Synthwave", "Techno", "House", "Trance"]
# to do:
# audio adjustments?
def track2chunks():
    input_directory = "/media/aaron/My Passport/FYP/sample_tracks/"
    output_directory = "/media/aaron/My Passport/FYP/chunks/"
    csv_directory = "/media/aaron/My Passport/FYP/"

    # CSV
    header = ["Genre", "Genre Index", "Subgenre", "Subgenre Index", "Subgenre Track Number", "Filename"]
    with open(csv_directory + "track_genre_label.csv", "w") as csv_file:
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
                subgenre_track_counter = 1
                for track in os.listdir(subgenre_directory):
                    print("track: ", track)
                    track_path = os.path.join(subgenre_directory, track)
                    #load audio track
                    audio_track = AudioSegment.from_file(track_path)
                    #write to csv
                    entry = [genre, subgenre, subgenre_track_counter, track]
                    print(entry)
                    writer.writerow(entry)
                    # Split to Chunk
                    n = len(audio_track)
                    print(n)
                    interval = 30 * 1000  # pydub calculates in millisec
                    overlap = 15 * 1000
                    # Make chunks of one sec
                    chunk_counter = 1

                    # Iterate from 0 to end of the file,
                    # with increment = interval
                    # why 2 n
                    # for i in range(0, 2 * n, interval):
                    for i in range(0, n * 2, interval):
                        if i == 0:
                            start = 0
                            end = interval
                            chunk = audio_track[start:end]
                        else:
                            start = end - overlap
                            end = start + interval
                            chunk = audio_track[start:end]

                            if end >= n:
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
