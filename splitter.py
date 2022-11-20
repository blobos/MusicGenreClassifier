from pydub import AudioSegment
import os


def splitter():
    input_directory = "/media/aaron/My Passport/FYP/sample_tracks/"
    output_directory = "/media/aaron/My Passport/FYP/chunks/"

    for filename in os.listdir(input_directory):
        f = os.path.join(input_directory, filename)
        if os.path.isfile(f):
            # print(f)
            # print(filename)

            audio_track = AudioSegment.from_file(f)

            n = len(audio_track)
            print(n)
            interval = 30 * 1000  # pydub calculates in millisec
            overlap = 15 * 1000
            # Make chunks of one sec
            counter = 1
            # Export all of the individual chunks as wav files

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

                output_filename = filename + "_chunk_" + "%02d" % (counter,) + '.ogg'

                chunk.export(output_directory + output_filename, format="ogg")
                print("Processing chunk " + str(counter) + ". Start = " + str(start) + " end = " + str(end))
                counter = counter + 1
