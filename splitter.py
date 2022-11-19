from pydub import AudioSegment


input_directory = "/media/aaron/My Passport/FYP/sample_tracks/"
output_directory = "/media/aaron/My Passport/FYP/chunks/"

audio_track = AudioSegment.from_file("/media/aaron/My Passport/FYP/sample_tracks/Metal/Black/05. War.mp3")

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

    filename = 'a' + str(counter) + '.ogg'
    chunk.export(output_directory+filename, format="ogg")
    print("Processing chunk " + str(counter) + ". Start = " + str(start) + " end = " + str(end))

    counter = counter + 1
