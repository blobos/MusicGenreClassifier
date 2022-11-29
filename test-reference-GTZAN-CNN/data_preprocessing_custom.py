#based on #from https://www.youtube.com/watch?v=szyGiObZymo

import os
import librosa

DATASET_PATH = "genre_dataset_"
jSON_PATH = 'data.json'

SAMPLE_RATE = 22050
DURATION = 30 #30 sec per chunk
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
#hop length = sliding window
    #dictionary to store data
    data = {
        "mapping":  [], #genre labels
        "mfcc": [], #data for each segment
        "labels": [] #label index
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments )
    #loop through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path))  #i = count, dirpath = root, dinrnames = genres...

        #ensure not at root and start in genre folders
        if dirpath is not dataset_path:

            #save semantic label
            dirpath_components = dirpath.split("/") #root/genre => ["root", "genre"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)

            #process files for a specific genre
            for f in filenames:

                file_path =  os.path.join(dirpath,f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                #process segments extracting mfcc and storing data(MFCC)
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                hop_length= hop_length)
