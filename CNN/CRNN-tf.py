import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Reshape, Bidirectional, Permute, BatchNormalization, LSTM, Dense
# Define the audio processing parameters
sr = 44100
duration = 30
hop_length = 512
n_fft = 1024
n_mels = 64
fmax = 8000
n_frames = 1 + int((sr * duration - n_fft + hop_length) / hop_length)


# Define the model architecture
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=((3, 3)), padding='same', activation='relu', input_shape=(n_mels, duration * sr // hop_length, 1)))
print(model.input_shape)
print(model.output_shape)
model.add(BatchNormalization(axis=3)) #on channel axis
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=.2))
print(model.output_shape)

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(rate=.2))
print(model.output_shape)

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(rate=.2))
print(model.output_shape)

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(rate=.2))
print(model.output_shape)
#4,161,256
#(batch_size, n_mels, time_steps, channels) to (batch_size, time_steps, n_mels, channels), time_steps == n_frames
model.add(Permute((2, 1, 3))) #permutation and reshape is necessary to make them as a time-series representation.
#161, 4, 256
print(model.output_shape, "permute ")
# model.add(Reshape((model.output_shape[1] * model.output_shape[2],model.output_shape[3])))
#(mel, )
#(n_mels * time_step * channels),channels = 1 (mono)

resize_shape = model.output_shape[2] * model.output_shape[3]
print(model.output_shape[1], resize_shape)
model.add(Reshape((model.output_shape[1], resize_shape)))

print(model.output_shape, "reshape ")
# model.add(Bidirectional(LSTM()))
#FIXME: find correct shape
model.add(LSTM(units=82432, activation='tanh', return_state=False, unroll=False))
# unrolling reduces computation, increases memory usage, unnecessary for short lengths (30 sec)

model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(12, activation="softmax"))
# Load the training data


def load_data(csv_file_path, sr=44100, duration=30, hop_length=512, n_fft=1024, n_mels=64, fmax=8000):
    print("Loading data:")
    # Read in the CSV file
    df = pd.read_csv(csv_file_path)
    # df['chunk_number'] = df['chunk_number'].astype(str).str.zfill(3)
    df['subgenre_track_counter'] = df['subgenre_track_counter'].astype(str).str.zfill(3)

    # Initialize the data and label arrays
    x = []
    y = []

    # Loop through each row in the CSV file

    # for index, row in df.iterrows():
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Load the audio file
        # file_directory = "/home/student/Music/1/FYP/data/train/chunks"
        file_directory = "/home/student/Music/1/FYP/data/miniDataset/chunks"
        subgenre = row['subgenre']
        subgenre_track_counter = row['subgenre_track_counter']
        chunk_number = row['chunk_number']
        filename = file_directory + "/" + subgenre + "/" + subgenre + "_" + subgenre_track_counter + "/" + row['chunk_file_name']
        # print(filename)
        audio, _ = librosa.load(filename, sr=sr, mono=True, res_type="kaiser_fast")


        # Compute the mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=fmax, hop_length=hop_length, n_fft=n_fft)
        #output = (y,x) = (n_mels, n_frames)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
        # Add the spectrogram to the data array
        # x.append(melspec_db.reshape(n_mels, -1, 1))
        x.append(mel_spec_db)
        # Load the label and total number of chunks for the audio file
        label = row['subgenre_id']
        total_chunks = row['total_chunk_number']

        # Add the label and total number of chunks to the label array
        y.append((label, total_chunks))


    # Convert the data and label arrays to numpy arrays
    x = np.array(x, dtype=object)
    y = np.array(y)

    return x, y

model.summary()

# x_train, y_train = load_data("/home/student/Music/1/FYP/data/train_annotations.csv")
x_train, y_train = load_data("/home/student/Music/1/FYP/data/mini_train_annotations.csv")

# Compile the model
#can change to sparse cat_crossentropy to get int instead of one-hot vec
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
history = model.fit(x_train, y_train, verbose=1, validation_split=0.2, batch_size=32, epochs=1)

