import os
import shutil
import sys
import torch
import torchaudio.transforms
import gradio as gr
import time
from pydub import AudioSegment
from tqdm import tqdm
import soundfile as sf

from collections import Counter
from FYP.MusicGenreClassifier.Predict.inference_with_aggregate_single_track import predict
from FYP.MusicGenreClassifier.CRNN.CRNN_biLSTM import NetworkModel
from FYP.MusicGenreClassifier.DataPreprocessing.datasetmelspecprep import DatasetMelSpecPrep
from FYP.MusicGenreClassifier.DataPreprocessing.chunks_to_CSV import chunks_to_CSV

class_mapping = [
    "Black Metal",
    "Death Metal",
    "Dreampop",
    "Heavy Metal",
    "House",
    "Post rock",
    "Progressive rock",
    "Punk rock",
    "Synthwave",
    "Techno",
    "Thrash metal",
    "Trance"
]


def get_file_path(audio_file, output_directory):
    # print(audio_file)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    sf.write(output_directory + "audio_test_file.wav", audio_file[1], samplerate=audio_file[0])
    return (output_directory + "audio_test_file.wav")


def predict_vote(dmsp, progress=gr.Progress(track_tqdm=True)):
    predictions = []
    # print(len(dmsp))
    progress(0, desc="Starting")
    progress_bar = tqdm(dmsp, desc="Predicting", unit='chunks')
    for input in progress_bar:
        networkModel = NetworkModel()
        # print("input", input)
        predicted = predict(networkModel, input)
        predictions.append(class_mapping[predicted.argmax(0)])

    print(predictions)
    top_predictions = Counter(predictions).most_common(3)
    print(top_predictions)
    top_predictions = [prediction[0] for prediction in top_predictions]
    while len(top_predictions) < 3:
        top_predictions.append("N/A")

    # for i in range(len(top_predictions)):
    #     top_predictions[i] = top_predictions[i][0].split("_")[0]
    #     top_predictions[i] = top_predictions[i][0].capitalize() + top_predictions[i][1:]

    # print("final class prediction:", top_predictions[0], top_predictions[1], top_predictions[2])
    return top_predictions


def split_audio(audio_track, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    audio_track = AudioSegment.from_file(audio_track)
    chunk_length = 30 * 1000  # 30 seconds pydub calculates in millisec
    overlap = 0.5 * chunk_length  # 15 seconds
    bitrate = "128k"  # 128kbps

    start = 0
    max_track_length = 4 * 60 * 1000  # 4 minutes
    track_length = len(audio_track)
    # print("track_length", track_length)

    # if >4 minutes, trim to 4 minutes
    if track_length > max_track_length:
        track_length = len(audio_track[:max_track_length])
    else:
        # trim to nearest 30 seconds
        track_length = track_length - (track_length % (30 * 1000))

    chunk_counter = 1
    total_subgenre_track_chunks = track_length / (chunk_length - overlap) - 1
    # print("total subgenre_track_chunks:", total_subgenre_track_chunks)
    # 30 sec chunks with 15 overlap, -1 because chunks are 30 sec not 15, so cannot fit chunk into last 15 sec

    # print(file_name, start, track_length)
    # print("-------------------")
    file_name = "prediction_track_chunks"
    chunk_directory = output_directory + "/" + file_name + "/"
    if not os.path.exists(chunk_directory):
        os.mkdir(chunk_directory)
    while chunk_counter < total_subgenre_track_chunks + 1:
        end = start + chunk_length
        chunk = audio_track[start:end]

        output_filename = file_name + "_chunk_" + str("%02d" % (chunk_counter,)) + 'of' + str(
            "%02d" % (total_subgenre_track_chunks,)) + '.wav'

        if not os.path.exists(chunk_directory):
            # print(f"creating chunk directory: f{chunk_directory}")
            os.makedirs(chunk_directory)
        # print("creating: ", output_filename)
        chunk.export(chunk_directory + output_filename, format="wav", bitrate=bitrate)
        # print("Processing chunk " + output_filename + ". Start = " + str(start) + " end = " + str(end))
        chunk_counter = chunk_counter + 1
        start += chunk_length - overlap
        # else:
        # print(f"file f{output_filename} in directory")
    return chunk_directory


def file_prep(file, output_directory):
    csv_path = output_directory + "prediction_track" + ".csv"
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    chunk_directory = split_audio(file, output_directory)  # exports chunks to output_dir/chunk
    # print(csv_path)
    chunks_to_CSV(chunk_directory, csv_path, False)
    # print(chunk_directory, csv_path)
    return chunk_directory, csv_path


def combined(audio_file, model_dir="/home/student/Music/1/FYP/MusicGenreClassifier/CRNN/CRNN_test/"):
    prediction_dir = "/home/student/Music/1/FYP/MusicGenreClassifier/Interface/predictions/"
    audio_file_path = get_file_path(audio_file, prediction_dir)
    print("audio file path:", audio_file_path)
    test_directory, csv_path = file_prep(audio_file_path, prediction_dir)
    model_path = model_dir + "highest_val_acc.pth"
    parameters = model_dir + "parameters.txt"

    with open(parameters, "r") as f:
        line = f.readlines()
        SAMPLE_RATE = int(line[6].split()[-1])
        NUM_SAMPLES = int(line[7].split()[-1])
        N_FFT = int(line[8].split()[-1])
        HOP_LENGTH = int(line[9].split()[-1])
        N_MELS = int(line[10].split()[-1])
        # print(f"Sample Rate: {SAMPLE_RATE}\n"
        #       f"N_FFT: {N_FFT}\n"
        #       f"Hop length: {HOP_LENGTH}\n"
        #       f"N_MELS: {N_MELS}")

    ANNOTATIONS_FILE = csv_path
    AUDIO_DIR = test_directory
    networkModel = NetworkModel()
    state_dict = torch.load(model_path)
    networkModel.load_state_dict(state_dict)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    dmsp = DatasetMelSpecPrep(ANNOTATIONS_FILE,
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              "cpu",
                              labelled=False)
    prediction = predict_vote(dmsp)
    os.remove(audio_file_path)
    shutil.rmtree(prediction_dir)
    return prediction


description = "# Instructions: \n" \
              'Upload an audio file and click "Submit"\n' \
              "### Trained subgenres: \n " \
              "Black metal, Death metal, Dreampop, Heavy metal, House, Post rock, Progressive rock, Punk rock, " \
              "Synthwave, Techno, Thrash Metal, Trance" \
              "\n"
interface = gr.Interface(fn=combined,
                         description=description,
                         inputs=gr.Audio(label="Audio File"),
                         outputs=[gr.Label(label="Top Prediction"),gr.Label(label="2nd Prediction"),gr.Label(label="3rd Prediction")],
                         title="Music Subgenre Classifier",
                         allow_flagging="never",
                         theme="soft")
interface.launch(share=True)

