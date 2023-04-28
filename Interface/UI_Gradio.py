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
    waveform = gr.make_waveform(audio_file, bg_color="#ffffff", bar_count=100, fg_alpha=0.5, bars_color=("#db2777", "#fbcfe8"))
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    #write audiofile locally as model input is path of file
    sf.write(output_directory + "audio_test_file.wav", audio_file[1], samplerate=audio_file[0])
    return (output_directory + "audio_test_file.wav"), waveform


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

    # print(predictions)
    top_predictions = Counter(predictions).most_common()
    total_count = sum(count for _, count in top_predictions)
    probabilities = {genre: count / total_count for genre, count in top_predictions}
    probabilities = dict(probabilities)
    # top_predictions = [str(prediction[0])+": "+str("%.2f" %(prediction[1]/len(predictions)*100)) + "%" for prediction in top_predictions]
    # print(top_predictions)
    # while len(top_predictions) < 3:
    #     top_predictions.append("N/A")
    # print(top_predictions)

    # for i in range(len(top_predictions)):
    #     top_predictions[i] = top_predictions[i][0].split("_")[0]
    #     top_predictions[i] = top_predictions[i][0].capitalize() + top_predictions[i][1:]

    # print("final class prediction:", top_predictions[0], top_predictions[1], top_predictions[2])
    return probabilities


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


def combined(audio_file, model_dir="/home/student/Music/1/FYP/MusicGenreClassifier/CRNN/checkpoints/"):
    prediction_dir = "/home/student/Music/1/FYP/MusicGenreClassifier/Interface/predictions/"
    audio_file_path, waveform = get_file_path(audio_file, prediction_dir)
    print("audio file path:", audio_file_path)
    test_directory, csv_path = file_prep(audio_file_path, prediction_dir)
    model_path = model_dir + "lowest_val_loss.pth"
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
    return waveform, prediction#[0], prediction[1], prediction[2]


description = "# Instructions: \n"\
                '&emsp;&emsp;Upload an audio file and click **Submit**\n'\
                '&emsp;&emsp;Track length must 30 seconds or longer and bitrate of at least 44.1khz\n'\
                '&emsp;&emsp;The predictions will only display subgenres that are included in the list at the bottom ' \
              'of the page.'

article = "## How it works:\n" \
          "&emsp;&emsp;The audio file is split into 30 second truncated chunks and each chunk is run through an " \
          "CRNN for individual predictions. The top 3 subgenre with the most predictions are then output.\n" \
          "## Trained subgenres: \n " \
          "&emsp;&emsp;**Metal**: Black metal, Death metal, Heavy metal, Thrash metal\n " \
          "&emsp;&emsp;**Rock**: Dreampop, Post rock, Progressive rock, Punk rock\n" \
          "&emsp;&emsp;**Electronic**: House, Synthwave, Techno, Trance\n" \
          "*https://github.com/blobos/MusicGenreClassifier*"

interface = gr.Interface(fn=combined,
                         description=description,
                         article=article,
                         inputs=gr.Audio(label="Audio File"),
                         outputs=[gr.Video(label="Waveform"), gr.Label(num_top_classes=11, label="Prediction")],#gr.Label(label="Prediction"),gr.Label(label="Prediction")],
                         title="Music Subgenre Classifier",
                         allow_flagging="never",
                         theme="ParityError/Anime")
interface.launch(share=True)

