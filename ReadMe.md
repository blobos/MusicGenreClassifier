# Music Subgenre Classifier
![Screenshot from 2023-04-27 11-31-00.png](ReadMe%2FScreenshot%20from%202023-04-27%2011-31-00.png)
![Screenshot from 2023-04-27 11-30-36.png](ReadMe%2FScreenshot%20from%202023-04-27%2011-30-36.png)
Dataset includes 6,887 Full length tracks (4800+ tracks from FMA Dataset (https://github.com/mdeff/fma)
Unbalanced Dataset
### 12 Subgenres:
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

Alternative Rock and Indie Rock subgenres removed due to perceived ambiguity in subgenre features by author


CRNN model
![training_log.txt_Loss.png](CRNN%2FCRNN_Final%2Ftraining_log.txt_Loss.png)
![song level prediction_confusion_matrix_voting.png](CRNN%2FCRNN_Final%2Fsong%20level%20prediction_confusion_matrix_voting.png)
#### Classification Report:
                      precision    recall  f1-score   support

         Black Metal       0.43      0.18      0.25        17
         Death Metal       0.50      0.75      0.60        16
            Dreampop       0.55      0.45      0.49        49
         Heavy Metal       0.48      0.86      0.62        28
               House       0.64      0.78      0.70        54
           Post rock       0.80      0.20      0.32        20
    Progressive rock       0.13      0.33      0.19         9
           Punk rock       0.83      0.42      0.56        45
           Synthwave       0.86      0.50      0.63        24
              Techno       0.33      0.49      0.39        35
        Thrash metal       0.69      0.75      0.72        12
              Trance       0.90      0.50      0.64        18

            accuracy                           0.54       327
           macro avg       0.59      0.52      0.51       327
        weighted avg       0.61      0.54      0.54       327
# How to train:
Audio Files should be enumerated per genre/subgenre and renamed to the following format:
SubgenreTrackCounter_Subgenre_(FileName).extension

##### i.e: 
    "038_death_metal_(Demigod_06_Before Aeons Came).mp3"

#### To train other subgenres modify the hashmap
    DataPreprocessing/chunks_to_CSV.py

    def chunks_to_CSV(chunk_directory, csv_path, labelled=True):
        subgenre_map = {0: "black_metal", 1: "death_metal", 2: "dreampop_rock", 3: "heavy_metal",
                        4: "house_electronic", 5: "post_rock", 6: "progressive_rock", 7: "punk_rock",
                        8: "synthwave_electronic", 9: "techno_electronic", 10: "thrash_metal", 11: "trance_electronic"}


#### For CRNN, train using
    CRNN/train_with_validation_CRNN_acc.py
    Predict/CRNN/confusion_matrix_classification_report_multitrack.py #for confusion matrix
    Predict/plotLossLog_training_acc.py # To plot Loss and Accuracy