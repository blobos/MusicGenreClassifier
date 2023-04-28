# Music Subgenre Classifier
![Screenshot 2023-04-28 at 12-38-18 Music Subgenre Classifier.png](ReadMe%2FScreenshot%202023-04-28%20at%2012-38-18%20Music%20Subgenre%20Classifier.png)
![Screenshot 2023-04-28 at 12-38-09 Music Subgenre Classifier.png](ReadMe%2FScreenshot%202023-04-28%20at%2012-38-09%20Music%20Subgenre%20Classifier.png)

Dataset includes 5,546 Full length tracks (4800+ tracks from FMA Dataset (https://github.com/mdeff/fma)
Unbalanced Dataset
### 12 Subgenres:
    Black metal (302 tracks),
    Death metal (395 tracks),
    Dreampop/Shoegaze (853 tracks),
    Heavy metal (203 tracks),
    House (686 tracks),
    Post rock (207 tracks),
    Progressive rock (762 tracks),
    Punk rock (762 tracks),
    Synthwave (209 tracks),
    Techno" (762 tracks),
    Thrash metal (205 tracks),
    Trance (200 tracks)

Alternative Rock and Indie Rock subgenres removed due to perceived ambiguity in subgenre features by author


### CRNN model predictions results
![training_log.txt_Loss.png](CRNN%2FCRNN_Final%2Ftraining_log.txt_Loss.png)
![song level prediction: lowest val loss_confusion_matrix_voting.png](CRNN%2FCRNN_Final%2Fsong%20level%20prediction%3A%20lowest%20val%20loss_confusion_matrix_voting.png)
#### Classification Report:
                  precision    recall  f1-score   support

         Black metal       0.21      0.18      0.19        17
         Death metal       0.40      0.62      0.49        16
            Dreampop       0.52      0.29      0.37        49
         Heavy metal       0.45      0.68      0.54        28
               House       0.69      0.74      0.71        54
           Post rock       0.43      0.15      0.22        20
    Progressive rock       0.08      0.22      0.11         9
           Punk rock       0.77      0.53      0.63        45
           Synthwave       0.68      0.54      0.60        24
              Techno       0.39      0.60      0.47        35
        Thrash metal       0.80      0.67      0.73        12
              Trance       0.86      0.67      0.75        18

            accuracy                           0.52       327
           macro avg       0.52      0.49      0.49       327
        weighted avg       0.56      0.52      0.52       327


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