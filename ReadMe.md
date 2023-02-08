                      precision    recall  f1-score   support

    alternative_rock       0.00      0.00      0.00       530
         black_metal       0.29      0.04      0.07       244
         death_metal       0.95      0.07      0.13       294
       dreampop_rock       0.19      0.24      0.21       728
         heavy_metal       0.00      0.00      0.00       418
    house_electronic       0.00      0.00      0.00       834
          indie_rock       0.07      0.22      0.10       240
           post_rock       0.00      0.00      0.00       308
    progressive_rock       0.04      0.23      0.07       136
           punk_rock       0.10      0.08      0.09       418
synthwave_electronic       0.23      0.59      0.34       338
   techno_electronic       0.35      0.14      0.20       554
        thrash_metal       0.13      0.92      0.23       174
   trance_electronic       0.26      0.44      0.33       280

            accuracy                           0.16      5496
           macro avg       0.19      0.21      0.13      5496
        weighted avg       0.17      0.16      0.12      5496

Current:  
![logLoss.png](CNN%2Ftrained%2F54%2FlogLoss.png)
![lowest_val_loss.pth_confusion_matrix.png](CNN%2Ftrained%2F54%2Flowest_val_loss.pth_confusion_matrix.png)
![latest.pth_confusion_matrix.png](CNN%2Ftrained%2F54%2Flatest.pth_confusion_matrix.png)
Done:  
fixed training loss not changing (pytorch crossentropyloss already contains softmax)
Add patience, validation to training, add log to training, 5th epoch & best saves
clean preprocessing script and added conditions for test set
2. bug: can't use Cuda for inference. why though? 



To do:  
add aggregation to inference  
transfer learning - copy other network




