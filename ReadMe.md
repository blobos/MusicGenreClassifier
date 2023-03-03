![song level prediction_confusion_matrix.png](CNN%2Ftrained%2Fvgg16%2Fsong%20level%20prediction_confusion_matrix.png)
![lowest_val_loss.pth_confusion_matrix.png](CNN%2Ftrained%2Fvgg16%2Flowest_val_loss.pth_confusion_matrix.png)
Previous:


Done:  
fixed training loss not changing (pytorch crossentropyloss already contains softmax)
Add patience, validation to training, add log to training, 5th epoch & best saves
clean preprocessing script and added conditions for test set
2. bug: can't use Cuda for inference. why though? 



To do:  
add aggregation to inference  
transfer learning - copy other network




