Model now in CNN folder and DataPreprocessing.

Current:  
training
![loss Graph.png](CNN%2Ftrained%2Fcheckpoints_14_Epoch_no_val_improvement_in_10%2Floss%20Graph.png)
lowest val loss
![lowest_val_loss.png](CNN%2Ftrained%2Fcheckpoints_14_Epoch_no_val_improvement_in_10%2Flowest_val_loss.png)
highest val acc
![high_val_acc.png](CNN%2Ftrained%2Fcheckpoints_14_Epoch_no_val_improvement_in_10%2Fhigh_val_acc.png)
latest(14)
![14_epochs_confusion_matrix_latest.png](CNN%2Ftrained%2Fcheckpoints_14_Epoch_no_val_improvement_in_10%2F14_epochs_confusion_matrix_latest.png)

Done:  
fixed training loss not changing (pytorch crossentropyloss already contains softmax)
Add patience, validation to training, add log to training, 5th epoch & best saves
clean preprocessing script and added conditions for test set
2. bug: can't use Cuda for inference. why though? 



To do:  
add aggregation to inference  
transfer learning - copy other network




