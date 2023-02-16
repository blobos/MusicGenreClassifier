batchsize = 1
epochs = 5
![train_loss_fix.txt_train_Loss.png](CNN%2Fcheckpoints3%2Ftrain_loss_fix.txt_train_Loss.png)
training (loss.item)log outliers:[training_loss_outliers.txt](CNN%2Fcheckpoints3%2Ftraining_loss_outliers.txt)
training log: [train_loss_fix.txt](CNN%2Fcheckpoints3%2Ftrain_loss_fix.txt)
![val-loss.txt_val_Loss.png](CNN%2Fcheckpoints3%2Fval-loss.txt_val_Loss.png)
[val-loss.txt](CNN%2Fcheckpoints3%2Fval-loss.txt)

normal training (vgg16 /w batchnorm + dropout)
![training_log.txt_Loss.png](CNN%2Fcheckpoints%2Ftraining_log.txt_Loss.png)
![lowest_val_loss.pth_confusion_matrix.png](CNN%2Ftrained%2Fvgg16%2Flowest_val_loss.pth_confusion_matrix.png)
[lowest_val_loss.pth_classification_report.txt](CNN%2Fcheckpoints%2Flowest_val_loss.pth_classification_report.txt)

Previous:
3.5 subgenres missing from training data

Current: Running again, vgg16 and smaller

Done:  
fixed training loss not changing (pytorch crossentropyloss already contains softmax)
Add patience, validation to training, add log to training, 5th epoch & best saves
clean preprocessing script and added conditions for test set
2. bug: can't use Cuda for inference. why though? 



To do:  
add aggregation to inference  
transfer learning - copy other network




