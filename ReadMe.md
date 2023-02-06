Model now in CNN folder and DataPreprocessing.

Current:  
training


Done:  
fixed training loss not changing (pytorch crossentropyloss already contains softmax)
Add patience, validation to training, add log to training, 5th epoch & best saves
clean preprocessing script and added conditions for test set  
debugged training?
debugging: 1. all predictions indie rock (66 epoch)
![img.png](img.png)
2. bug: can't use Cuda for inference. why though? 



To do:  
add aggregation to inference  
transfer learning - copy other network




