Model now in CNN folder and DataPreprocessing.

Currently debugging: predictions (inference.py) 
weights of model are GPU tensor, input tensor not
 (Tensor input, Tensor weight, Tensor bias, tuple of ints stride, str padding, tuple of ints dilation, int groups)
     didn't match because some of the arguments have invalid types: (!builtin_function_or_method!, !Parameter!, !Parameter!, !tuple!, !tuple!, !tuple!, int)
Will find a existing model for transfer learning as debugging consuming too much time.

Done:
training complete
clean preprocessing script and added conditions for test set
dataloader added conditions for test set

To do:
Add validation to training, add log to training, add aggregation to inference



