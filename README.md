# LS-DDPM

<img src="https://github.com/lindan1128/LS-DDPM/blob/main/gif.gif"/>

## Requirements

The code requires:
  * Python 3.8 or higher
  * Numpy 1.24.4 or higher
  * Pandas 1.3.4 or higher
  * Keras 2.7.0 or higher
  * Tensorflow 2.7.0 or higher

 		Install all required packages
		pip install -r requirements.txt

  ## Modeling
  
  		python timediffusion.py
  The key hyperparameters for the model are:
	* --path PATH           Path to the data file. Please provide the absolute path
	* --step STEP           Time step for diffusion
	* --epoch EPOCH         The number of training epoch
	* --batch_size BATCH_SIZE Training batch size
	* --new_num NEW_NUM     The number of generating new samples

   The output for the model are:
	* cp.ckpt: checkpoint file
	* new_samples.npy: generated new samples in .npy format

 
