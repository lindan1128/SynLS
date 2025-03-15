# SynLS

### SynLS: a framework for generating high-quality wearable sensor time series data validated in livestock monitoring

SynLS is a model designed for generating highly realistic livestock wearable sensor data using diffusion architecture and transformer encoder mechanism. 

![Image text](https://github.com/lindan1128/SynLS/blob/main/Workflow.png)

### Project structure

	SynLS/
	├── data/                         # Folder for datasets
	├── model/                        # Folder for source code (modeling)
	│   ├── timevae.py                # TimeVAE model
	│   ├── timegan.py                # TimeGAN model
	│   ├── diffusion.py              # SynLS source code
	│ 	├── utils.py                  # Utils to normalize data and create time step
	│ 	├── zoo.py              	  # Base moduler for TimeGAN and TimeVAE
	├── results/                      # Folder for some supplemental tables
	│   ├── Supplemental Table5
	│   ├── Supplemental Table6
	│   ├── Supplemental Table7
	│   ├── Supplemental Table7
	│   ├── Supplemental Table9
	│── timevae.py                    # Main function for TimeVae
	│── timegan.py                    # Main function for TimeGAN
	│── timediffusion.py              # Main function for SynLS
	├── README.md                     # Readme file
	└── requirements.txt              # Dependencies

### Requirements
The code requires

	* Python 3.8 or higher
	* Numpy 1.24.4 or higher
	* Pandas 1.3.4 or higher
	* Keras 2.7.0 or higher
	* Tensorflow 2.7.0 or higher

	# Install all required packages
	pip install -r requirements.txt

 
### Modeling

	python timediffusion.py 
	
	The key hyperparameters for the model are:
	* --path PATH           Path to the data file. Please provide the absolute path
	* --step STEP           Time step for diffusion, e.g., 10
	* --epoch EPOCH         The number of training epoch
	* --batch_size BATCH_SIZE Training batch size
	* --new_num NEW_NUM     The number of generating new samples
	* --output				Output file name for generated samples (default: new_samples.npy)

	The output for the model are:
	* cp.ckpt: checkpoint file
	* new_samples.npy: generated new samples in .npy format
