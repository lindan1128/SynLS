# LS-DDPM

## Requirements

The code requires:
  * Python 3.8 or higher
  * Numpy 1.24.4 or higher
  * Pandas 1.3.4 or higher
  * Keras 2.7.0 or higher
  * Tensorflow 2.7.0 or higher

 		Install all required packages
		pip install -r requirements.txt

## Datasets
A total of 4 cow wearable sensor datasets were used in this study:
	
 * D1: a low-resolution database that comprises daily activity and rumination data recorded by a neck-mounted and ear-mounted electronic activity and rumination monitoring tag (both SCR Dairy, Netanya, Israel). This database was collected from 185 Holstein-Friesian cows at a commercial dairy farm in Cayuga County, New York, USA, from March 2021 to March 2022.
  [Paper link](https://www.sciencedirect.com/science/article/abs/pii/S0168169923000261)

 * D2: a low-resolution database that consists of behavioral data regarding lying, chewing, and activity times recorded by an ear-tag-based accelerometer (Smartbow GmbH, Weibern, Austria). This database was collected from 369 Holstein-Friesian cows at a dairy farm located in the Po Valley, Italy, for the period from August 2019 to September 2021.
  [Paper link](https://link.springer.com/article/10.1007/s00484-023-02561-w)
  [Data link](https://portal.edirepository.org/nis/metadataviewer?packageid=edi.1406.1)
  	
* D3: a high-resolution database containing behavioral data including hourly eating, resting and in alleys times. This database was recorded by the neck collars CowView system (GEA Farm Technology, Bonen, Germany) attached to 28 Holstein-Friesian cows in INRAE Herbipôle experimental farm in France from October 2018 to April 2019.
    
* D4: The database presents the same format with D3, but it was collected from 300 Holstein-Friesian cows in INRAE Herbipôle experimental farm in France from December 2014 to December 2015. 
  [Paper link](https://www.sciencedirect.com/science/article/abs/pii/S1046202320301985#:~:text=Circadian%20changes%20in%20cows%20link,help%20detect%20animals%20needing%20care)
  [Data link](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.15454/52J8YS)

   The raw data and pre-processed data could be found in the dataset fold.

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

 
