# DavidAI-WorkTrial
This repo contains my implementation of Rule Based and Neural approach to creating a MOS audio predictor. 

#Data
Locally in my repo I have a data folder with three subfolders: bvcc,somos,urgent. These represent the different data sources I used
to accquired baseline MOS data. Each of these folders contains WAV files and I also mantain a json mapping specific MOS score to a specific file. To replicate training you will have to download this data locally as it is too large to store on github. The urgent dataset can be found here: https://huggingface.co/datasets/urgent-challenge/urgent2024_mos, the bvcc data can be found here: https://zenodo.org/records/6572573, and the somos data can be found here: https://zenodo.org/records/6572573. 

#Training 

I have two neural models that need to be trained, the shallow and deep network. For the shallow network it is a simple sklearn MLP regressor so I did not include a training script. For the deep neural network you can train the model using deep_network.py in the inference folder. Thsi training leverages data_loaders in data_laoder.py in the inference folder. This data loader assumes you have done data downloading and have a data directory like the one I outlined above. Otherwise it will  not work. 

 # Testing
 I provided two scripts to test model performance both under the inference folder. The first script : prediction_generator.py loads all three models and runs testing on a test set of the urgent dataset. The second test folder is called compare_to_distill_mos.py which compares model prediction of three models on a random audio dataset to the distill_mos model. 

 #General Structure of Repo

 Data_Extraction: This folder has scripts forgetting the data from my local flash drive, running preprocessing and storing it into the repo. 

 Inference: This folder has scripts related to testing and training the models. 

 Vosk: local subfolder for the vosk asr model dependencies. 

 Important Notes: For the deep neural net you will have to have git lfs enabled to download it. Its correct size is 116 MB if it is less thant it is not the correct weights file. 