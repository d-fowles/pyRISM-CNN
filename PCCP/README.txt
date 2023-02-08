INSTRUCTIONS ON HOW TO TRAIN CONVOLUTIONAL NEURAL NETWORK MODELS ON SFED DESCRIPTORS
######################################################################################
SYSTEM REQUIREMENTS

Software requirements:
All models were developed and trained on a Linux operating sytsem:
Linux: AlmaLinux release 8.5 (Arctic Sphynx)
##################

INSTALLATION GUIDE

All models require the Python programming language, and were trained on the following version of Python:
Python 3.8.8
This can be installed at https://www.python.org/downloads/release/python-388/

The model requires the following libraries to be installed (all modules can also be found in the model training scripts). The versions used are as follows:
math 0.0.1
numpy 1.20.1
pandas 1.3.4
os 3.10.7
matplotlib 3.3.4
glob 3.10.7
statistics 1.0.3.5
tensorflow 2.10.0
scikit-learn 1.1.2
csv 0.0.13
scipy 1.6.2

These libraries can be installed using 'pip', an example being:
'pip install matplotlib'
Once these libraries are installed, the model is ready to be trained.

pyRISM source code will be needed for any pyRISM 1D-RISM calculations and can be found at: https://github.com/2AUK/pyRISM

##################
CONTENTS OF REPOSITORY

'datasets': contains .csv files of SFED datasets produced using GAFF solute parameters and the GF free energy functional. The 'thermo_dataset_gf.csv' file contains solute data with experimental solvation enthalpy, entropy and free energy. The 'free_energy_neutral_ionised_dataset_gf.csv' file contains neutral and ionised solute data with experimental salvation free energy values.
'scripts': scripts for training CNN models can be found here. Three different models are provided - a single task CNN for predicting solvation free energy for neutral and ionised solutes, a single task CNN for predicting solvation enthalpy, entropy or free energy for neutral solutes, and a multi task CNN for predicting solvation enthalpy, entropy and free energy for neutral solutes. 
'toml_input_files': contains the solvent specific input files used by pyRISM for 1D-RISM calculations. 

##################

DEMO FOR TRAINING A MODEL

There are only two files needed to train a model: a given SFED dataset (e.g. 'free_energy_neutral_ionised_dataset_gf.csv') and a model training script (e.g. 'free_energy_neutral_ionised_single_task_cnn.py').


The model training scripts are set up to import the necessary libraries, read in the dataset, train the model over a given number of train-test splits, tune the model hyperparameters over five-fold cross-validation, and produce various statistics.

Two points must be edited in the model training script before a model can be trained:
1. To train on a specific dataset, you must edit the 'input_dataset' variable to load the SFED dataset. 

2. The 1D-RISM functional used to generate the SFED dataset must be specified. The functional will be noted in the dataset .csv file name (e.g. 'kh', 'hnc', 'gf'). 
   To specify the functional edit the 'sfed_type' variable. 

A third point must be edited for the single task CNN which can predict for solvation enthalpy, entropy or free energy. To train for a specific parameter, you must edit the 'param' variable (e.g. 'DH', 'TDS', 'DG').


To recreate the results we have in the paper, keep all other variables as they are. The number of resamples can be changed from 50 to another value if desired by editing 'no_resamples'. 

From the working directory simply enter this from your chosen operating system terminal:
e.g. 'python free_energy_neutral_ionised_single_task_cnn.py'
This will train the chosen algorithm, taking approximately 2 hours runtime (for the multi-temperature and multi-solvent dataset) with the following specs:
RAM: 38.4 GB (4.8GB/core)
CPU: 8 cores, 2.0 GHz/core

There is no random_seed set in the algorithms, so expect your results to be slightly different than those in the paper. This is because the 70%/30% train-test split is different every time, and therefore will produce slightly different test statistics during each run. However, the overall trends should be the same.

All output will be saved in the same directory that the training script and dataset are present in. Here is an explanation of the expected output:
'resample_*' folders: include .txt statistics for that resample as well as train/test/val .csv prediction data for the full resample split, as well as for the individual solvent predictions. Cross-validation subfolders for the given resample are also available with similar information.

##################