INSTRUCTIONS ON HOW TO TRAIN CONVOLUTIONAL NEURAL NETWORKS, RANDOM FORESTS AND PARTIAL LEAST SQUARES MODELS ON SFED DESCRIPTORS
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
##################

DEMO

There are only two files needed to train a model: a given SFED dataset (e.g. '298_gf_dataset_water_chloroform_carbontet.csv') and a model training script (e.g. 'cnn_kfold_cv_final_no_temp.py').


The model training scripts are set up to import the necessary libraries, read in the dataset, train the model over a given number of train-test splits, tune the model hyperparameters over five-fold cross-validation, and produce various statistics.

Two points must be edited in the model training script before a model can be trained:
1. To train on a specific dataset, you must edit the 'input_dataset' variable to load the SFED dataset (CNN scripts - line 73, PLS scripts - line 26, RF scripts - line 48). 

2. The 1D-RISM functional used to generate the SFED dataset must be specified. The functional will be noted in the dataset .csv file name (e.g. 'kh', 'hnc', 'gf'). 
   To specify the functional, change: 
   col[:5]=='X_w_' to col[:5]=='kh_w_' or col[:6]=='hnc_w_' or col[:5]=='gf_w_'

   For scripts that do not use temperature descriptors this can be edited in: CNN scripts - lines 283, 285, 287 & 641, 643, 645. 
                                                                              PLS scripts - lines 196, 198 & 416, 418.
                                                                              RF scripts - lines 239, 241 & 477, 479.

   For scripts that do use temperature descriptors this can be edited in: CNN scripts - lines 285, 289, 293 & 650, 654, 658. 
                                                                              PLS scripts - lines 196, 200 & 420, 424.
                                                                              RF scripts - lines 240, 244 & 482, 486.

To recreate the results we have in the paper, keep all other variables as they are. The number of resamples can be changed from 50 to another value if desired (e.g. line 81 in CNN scripts). 

From the working directory simply enter this from your chosen operating system terminal:
e.g. 'python cnn_kfold_cv_final_no_temp.py'
This will train the chosen algorithm, taking approximately 2 hours runtime (for the multi-temperature and multi-solvent datasets) with the following specs:
RAM: 38.4 GB (4.8GB/core)
CPU: 8 cores, 2.0 GHz/core

There is no random_seed set in the algorithms, so expect your results to be slightly different than those in the paper. This is because the 70%/30% train-test split is different every time, and therefore will produce slightly different test statistics during each run. However, the overall trends should be the same.

All output will be saved in the same directory that the training script and dataset are present in. Here is an explanation of the expected output:
'resample_*' folders: include .txt statistics for that resample as well as train/test/val .csv prediction data for the full resample split, as well as for the individual solvent predictions. Cross-validation subfolders for the given resample are also available with similar information.

'resample_pred_test/train/val_avg.csv': summary of predictions for each solute across all resamples.

'average_pred_over_resamples.csv': contains the average prediction for each solute over all resamples.

'avg_pred_over_resamples.txt': average test set predictions over all resamples, calculated from the individual solute predictions in 'average_pred_over_resamples.csv'.

'avg_pred_test_stats_chloroform/carbontet/water.txt': average test set predictions over all resamples for each individual solvent. Calculated by only taking the solvent specific solute predictions at each resample. Individual solvent predictions can be found in each 'resample_*/resample_*/individual_solvent_predictions' folder. 

'CV_avg_stats_against_resamples.txt': The average Cross-validation predictions over the 5-fold Cross-validation within each resample is recorded in the given 'resample_*' folders. The average Cross-validation predictions over all resample can be found in this file.

'resample_avg_stats.txt': The average predictions over all resamples.

'sd_test_stats.txt': The standard deviation of each statistic from all resample test set predictions.

'avg_pred_over_resamples_plot_with_pred_avg_stats.png': plot of the average prediction for each solute over all resamples with statistics from 'avg_pred_over_resamples.txt'.

'avg_pred_over_resamples_plot_with_resample_avg_stats.png': plot of the average prediction for each solute over all resamples with statistics from 'resample_avg_stats.txt'.


