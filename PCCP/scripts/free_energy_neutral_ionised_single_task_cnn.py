import math
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import glob
import statistics as stats

import tensorflow as tf
from tensorflow.keras import layers, models, initializers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger

from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV

root_dir = os.getcwd()

# Define NN architecture:
# -----------------------

def build_model():
    model = models.Sequential()
    # Add layers in order:
    # Conv1D layers with pooling and normalisation
    model.add(layers.Conv1D(32,                 # Number of output filters
                     kernel_size=(3),    # x, y size of output filters  
                     strides=(2),        # x, y stride length of convolution (Default value = 1)
                     padding='valid',    # Add padding around image, 'valid' means no padding (Default)
                     activation='relu',  # Use ReLU activation function
                     input_shape=(160,1) # Shape of input data, in order x, y, channels
                    ))
    model.add(layers.MaxPooling1D((2))) 
    model.add(layers.BatchNormalization()) 
    model.add(layers.Conv1D(32,                 # Number of output filters
                     kernel_size=(3),    # x, y size of output filters  
                     strides=(2),        # x, y stride length of convolution (Default value = 1) 
                     padding='valid',    # Add padding around image, 'valid' means no padding (Default)
                     activation='relu'   # Use ReLU activation function
                    ))
    model.add(layers.MaxPooling1D((2))) 
    model.add(layers.BatchNormalization()) 
    model.add(layers.Conv1D(32,                 # Number of output filters
                     kernel_size=(3),    # x, y size of output filters  
                     strides=(2),        # x, y stride length of convolution (Default value = 1) 
                     padding='valid',    # Add padding around image, 'valid' means no padding (Default)
                     activation='relu'   # Use ReLU activation function
                    ))
    model.add(layers.MaxPooling1D((2))) 
    model.add(layers.BatchNormalization()) 
    model.add(layers.Flatten())

    # Output layer:
    model.add(layers.Dense(1))
    # Compile model:
    model.compile(loss = "mse", # Minimise mean squared error
                  optimizer =tf.keras.optimizers.Adam(lr=0.01),
                  metrics = ["mse"])
    return model

# Initialise the model and print model summary:
model = build_model()

### Prepare input data with a unique id for each molecule and its corresponding descr.
input_dataset = pd.read_csv(f'{root_dir}/free_energy_neutral_ionised_dataset_gf.csv')
input_dataset = input_dataset.sort_values('Mol')
input_dataset = input_dataset.reset_index()
input_dataset = input_dataset.drop(['index'], axis=1)
df_id = input_dataset.groupby(['Mol']).ngroup()
input_dataset.insert(0, 'id', df_id)

### Specify SFED type
sfed_type = 'gf' #kh, hnc or gf

### Start resampling loop
no_resamples = 50

# Set up variables to save all resamples statistics for each resample loop
r2_train_sum_resample = 0
rmsd_train_sum_resample = 0
bias_train_sum_resample = 0
sdep_train_sum_resample = 0

r2_val_sum_resample = 0
rmsd_val_sum_resample = 0
bias_val_sum_resample = 0
sdep_val_sum_resample = 0

r2_test_sum_resample = 0
rmsd_test_sum_resample = 0
bias_test_sum_resample = 0
sdep_test_sum_resample = 0

# Lists to save resamples predictions
all_preds_train_resample = []
all_preds_val_resample = []
all_preds_test_resample = []

for resample in range(1, no_resamples + 1):
    
    # Make directory for each resample loop and dump all related files into it
    try:
        os.mkdir(f'{root_dir}/resample_{resample}')
    except FileExistsError:
        pass

    # Shuffle and split dataset along unique id for each resample into train val and test sets
    input_dataset_shuffle = input_dataset.sample(frac=1).reset_index(drop=True)

    train_val_resample_frac = 0.7
    unique_resample_id = input_dataset_shuffle['id'].unique()
    train_val_resample_idx = input_dataset_shuffle['id'].isin(unique_resample_id[:round(train_val_resample_frac * len(unique_resample_id))])
    train_val_resample = input_dataset_shuffle.loc[train_val_resample_idx, :]
    test_resample = input_dataset_shuffle.loc[~train_val_resample_idx, :]

    # Prepare validation set from this resamples train val set
    val_resample_dataset_shuffle = train_val_resample.sample(frac=1).reset_index(drop=True)        

    val_resample_frac = 0.2
    unique_val_resample_id = val_resample_dataset_shuffle['id'].unique()
    val_resample_idx = val_resample_dataset_shuffle['id'].isin(unique_val_resample_id[:round(val_resample_frac * len(unique_val_resample_id))])
    val_resample = val_resample_dataset_shuffle.loc[val_resample_idx, :]
    train_resample = val_resample_dataset_shuffle.loc[~val_resample_idx, :]

    ### Start CV set up and loop
    no_folds = 5

    # Set up variables to save all CV fold statistics for this resample
    r2_train_sum_cv = 0
    rmsd_train_sum_cv = 0
    bias_train_sum_cv = 0
    sdep_train_sum_cv = 0

    r2_val_sum_cv = 0
    rmsd_val_sum_cv = 0
    bias_val_sum_cv = 0
    sdep_val_sum_cv = 0

    r2_test_sum_cv = 0
    rmsd_test_sum_cv= 0
    bias_test_sum_cv = 0
    sdep_test_sum_cv = 0

    # Lists to save CV fold predictions
    all_preds_train_cv = []
    all_preds_val_cv = []
    all_preds_test_cv = []

    # Reshuffle and split resample training set into five parts for CV
    fold_dataset_shuffle = train_resample.sample(frac=1).reset_index(drop=True)
    headers = fold_dataset_shuffle.columns

    mol_ids = fold_dataset_shuffle['id'].unique().tolist()
    fold_dataset_dict = {id: fold_dataset_shuffle.loc[fold_dataset_shuffle['id'] == id] for id in mol_ids}

    dict_items = fold_dataset_dict.items()
    split = int(len(dict_items) / no_folds)

    fold_1_list = list(dict_items)[:split]
    fold_1 = []
    for fold in fold_1_list:
        fold_stuff = fold[1]
        fold_stuff_pd = pd.concat([fold_stuff])
        fold_1.append(fold_stuff_pd)
      
    fold_1 = pd.concat(fold_1)
    other_4_folds = list(dict_items)[split:]
   
    fold_2_list = list(other_4_folds)[:split]
    fold_2 = []
    for fold in fold_2_list:
        fold_stuff = fold[1]
        fold_stuff_pd = pd.concat([fold_stuff])
        fold_2.append(fold_stuff_pd)
      
    fold_2 = pd.concat(fold_2)
    other_3_folds = list(other_4_folds)[split:]

    fold_3_list = list(other_3_folds)[:split]
    fold_3 = []
    for fold in fold_3_list:
        fold_stuff = fold[1]
        fold_stuff_pd = pd.concat([fold_stuff])
        fold_3.append(fold_stuff_pd)
      
    fold_3 = pd.concat(fold_3)
    other_2_folds = list(other_3_folds)[split:]

    fold_4_list = list(other_2_folds)[:split]
    fold_4 = []
    for fold in fold_4_list:
        fold_stuff = fold[1]
        fold_stuff_pd = pd.concat([fold_stuff])
        fold_4.append(fold_stuff_pd)
      
    fold_4 = pd.concat(fold_4)
    other_1_folds = list(other_2_folds)[split:]

    fold_5_list = list(other_1_folds)[:split]
    fold_5 = []
    for fold in fold_5_list:
        fold_stuff = fold[1]
        fold_stuff_pd = pd.concat([fold_stuff])
        fold_5.append(fold_stuff_pd)
      
    fold_5 = pd.concat(fold_5)

    fold_list_loop = [fold_1, fold_2, fold_3, fold_4, fold_5]

    # Start of CV loop
    
    for i, fold in enumerate(fold_list_loop):

        try:
            os.mkdir(f'{root_dir}/resample_{resample}/fold_{i}')
        except FileExistsError:
            pass

        os.chdir(f'{root_dir}/resample_{resample}/fold_{i}')

        fold_copy = fold_list_loop.copy()
        _ = fold_copy.pop(i)
        other_folds_list = fold_copy
        other_folds = pd.concat(other_folds_list)

        # Each fold has already been preshuffled for this resample loop
        # Only the validation set needs to be prepared, along with X and y variables
        train_val_fold = other_folds
        test_fold = fold

        ### Prepare validation set from this folds train val set
        val_fold_dataset_shuffle = train_val_fold.sample(frac=1).reset_index(drop=True)        

        val_fold_frac = 0.2
        unique_val_fold_id = val_fold_dataset_shuffle['id'].unique()
        df_val_fold_idx = val_fold_dataset_shuffle['id'].isin(unique_val_fold_id[:round(val_fold_frac * len(unique_val_fold_id))])
        val_fold = val_fold_dataset_shuffle.loc[df_val_fold_idx, :]
        train_fold = val_fold_dataset_shuffle.loc[~df_val_fold_idx, :]

        ### Prepare data into X and y for each fold
        # Take exp HFE as y input in dataframe format
        y_train_fold = train_fold[['HFE']].copy()

        y_train_fold_output = pd.DataFrame()
        y_train_fold_output['Mol'] = train_fold['Mol']
        y_train_fold_output['Temp'] = train_fold['Temp']
        y_train_fold_output['HFE'] = train_fold['HFE']
        y_train_fold_output.to_csv(f'resample_{resample}_fold_{i}_train_plot.csv', index=False)

        y_val_fold = val_fold[['HFE']].copy()

        y_val_fold_output = pd.DataFrame()
        y_val_fold_output['Mol'] = val_fold['Mol']
        y_val_fold_output['Temp'] = val_fold['Temp']
        y_val_fold_output['HFE'] = val_fold['HFE']
        y_val_fold_output.to_csv(f'resample_{resample}_fold_{i}_val_plot.csv', index=False)

        y_test_fold = test_fold[['HFE']].copy()

        y_test_fold_output = pd.DataFrame()
        y_test_fold_output['Mol'] = test_fold['Mol']
        y_test_fold_output['Temp'] = test_fold['Temp']
        y_test_fold_output['HFE'] = test_fold['HFE']
        y_test_fold_output.to_csv(f'resample_{resample}_fold_{i}_test_plot.csv', index=False)

        # Convert y data into required input shape
        y_train_cv = y_train_fold.to_numpy()
        y_train_cv = y_train_cv.reshape(y_train_cv.shape[0])

        y_val_cv = y_val_fold.to_numpy()
        y_val_cv = y_val_cv.reshape(y_val_cv.shape[0])

        y_test_cv = y_test_fold.to_numpy()
        y_test_cv = y_test_cv.reshape(y_test_cv.shape[0])

        # Take descr. columns as X data in dataframe format
        # Doesnt use temp as a descr, Select all columns beginning with 'X_w_' as the X data
        if sfed_type == 'hnc':
            X_descr_train = train_fold[[col for col in train_fold.columns if col[:6]==f'{sfed_type}_w_']]

            X_descr_val = val_fold[[col for col in val_fold.columns if col[:6]==f'{sfed_type}_w_']]

            X_descr_test = test_fold[[col for col in test_fold.columns if col[:6]==f'{sfed_type}_w_']]

        else:
            X_descr_train = train_fold[[col for col in train_fold.columns if col[:5]==f'{sfed_type}_w_']]
            
            X_descr_val = val_fold[[col for col in val_fold.columns if col[:5]==f'{sfed_type}_w_']]
            
            X_descr_test = test_fold[[col for col in test_fold.columns if col[:5]==f'{sfed_type}_w_']]

        # Scale train and use that to scale val and test
        # No temp
        #means = X_descr_train.mean(axis=0)
        #sds = X_descr_train.std(axis=0)

        #X_descr_train_scaled = (X_descr_train - means) / sds
        #X_descr_val_scaled = (X_descr_val - means) / sds
        #X_descr_test_scaled = (X_descr_test - means) / sds

        # Convert X data into required input shape

        # No temp
        X_train_cv = X_descr_train.to_numpy()
        X_train_cv = X_train_cv.reshape(X_train_cv.shape[0], X_train_cv.shape[1],1)

        X_val_cv = X_descr_val.to_numpy()
        X_val_cv = X_val_cv.reshape(X_val_cv.shape[0], X_val_cv.shape[1],1)

        X_test_cv = X_descr_test.to_numpy()
        X_test_cv = X_test_cv.reshape(X_test_cv.shape[0], X_test_cv.shape[1],1)

        # simple early stopping for model
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        csv_logger = CSVLogger(f"model_history_log_resample_{resample}_fold_{i}.csv", append=True)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'training_{i}/cp.ckpt',
                                                         save_weights_only=True,
                                                         verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tensorboard_logs/", histogram_freq=1)

        # Train the model:
        # ----------------
        # Fit the model:
        epochs = 60
        history = model.fit(X_train_cv, y_train_cv,
                            epochs = epochs,
                            verbose = 1,
                            validation_data =(X_val_cv, y_val_cv),
                            callbacks=[es, csv_logger, cp_callback, tensorboard_callback])
        
        # Get weights
        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights()
        np.savetxt('weights.csv' , weights , fmt='%s', delimiter=',')

        with open('weights.txt', 'w') as f:

            for name, weight in zip(names, weights):
                print(name, weight.shape, weight, file=f)
        
        # Make predictions on train, val and test set using trained model:
        y_pred_train_cv = model.predict(X_train_cv)
        y_pred_val_cv = model.predict(X_val_cv)
        y_pred_test_cv = model.predict(X_test_cv)

        # Assess performace of model based on predictions:

        # Reduce dimensionality of predcition arrays
        y_pred_train_cv = y_pred_train_cv.squeeze()
        y_pred_val_cv = y_pred_val_cv.squeeze()
        y_pred_test_cv = y_pred_test_cv.squeeze()

        # Coefficient of determination
        r2_train_cv = r2_score(y_train_cv, y_pred_train_cv)
        r2_val_cv = r2_score(y_val_cv, y_pred_val_cv)
        r2_test_cv = r2_score(y_test_cv, y_pred_test_cv)
        # Root mean squared error
        rmsd_train_cv = (mean_squared_error(y_train_cv, y_pred_train_cv))**0.5
        rmsd_val_cv = (mean_squared_error(y_val_cv, y_pred_val_cv))**0.5
        rmsd_test_cv = (mean_squared_error(y_test_cv, y_pred_test_cv))**0.5
        # Bias
        bias_train_cv = np.mean(y_pred_train_cv - y_train_cv)
        bias_val_cv = np.mean(y_pred_val_cv - y_val_cv)
        bias_test_cv = np.mean(y_pred_test_cv - y_test_cv)
        # Standard deviation of the error of prediction
        sdep_train_cv = (np.mean((y_pred_train_cv - y_train_cv - bias_train_cv)**2))**0.5
        sdep_val_cv = (np.mean((y_pred_val_cv - y_val_cv - bias_val_cv)**2))**0.5
        sdep_test_cv = (np.mean(((y_pred_test_cv - y_test_cv) - bias_test_cv)**2))**0.5
        
        # Save stats for each individual fold 
        r2_train_ind_cv = '{:.3f}'.format(r2_train_cv)
        rmsd_train_ind_cv = '{:.3f}'.format(rmsd_train_cv)
        bias_train_ind_cv = '{:.3f}'.format(bias_train_cv)
        sdep_train_ind_cv = '{:.3f}'.format(sdep_train_cv)

        r2_val_ind_cv = '{:.3f}'.format(r2_val_cv)
        rmsd_val_ind_cv = '{:.3f}'.format(rmsd_val_cv)
        bias_val_ind_cv = '{:.3f}'.format(bias_val_cv)
        sdep_val_ind_cv = '{:.3f}'.format(sdep_val_cv)

        r2_test_ind_cv = '{:.3f}'.format(r2_test_cv)
        rmsd_test_ind_cv = '{:.3f}'.format(rmsd_test_cv)
        bias_test_ind_cv = '{:.3f}'.format(bias_test_cv)
        sdep_test_ind_cv = '{:.3f}'.format(sdep_test_cv)

        results_file = open(f'fold_{i}_CV_stats.txt', 'w')
        results_file.write(f'r2_train: {r2_train_ind_cv}')
        results_file.write('\n')
        results_file.write(f'rmsd_train: {rmsd_train_ind_cv}')
        results_file.write('\n')
        results_file.write(f'bias_train: {bias_train_ind_cv}')
        results_file.write('\n')
        results_file.write(f'sdep_train: {sdep_train_ind_cv}')
        results_file.write('\n')
        results_file.write('\n')
        results_file.write(f'r2_val: {r2_val_ind_cv}')
        results_file.write('\n')
        results_file.write(f'rmsd_val: {rmsd_val_ind_cv}')
        results_file.write('\n')
        results_file.write(f'bias_val: {bias_val_ind_cv}')
        results_file.write('\n')
        results_file.write(f'sdep_val: {sdep_val_ind_cv}')
        results_file.write('\n')
        results_file.write('\n')
        results_file.write(f'r2_test: {r2_test_ind_cv}')
        results_file.write('\n')
        results_file.write(f'rmsd_test: {rmsd_test_ind_cv}')
        results_file.write('\n')
        results_file.write(f'bias_test: {bias_test_ind_cv}')
        results_file.write('\n')
        results_file.write(f'sdep_test: {sdep_test_ind_cv}')
        results_file.close()

        # Save running sum of results:
        r2_train_sum_cv += r2_train_cv
        rmsd_train_sum_cv += rmsd_train_cv
        bias_train_sum_cv += bias_train_cv
        sdep_train_sum_cv += sdep_train_cv
 
        r2_val_sum_cv += r2_val_cv
        rmsd_val_sum_cv += rmsd_val_cv
        bias_val_sum_cv += bias_val_cv
        sdep_val_sum_cv += sdep_val_cv

        r2_test_sum_cv += r2_test_cv
        rmsd_test_sum_cv += rmsd_test_cv
        bias_test_sum_cv += bias_test_cv
        sdep_test_sum_cv += sdep_test_cv

        # Save individual predictions for fold in dataframes
        y_pred_train_cv = y_pred_train_cv.reshape(y_pred_train_cv.shape[0])
        y_pred_train_cv = pd.DataFrame(y_pred_train_cv, columns=[f'fold_{i}'])
        y_pred_train_cv_labeled = pd.concat([train_fold['Mol'], train_fold['Temp'], train_fold['HFE'], y_pred_train_cv], axis=1)
        y_pred_train_cv_labeled[''] = ''
        all_preds_train_cv.append(y_pred_train_cv_labeled)

        y_pred_val_cv = y_pred_val_cv.reshape(y_pred_val_cv.shape[0])
        y_pred_val_cv = pd.DataFrame(y_pred_val_cv, columns=[f'fold_{i}'])
        y_pred_val_cv_labeled = pd.concat([val_fold['Mol'], val_fold['Temp'], val_fold['HFE'], y_pred_val_cv], axis=1)
        y_pred_val_cv_labeled[''] = ''
        all_preds_val_cv.append(y_pred_val_cv_labeled)

        y_pred_test_cv = y_pred_test_cv.reshape(y_pred_test_cv.shape[0])
        y_pred_test_cv = pd.DataFrame(y_pred_test_cv, columns=[f'fold_{i}'])
        y_pred_test_cv_labeled = pd.concat([test_fold['Mol'], test_fold['Temp'], test_fold['HFE'], y_pred_test_cv], axis=1)
        y_pred_test_cv_labeled[''] = ''
        all_preds_test_cv.append(y_pred_test_cv_labeled)

    # Average results over CV folds:
    r2_train_av_cv = r2_train_sum_cv/no_folds
    rmsd_train_av_cv = rmsd_train_sum_cv/no_folds
    bias_train_av_cv = bias_train_sum_cv/no_folds
    sdep_train_av_cv = sdep_train_sum_cv/no_folds

    r2_train_av_cv = '{:.3f}'.format(r2_train_av_cv)
    rmsd_train_av_cv = '{:.3f}'.format(rmsd_train_av_cv)
    bias_train_av_cv = '{:.3f}'.format(bias_train_av_cv)
    sdep_train_av_cv = '{:.3f}'.format(sdep_train_av_cv)

    r2_val_av_cv = r2_val_sum_cv/no_folds
    rmsd_val_av_cv = rmsd_val_sum_cv/no_folds
    bias_val_av_cv = bias_val_sum_cv/no_folds
    sdep_val_av_cv = sdep_val_sum_cv/no_folds

    r2_val_av_cv = '{:.3f}'.format(r2_val_av_cv)
    rmsd_val_av_cv = '{:.3f}'.format(rmsd_val_av_cv)
    bias_val_av_cv = '{:.3f}'.format(bias_val_av_cv)
    sdep_val_av_cv = '{:.3f}'.format(sdep_val_av_cv)

    r2_test_av_cv = r2_test_sum_cv/no_folds
    rmsd_test_av_cv = rmsd_test_sum_cv/no_folds
    bias_test_av_cv = bias_test_sum_cv/no_folds
    sdep_test_av_cv = sdep_test_sum_cv/no_folds

    r2_test_av_cv = '{:.3f}'.format(r2_test_av_cv)
    rmsd_test_av_cv = '{:.3f}'.format(rmsd_test_av_cv)
    bias_test_av_cv = '{:.3f}'.format(bias_test_av_cv)
    sdep_test_av_cv = '{:.3f}'.format(sdep_test_av_cv)

    # Write average results to a file:
    results_file = open(f'{root_dir}/resample_{resample}/resample_{resample}_CV_avg_stats.txt', 'w')
    results_file.write(f'r2_train: {r2_train_av_cv}')
    results_file.write('\n')
    results_file.write(f'rmsd_train: {rmsd_train_av_cv}')
    results_file.write('\n')
    results_file.write(f'bias_train: {bias_train_av_cv}')
    results_file.write('\n')
    results_file.write(f'sdep_train: {sdep_train_av_cv}')
    results_file.write('\n')
    results_file.write('\n')
    results_file.write(f'r2_val: {r2_val_av_cv}')
    results_file.write('\n')
    results_file.write(f'rmsd_val: {rmsd_val_av_cv}')
    results_file.write('\n')
    results_file.write(f'bias_val: {bias_val_av_cv}')
    results_file.write('\n')
    results_file.write(f'sdep_val: {sdep_val_av_cv}')
    results_file.write('\n')
    results_file.write('\n')
    results_file.write(f'r2_test: {r2_test_av_cv}')
    results_file.write('\n')
    results_file.write(f'rmsd_test: {rmsd_test_av_cv}')
    results_file.write('\n')
    results_file.write(f'bias_test: {bias_test_av_cv}')
    results_file.write('\n')
    results_file.write(f'sdep_test: {sdep_test_av_cv}')
    results_file.close()

    # Save all individual predictions to train file:
    all_preds_train_cv = pd.concat(all_preds_train_cv, axis=1)
    all_preds_train_cv['Mol'].replace('', np.nan, inplace=True)
    all_preds_train_cv = all_preds_train_cv.apply(lambda x: pd.Series(x.dropna().values))
    all_preds_train_cv.to_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_train_CV_avg.csv')

    all_preds_val_cv = pd.concat(all_preds_val_cv, axis=1)
    all_preds_val_cv['Mol'].replace('', np.nan, inplace=True)
    all_preds_val_cv = all_preds_val_cv.apply(lambda x: pd.Series(x.dropna().values))
    all_preds_val_cv.to_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_val_CV_avg.csv')

    all_preds_test_cv = pd.concat(all_preds_test_cv, axis=1)
    all_preds_test_cv['Mol'].replace('', np.nan, inplace=True)
    all_preds_test_cv = all_preds_test_cv.apply(lambda x: pd.Series(x.dropna().values))
    all_preds_test_cv.to_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_test_CV_avg.csv')
    
    # Output for each individual fold

    for num in range(0, 5):

        os.chdir(f'{root_dir}/resample_{resample}/fold_{num}')

        ### Train
        df = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_train_CV_avg.csv', index_col=False)
        df2 = pd.read_csv(f'resample_{resample}_fold_{num}_train_plot.csv', index_col=False)
        df2.columns = ['Mol', 'Temp', 'y']
        df2['y_pred'] = df[f'fold_{num}']
        df2.to_csv(f'resample_{resample}_fold_{num}_train_plot.csv', index=False)

        ### Val
        df = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_val_CV_avg.csv', index_col=False)
        df2 = pd.read_csv(f'resample_{resample}_fold_{num}_val_plot.csv', index_col=False)
        df2.columns = ['Mol', 'Temp', 'y']
        df2['y_pred'] = df[f'fold_{num}']
        df2.to_csv(f'resample_{resample}_fold_{num}_val_plot.csv', index=False)

        ### Test
        df = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_test_CV_avg.csv', index_col=False)
        df2 = pd.read_csv(f'resample_{resample}_fold_{num}_test_plot.csv', index_col=False)
        df2.columns = ['Mol', 'Temp', 'y']
        df2['y_pred'] = df[f'fold_{num}']
        df2.to_csv(f'resample_{resample}_fold_{num}_test_plot.csv', index=False)

        ### Individual solvent predictions
        try:
            os.mkdir('individual_solvent_predictions')
            os.mkdir('individual_solvent_predictions/train')
            os.mkdir('individual_solvent_predictions/val')
            os.mkdir('individual_solvent_predictions/test')
        except FileExistsError:
            pass

        sets = ['train', 'val', 'test']

        for set in sets:

            df_charged = pd.read_csv(f'resample_{resample}_fold_{num}_{set}_plot.csv', index_col=False)
            df_water = pd.read_csv(f'resample_{resample}_fold_{num}_{set}_plot.csv', index_col=False)
            df_methanol = pd.read_csv(f'resample_{resample}_fold_{num}_{set}_plot.csv', index_col=False)
            df_chloroform = pd.read_csv(f'resample_{resample}_fold_{num}_{set}_plot.csv', index_col=False)
            df_carbontet = pd.read_csv(f'resample_{resample}_fold_{num}_{set}_plot.csv', index_col=False)
                    
            for mol in df_charged['Mol']:
                if 'ion' not in mol:
                    df_charged.drop(df_charged.loc[df_charged['Mol']==mol].index, inplace=True)

            df_charged.reset_index(inplace=True)
            df_charged.drop(columns = ['index'], inplace=True)
            df_charged.to_csv(f'individual_solvent_predictions/{set}/resample_{resample}_fold_{num}_{set}_charged_pred.csv')
                    
            for mol in df_methanol['Mol']:
                if 'methanol' not in mol:
                    df_methanol.drop(df_methanol.loc[df_methanol['Mol']==mol].index, inplace=True)
                if 'ion' in mol:
                    df_methanol.drop(df_methanol.loc[df_methanol['Mol']==mol].index, inplace=True)

            df_methanol.reset_index(inplace=True)
            df_methanol.drop(columns = ['index'], inplace=True)
            df_methanol.to_csv(f'individual_solvent_predictions/{set}/resample_{resample}_fold_{num}_{set}_methanol_pred.csv')

            for mol in df_chloroform['Mol']:
                if 'chloroform' not in mol:
                    df_chloroform.drop(df_chloroform.loc[df_chloroform['Mol']==mol].index, inplace=True)

            df_chloroform.reset_index(inplace=True)
            df_chloroform.drop(columns = ['index'], inplace=True)
            df_chloroform.to_csv(f'individual_solvent_predictions/{set}/resample_{resample}_fold_{num}_{set}_chloroform_pred.csv')

            for mol in df_carbontet['Mol']:
                if 'carbontet' not in mol:
                    df_carbontet.drop(df_carbontet.loc[df_carbontet['Mol']==mol].index, inplace=True)

            df_carbontet.reset_index(inplace=True)
            df_carbontet.drop(columns = ['index'], inplace=True)
            df_carbontet.to_csv(f'individual_solvent_predictions/{set}/resample_{resample}_fold_{num}_{set}_carbontet_pred.csv')

            for mol in df_water['Mol']:
                if 'ion' in mol:
                    df_water.drop(df_water.loc[df_water['Mol']==mol].index, inplace=True)
                if 'methanol' in mol:
                    df_water.drop(df_water.loc[df_water['Mol']==mol].index, inplace=True)
                if 'chloroform' in mol:
                    df_water.drop(df_water.loc[df_water['Mol']==mol].index, inplace=True)
                if 'carbontet' in mol:
                    df_water.drop(df_water.loc[df_water['Mol']==mol].index, inplace=True)

            df_water.reset_index(inplace=True)
            df_water.drop(columns = ['index'], inplace=True)
            df_water.to_csv(f'individual_solvent_predictions/{set}/resample_{resample}_fold_{num}_{set}_water_pred.csv')

    ### End of CV loop for this resample
    # Start data prep and model training for this resample 
       
    # Make a resample folder within each resample directory  
    try:
        os.mkdir(f'{root_dir}/resample_{resample}/resample_{resample}')
    except FileExistsError:
        pass

    os.chdir(f'{root_dir}/resample_{resample}/resample_{resample}')

    ### Prepare data into X and y for each resample
    # Take exp HFE as y input in dataframe format
    y_train_resample = train_resample[['HFE']].copy()

    y_train_resample_output = pd.DataFrame()
    y_train_resample_output['Mol'] = train_resample['Mol']
    y_train_resample_output['Temp'] = train_resample['Temp']
    y_train_resample_output['HFE'] = train_resample['HFE']
    y_train_resample_output.to_csv(f'resample_{resample}_train_plot.csv', index=False)

    y_val_resample = val_resample[['HFE']].copy()

    y_val_resample_output = pd.DataFrame()
    y_val_resample_output['Mol'] = val_resample['Mol']
    y_val_resample_output['Temp'] = val_resample['Temp']
    y_val_resample_output['HFE'] = val_resample['HFE']
    y_val_resample_output.to_csv(f'resample_{resample}_val_plot.csv', index=False)

    y_test_resample = test_resample[['HFE']].copy()

    y_test_resample_output = pd.DataFrame()
    y_test_resample_output['Mol'] = test_resample['Mol']
    y_test_resample_output['Temp'] = test_resample['Temp']
    y_test_resample_output['HFE'] = test_resample['HFE']
    y_test_resample_output.to_csv(f'resample_{resample}_test_plot.csv', index=False)

    # Convert y data into required input shape
    y_train_resample = y_train_resample.to_numpy()
    y_train_resample = y_train_resample.reshape(y_train_resample.shape[0])

    y_val_resample = y_val_resample.to_numpy()
    y_val_resample = y_val_resample.reshape(y_val_resample.shape[0])

    y_test_resample = y_test_resample.to_numpy()
    y_test_resample = y_test_resample.reshape(y_test_resample.shape[0])

    # Take descr. columns as X data in dataframe format
    # Doesnt use temp as a descr, Select all columns beginning with 'X_w_' as the X data
    if sfed_type == 'hnc':
        X_descr_train = train_resample[[col for col in train_resample.columns if col[:6]==f'{sfed_type}_w_']]

        X_descr_val = val_resample[[col for col in val_resample.columns if col[:6]==f'{sfed_type}_w_']]

        X_descr_test = test_resample[[col for col in test_resample.columns if col[:6]==f'{sfed_type}_w_']]

    else:
        X_descr_train = train_resample[[col for col in train_resample.columns if col[:5]==f'{sfed_type}_w_']]
        
        X_descr_val = val_resample[[col for col in val_resample.columns if col[:5]==f'{sfed_type}_w_']]
        
        X_descr_test = test_resample[[col for col in test_resample.columns if col[:5]==f'{sfed_type}_w_']]

    # Scale train and use that to scale val and test
    # No temp
    means = X_descr_train.mean(axis=0)
    sds = X_descr_train.std(axis=0)

    X_descr_train_scaled = (X_descr_train - means) / sds
    X_descr_val_scaled = (X_descr_val - means) / sds
    X_descr_test_scaled = (X_descr_test - means) / sds

    # Convert X data into required input shape
    # No temp
    X_train_resample = X_descr_train_scaled.to_numpy()
    X_train_resample = X_train_resample.reshape(X_train_resample.shape[0], X_train_resample.shape[1],1)

    X_val_resample = X_descr_val_scaled.to_numpy()
    X_val_resample = X_val_resample.reshape(X_val_resample.shape[0], X_val_resample.shape[1],1)

    X_test_resample = X_descr_test_scaled.to_numpy()
    X_test_resample = X_test_resample.reshape(X_test_resample.shape[0], X_test_resample.shape[1],1)


    # simple early stopping for model
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
    csv_logger = CSVLogger(f"model_history_log_resample_{resample}.csv", append=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'training_{resample}/cp.ckpt',
                                                     save_weights_only=True,
                                                     verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tensorboard_logs/", histogram_freq=1)

    # Train the model:
    # ----------------
    # Fit the model:
    epochs = 60
    history = model.fit(X_train_resample, y_train_resample,
                        epochs = epochs,
                        verbose = 1,
                        validation_data =(X_val_resample, y_val_resample),
                        callbacks=[es, csv_logger, cp_callback, tensorboard_callback])
        
    # Get weights
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    np.savetxt('weights.csv' , weights , fmt='%s', delimiter=',')

    with open('weights.txt', 'w') as f:

        for name, weight in zip(names, weights):
            print(name, weight.shape, weight, file=f)

    # Save Model
    model.save('saved_model/my_model')

    # Make predictions on train, val and test set using trained model:
    y_pred_train_resample = model.predict(X_train_resample)
    y_pred_val_resample = model.predict(X_val_resample)
    y_pred_test_resample = model.predict(X_test_resample)

    # Assess performace of model based on predictions:

    # Reduce dimensionality of predcition arrays
    y_pred_train_resample = y_pred_train_resample.squeeze()
    y_pred_val_resample = y_pred_val_resample.squeeze()
    y_pred_test_resample = y_pred_test_resample.squeeze()

    # Coefficient of determination
    r2_train_resample = r2_score(y_train_resample, y_pred_train_resample)
    r2_val_resample = r2_score(y_val_resample, y_pred_val_resample)
    r2_test_resample = r2_score(y_test_resample, y_pred_test_resample)
    # Root mean squared error
    rmsd_train_resample = (mean_squared_error(y_train_resample, y_pred_train_resample))**0.5
    rmsd_val_resample = (mean_squared_error(y_val_resample, y_pred_val_resample))**0.5
    rmsd_test_resample = (mean_squared_error(y_test_resample, y_pred_test_resample))**0.5
    # Bias
    bias_train_resample = np.mean(y_pred_train_resample - y_train_resample)
    bias_val_resample = np.mean(y_pred_val_resample - y_val_resample)
    bias_test_resample = np.mean(y_pred_test_resample - y_test_resample)
    # Standard deviation of the error of prediction
    sdep_train_resample = (np.mean((y_pred_train_resample - y_train_resample - bias_train_resample)**2))**0.5
    sdep_val_resample = (np.mean((y_pred_val_resample - y_val_resample - bias_val_resample)**2))**0.5
    sdep_test_resample = (np.mean((y_pred_test_resample - y_test_resample - bias_test_resample)**2))**0.5

    # Save running sum of results:
    r2_train_sum_resample += r2_train_resample
    rmsd_train_sum_resample += rmsd_train_resample
    bias_train_sum_resample += bias_train_resample
    sdep_train_sum_resample += sdep_train_resample

    r2_val_sum_resample += r2_val_resample
    rmsd_val_sum_resample += rmsd_val_resample
    bias_val_sum_resample += bias_val_resample
    sdep_val_sum_resample += sdep_val_resample

    r2_test_sum_resample += r2_test_resample
    rmsd_test_sum_resample += rmsd_test_resample
    bias_test_sum_resample += bias_test_resample
    sdep_test_sum_resample += sdep_test_resample

    # Save individual predictions for resample in dataframes
    y_pred_train_resample = y_pred_train_resample.reshape(y_pred_train_resample.shape[0])
    y_pred_train_resample = pd.DataFrame(y_pred_train_resample, columns=[f'resample_{resample}'])
    y_pred_train_resample_labeled = pd.concat([train_resample['Mol'], train_resample['Temp'], train_resample['HFE'], y_pred_train_resample], axis=1)
    y_pred_train_resample_labeled[''] = ''
    all_preds_train_resample.append(y_pred_train_resample_labeled)

    y_pred_val_resample = y_pred_val_resample.reshape(y_pred_val_resample.shape[0])
    y_pred_val_resample = pd.DataFrame(y_pred_val_resample, columns=[f'resample_{resample}'])
    y_pred_val_resample_labeled = pd.concat([val_resample['Mol'], val_resample['Temp'], val_resample['HFE'], y_pred_val_resample], axis=1)
    y_pred_val_resample_labeled[''] = ''
    all_preds_val_resample.append(y_pred_val_resample_labeled)

    y_pred_test_resample = y_pred_test_resample.reshape(y_pred_test_resample.shape[0])
    y_pred_test_resample = pd.DataFrame(y_pred_test_resample, columns=[f'resample_{resample}'])
    y_pred_test_resample_labeled = pd.concat([test_resample['Mol'], test_resample['Temp'], test_resample['HFE'], y_pred_test_resample], axis=1)
    y_pred_test_resample_labeled[''] = ''
    all_preds_test_resample.append(y_pred_test_resample_labeled)

# Average results over resamples:
r2_train_av_resample = r2_train_sum_resample/no_resamples
rmsd_train_av_resample = rmsd_train_sum_resample/no_resamples
bias_train_av_resample = bias_train_sum_resample/no_resamples
sdep_train_av_resample = sdep_train_sum_resample/no_resamples

r2_train_av_resample = '{:.3f}'.format(r2_train_av_resample)
rmsd_train_av_resample = '{:.3f}'.format(rmsd_train_av_resample)
bias_train_av_resample = '{:.3f}'.format(bias_train_av_resample)
sdep_train_av_resample = '{:.3f}'.format(sdep_train_av_resample)

r2_val_av_resample = r2_val_sum_resample/no_resamples
rmsd_val_av_resample = rmsd_val_sum_resample/no_resamples
bias_val_av_resample = bias_val_sum_resample/no_resamples
sdep_val_av_resample = sdep_val_sum_resample/no_resamples

r2_val_av_resample = '{:.3f}'.format(r2_val_av_resample)
rmsd_val_av_resample = '{:.3f}'.format(rmsd_val_av_resample)
bias_val_av_resample = '{:.3f}'.format(bias_val_av_resample)
sdep_val_av_resample = '{:.3f}'.format(sdep_val_av_resample)

r2_test_av_resample = r2_test_sum_resample/no_resamples
rmsd_test_av_resample = rmsd_test_sum_resample/no_resamples
bias_test_av_resample = bias_test_sum_resample/no_resamples
sdep_test_av_resample = sdep_test_sum_resample/no_resamples

r2_test_av_resample = '{:.3f}'.format(r2_test_av_resample)
rmsd_test_av_resample = '{:.3f}'.format(rmsd_test_av_resample)
bias_test_av_resample = '{:.3f}'.format(bias_test_av_resample)
sdep_test_av_resample = '{:.3f}'.format(sdep_test_av_resample)

# Write average results to a file:
results_file = open(f'{root_dir}/avg_resample_stats.txt', 'w')
results_file.write(f'r2_train: {r2_train_av_resample}')
results_file.write('\n')
results_file.write(f'rmsd_train: {rmsd_train_av_resample}')
results_file.write('\n')
results_file.write(f'bias_train: {bias_train_av_resample}')
results_file.write('\n')
results_file.write(f'sdep_train: {sdep_train_av_resample}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_val: {r2_val_av_resample}')
results_file.write('\n')
results_file.write(f'rmsd_val: {rmsd_val_av_resample}')
results_file.write('\n')
results_file.write(f'bias_val: {bias_val_av_resample}')
results_file.write('\n')
results_file.write(f'sdep_val: {sdep_val_av_resample}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_test: {r2_test_av_resample}')
results_file.write('\n')
results_file.write(f'rmsd_test: {rmsd_test_av_resample}')
results_file.write('\n')
results_file.write(f'bias_test: {bias_test_av_resample}')
results_file.write('\n')
results_file.write(f'sdep_test: {sdep_test_av_resample}')
results_file.close()

# Save all individual predictions to train file:
all_preds_train_resample = pd.concat(all_preds_train_resample, axis=1)
all_preds_train_resample['Mol'].replace('', np.nan, inplace=True)
all_preds_train_resample = all_preds_train_resample.apply(lambda x: pd.Series(x.dropna().values))
all_preds_train_resample.to_csv(f'{root_dir}/resample_pred_train_avg.csv')

all_preds_val_resample = pd.concat(all_preds_val_resample, axis=1)
all_preds_val_resample['Mol'].replace('', np.nan, inplace=True)
all_preds_val_resample = all_preds_val_resample.apply(lambda x: pd.Series(x.dropna().values))
all_preds_val_resample.to_csv(f'{root_dir}/resample_pred_val_avg.csv')

all_preds_test_resample = pd.concat(all_preds_test_resample, axis=1)
all_preds_test_resample['Mol'].replace('', np.nan, inplace=True)
all_preds_test_resample = all_preds_test_resample.apply(lambda x: pd.Series(x.dropna().values))
all_preds_test_resample.to_csv(f'{root_dir}/resample_pred_test_avg.csv')

# Scatter plots for each individual resample

for resample in range(1, no_resamples + 1):

    os.chdir(f'{root_dir}/resample_{resample}/resample_{resample}')

    ### Train
    df = pd.read_csv(f'{root_dir}/resample_pred_train_avg.csv', index_col=False)
    df2 = pd.read_csv(f'resample_{resample}_train_plot.csv', index_col=False)
    df2.columns = ['Mol', 'Temp', 'y']
    df2['y_pred'] = df[f'resample_{resample}']
    df2.to_csv(f'resample_{resample}_train_plot.csv', index=False)

    np_y = df2['y'].to_numpy()
    np_y_pred = df2['y_pred'].to_numpy()

    r2 = r2_score(np_y, np_y_pred)
    rmsd = (mean_squared_error(np_y, np_y_pred))**0.5
    bias = np.mean(np_y_pred - np_y)
    sdep = (np.mean((np_y_pred - np_y - bias)**2))**0.5

    r2 = '{:.3f}'.format(r2)
    rmsd = '{:.3f}'.format(rmsd)
    bias = '{:.3f}'.format(bias)
    sdep = '{:.3f}'.format(sdep)

    results_file = open(f'resample_{resample}_stats.txt', 'w')
    results_file.write(f'r2_train: {r2}')
    results_file.write('\n')
    results_file.write(f'rmsd_train: {rmsd}')
    results_file.write('\n')
    results_file.write(f'bias_train: {bias}')
    results_file.write('\n')
    results_file.write(f'sdep_train: {sdep}')
    results_file.write('\n')
    results_file.write('\n')

    fig, ax = plt.subplots()

    # Line of best fit
    try:
        a, b = np.polyfit(df2['y'], df2['y_pred'], 1)
        plot_a = '{:.3f}'.format(a)
        plot_b = '{:.3f}'.format(b)
        plt.plot(df2['y'], a * df2['y'] + b, color='purple')
    except np.linalg.LinAlgError:
        pass

    # Plot everything
    try:
        plt.plot([], [], ' ', label='Training set')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
        plt.plot([], [], ' ', label=f'Bias : {bias}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep}')
        plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
        plt.scatter(df2['y'], df2['y_pred'])
        order = [0,1,2,3,4,5]
    except NameError:
        plt.plot([], [], ' ', label='Training set')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
        plt.plot([], [], ' ', label=f'Bias : {bias}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep}')
        plt.scatter(df2['y'], df2['y_pred'])
        order = [0,1,2,3,4]
        pass

    # x=y line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Legend
    plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
    plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
    handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    leg.get_frame().set_linewidth(0.0)
    fig.savefig(f'resample_{resample}_train_pred.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.cla()
    plt.close(fig)

    ### Val
    df = pd.read_csv(f'{root_dir}/resample_pred_val_avg.csv', index_col=False)
    df2 = pd.read_csv(f'resample_{resample}_val_plot.csv', index_col=False)
    df2.columns = ['Mol', 'Temp', 'y']
    df2['y_pred'] = df[f'resample_{resample}']
    df2.to_csv(f'resample_{resample}_val_plot.csv', index=False)

    np_y = df2['y'].to_numpy()
    np_y_pred = df2['y_pred'].to_numpy()

    r2 = r2_score(np_y, np_y_pred)
    rmsd = (mean_squared_error(np_y, np_y_pred))**0.5
    bias = np.mean(np_y_pred - np_y)
    sdep = (np.mean((np_y_pred - np_y - bias)**2))**0.5

    r2 = '{:.3f}'.format(r2)
    rmsd = '{:.3f}'.format(rmsd)
    bias = '{:.3f}'.format(bias)
    sdep = '{:.3f}'.format(sdep)

    results_file.write(f'r2_val: {r2}')
    results_file.write('\n')
    results_file.write(f'rmsd_val: {rmsd}')
    results_file.write('\n')
    results_file.write(f'bias_val: {bias}')
    results_file.write('\n')
    results_file.write(f'sdep_val: {sdep}')
    results_file.write('\n')
    results_file.write('\n')

    fig, ax = plt.subplots()

    # Line of best fit
    try:
        a, b = np.polyfit(df2['y'], df2['y_pred'], 1)
        plot_a = '{:.3f}'.format(a)
        plot_b = '{:.3f}'.format(b)
        plt.plot(df2['y'], a * df2['y'] + b, color='purple')
    except np.linalg.LinAlgError:
        pass

    # Plot everything
    try:
        plt.plot([], [], ' ', label='Validation set')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
        plt.plot([], [], ' ', label=f'Bias : {bias}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep}')
        plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
        plt.scatter(df2['y'], df2['y_pred'])
        order = [0,1,2,3,4,5]
    except NameError:
        plt.plot([], [], ' ', label='Validation set')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
        plt.plot([], [], ' ', label=f'Bias : {bias}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep}')
        plt.scatter(df2['y'], df2['y_pred'])
        order = [0,1,2,3,4]
        pass

    # x=y line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Legend
    plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
    plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
    handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    leg.get_frame().set_linewidth(0.0)
    fig.savefig(f'resample_{resample}_val_pred.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.cla()
    plt.close(fig)

    ### Test
    df = pd.read_csv(f'{root_dir}/resample_pred_test_avg.csv', index_col=False)
    df2 = pd.read_csv(f'resample_{resample}_test_plot.csv', index_col=False)
    df2.columns = ['Mol', 'Temp', 'y']
    df2['y_pred'] = df[f'resample_{resample}']
    df2.to_csv(f'resample_{resample}_test_plot.csv', index=False)

    np_y = df2['y'].to_numpy()
    np_y_pred = df2['y_pred'].to_numpy()

    r2 = r2_score(np_y, np_y_pred)
    rmsd = (mean_squared_error(np_y, np_y_pred))**0.5
    bias = np.mean(np_y_pred - np_y)
    sdep = (np.mean((np_y_pred - np_y - bias)**2))**0.5

    r2 = '{:.3f}'.format(r2)
    rmsd = '{:.3f}'.format(rmsd)
    bias = '{:.3f}'.format(bias)
    sdep = '{:.3f}'.format(sdep)

    results_file.write(f'r2_test: {r2}')
    results_file.write('\n')
    results_file.write(f'rmsd_test: {rmsd}')
    results_file.write('\n')
    results_file.write(f'bias_test: {bias}')
    results_file.write('\n')
    results_file.write(f'sdep_test: {sdep}')
    results_file.write('\n')
    results_file.write('\n')
    results_file.close()

    fig, ax = plt.subplots()

    # Line of best fit
    try:
        a, b = np.polyfit(df2['y'], df2['y_pred'], 1)
        plot_a = '{:.3f}'.format(a)
        plot_b = '{:.3f}'.format(b)
        plt.plot(df2['y'], a * df2['y'] + b, color='purple')
    except np.linalg.LinAlgError:
        pass

    # Plot everything
    try:
        plt.plot([], [], ' ', label='Test set')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
        plt.plot([], [], ' ', label=f'Bias : {bias}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep}')
        plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
        plt.scatter(df2['y'], df2['y_pred'])
        order = [0,1,2,3,4,5]
    except NameError:
        plt.plot([], [], ' ', label='Test set')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
        plt.plot([], [], ' ', label=f'Bias : {bias}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep}')
        plt.scatter(df2['y'], df2['y_pred'])
        order = [0,1,2,3,4]
        pass

    # x=y line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Legend
    plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
    plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
    handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    leg.get_frame().set_linewidth(0.0)
    fig.savefig(f'resample_{resample}_test_pred.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.cla()
    plt.close(fig)

    ### Individual solvent predictions
    try:
        os.mkdir('individual_solvent_predictions')
        os.mkdir('individual_solvent_predictions/train')
        os.mkdir('individual_solvent_predictions/val')
        os.mkdir('individual_solvent_predictions/test')
    except FileExistsError:
        pass

    sets = ['train', 'val', 'test']

    for set in sets:

        df_charged = pd.read_csv(f'resample_{resample}_{set}_plot.csv', index_col=False)
        df_water = pd.read_csv(f'resample_{resample}_{set}_plot.csv', index_col=False)
        df_methanol = pd.read_csv(f'resample_{resample}_{set}_plot.csv', index_col=False)
        df_chloroform = pd.read_csv(f'resample_{resample}_{set}_plot.csv', index_col=False)
        df_carbontet = pd.read_csv(f'resample_{resample}_{set}_plot.csv', index_col=False)
                    
        for mol in df_charged['Mol']:
            if 'ion' not in mol:
                df_charged.drop(df_charged.loc[df_charged['Mol']==mol].index, inplace=True)

        df_charged.reset_index(inplace=True)
        df_charged.drop(columns = ['index'], inplace=True)
        df_charged.to_csv(f'individual_solvent_predictions/{set}/resample_{resample}_{set}_charged_pred.csv')
                    
        for mol in df_methanol['Mol']:
            if 'methanol' not in mol:
                df_methanol.drop(df_methanol.loc[df_methanol['Mol']==mol].index, inplace=True)
            if 'ion' in mol:
                df_methanol.drop(df_methanol.loc[df_methanol['Mol']==mol].index, inplace=True)

        df_methanol.reset_index(inplace=True)
        df_methanol.drop(columns = ['index'], inplace=True)
        df_methanol.to_csv(f'individual_solvent_predictions/{set}/resample_{resample}_{set}_methanol_pred.csv')

        for mol in df_chloroform['Mol']:
            if 'chloroform' not in mol:
                df_chloroform.drop(df_chloroform.loc[df_chloroform['Mol']==mol].index, inplace=True)

        df_chloroform.reset_index(inplace=True)
        df_chloroform.drop(columns = ['index'], inplace=True)
        df_chloroform.to_csv(f'individual_solvent_predictions/{set}/resample_{resample}_{set}_chloroform_pred.csv')

        for mol in df_carbontet['Mol']:
            if 'carbontet' not in mol:
                df_carbontet.drop(df_carbontet.loc[df_carbontet['Mol']==mol].index, inplace=True)

        df_carbontet.reset_index(inplace=True)
        df_carbontet.drop(columns = ['index'], inplace=True)
        df_carbontet.to_csv(f'individual_solvent_predictions/{set}/resample_{resample}_{set}_carbontet_pred.csv')

        for mol in df_water['Mol']:
            if 'ion' in mol:
                df_water.drop(df_water.loc[df_water['Mol']==mol].index, inplace=True)
            if 'methanol' in mol:
                df_water.drop(df_water.loc[df_water['Mol']==mol].index, inplace=True)
            if 'chloroform' in mol:
                df_water.drop(df_water.loc[df_water['Mol']==mol].index, inplace=True)
            if 'carbontet' in mol:
                df_water.drop(df_water.loc[df_water['Mol']==mol].index, inplace=True)

        df_water.reset_index(inplace=True)
        df_water.drop(columns = ['index'], inplace=True)
        df_water.to_csv(f'individual_solvent_predictions/{set}/resample_{resample}_{set}_water_pred.csv')

### Compare CV average results against resample average results, should be similar

r2_train = []
rmsd_train = []
bias_train = []
sdep_train = []

r2_val = []
rmsd_val = []
bias_val = []
sdep_val = []

r2_test = []
rmsd_test = []
bias_test = []
sdep_test = []

for folders in glob.glob(f'{root_dir}/*/*.txt'):

    with open(f'{folders}', 'r') as f:
        lines = f.readlines()		
        r2_line_test = [line for line in lines if line.startswith('r2_test')]
        rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
        bias_line_test = [line for line in lines if line.startswith('bias_test')]		
        sdep_line_test = [line for line in lines if line.startswith('sdep_test')]

        r2_test.append(float(r2_line_test[0].split()[1]))
        rmsd_test.append(float(rmsd_line_test[0].split()[1]))
        bias_test.append(float(bias_line_test[0].split()[1]))
        sdep_test.append(float(sdep_line_test[0].split()[1]))

        r2_line_train = [line for line in lines if line.startswith('r2_train')]
        rmsd_line_train = [line for line in lines if line.startswith('rmsd_train')]
        bias_line_train = [line for line in lines if line.startswith('bias_train')]		
        sdep_line_train = [line for line in lines if line.startswith('sdep_train')]

        r2_train.append(float(r2_line_train[0].split()[1]))
        rmsd_train.append(float(rmsd_line_train[0].split()[1]))
        bias_train.append(float(bias_line_train[0].split()[1]))
        sdep_train.append(float(sdep_line_train[0].split()[1]))

        r2_line_val = [line for line in lines if line.startswith('r2_val')]
        rmsd_line_val = [line for line in lines if line.startswith('rmsd_val')]
        bias_line_val = [line for line in lines if line.startswith('bias_val')]		
        sdep_line_val = [line for line in lines if line.startswith('sdep_val')]

        r2_val.append(float(r2_line_val[0].split()[1]))
        rmsd_val.append(float(rmsd_line_val[0].split()[1]))
        bias_val.append(float(bias_line_val[0].split()[1]))
        sdep_val.append(float(sdep_line_val[0].split()[1]))

r2_train_avg = round(sum(r2_train) / len(r2_train), 3)
rmsd_train_avg = round(sum(rmsd_train) / len(rmsd_train), 3)
bias_train_avg = round(sum(bias_train) / len(bias_train), 3)
sdep_train_avg = round(sum(sdep_train) / len(sdep_train), 3)

r2_test_avg = round(sum(r2_test) / len(r2_test), 3)
rmsd_test_avg = round(sum(rmsd_test) / len(rmsd_test), 3)
bias_test_avg = round(sum(bias_test) / len(bias_test), 3)
sdep_test_avg = round(sum(sdep_test) / len(sdep_test), 3)

r2_val_avg = round(sum(r2_val) / len(r2_val), 3)
rmsd_val_avg = round(sum(rmsd_val) / len(rmsd_val), 3)
bias_val_avg = round(sum(bias_val) / len(bias_val), 3)
sdep_val_avg = round(sum(sdep_val) / len(sdep_val), 3)

results_file = open(f'{root_dir}/CV_avg_stats_against_resamples.txt', 'w')
results_file.write(f'r2_train_avg: {r2_train_avg}')
results_file.write('\n')
results_file.write(f'rmsd_train_avg: {rmsd_train_avg}')
results_file.write('\n')
results_file.write(f'bias_train_avg: {bias_train_avg}')
results_file.write('\n')
results_file.write(f'sdep_train_avg: {sdep_train_avg}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_val_avg: {r2_val_avg}')
results_file.write('\n')
results_file.write(f'rmsd_val_avg: {rmsd_val_avg}')
results_file.write('\n')
results_file.write(f'bias_val_avg: {bias_val_avg}')
results_file.write('\n')
results_file.write(f'sdep_val_avg: {sdep_val_avg}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_test_avg: {r2_test_avg}')
results_file.write('\n')
results_file.write(f'rmsd_test_avg: {rmsd_test_avg}')
results_file.write('\n')
results_file.write(f'bias_test_avg: {bias_test_avg}')
results_file.write('\n')
results_file.write(f'sdep_test_avg: {sdep_test_avg}')


### Output average individual solvent predictions for test set

water_r2_list = []
water_rmsd_list = []
water_bias_list = []
water_sdep_list = []

methanol_r2_list = []
methanol_rmsd_list = []
methanol_bias_list = []
methanol_sdep_list = []

chloroform_r2_list = []
chloroform_rmsd_list = []
chloroform_bias_list = []
chloroform_sdep_list = []

carbontet_r2_list = []
carbontet_rmsd_list = []
carbontet_bias_list = []
carbontet_sdep_list = []

charged_r2_list = []
charged_rmsd_list = []
charged_bias_list = []
charged_sdep_list = []


for resample in range(1, no_resamples + 1):
    for csv in os.listdir(f'{root_dir}/resample_{resample}/resample_{resample}/individual_solvent_predictions/test'):
        if csv.endswith('.csv'):
            solvent = csv.split('_')[3]
            df = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}/individual_solvent_predictions/test/{csv}')
                                         
            np_y = df['y'].to_numpy()
            np_y_pred = df['y_pred'].to_numpy()

            r2 = r2_score(np_y, np_y_pred)
            rmsd = (mean_squared_error(np_y, np_y_pred))**0.5
            bias = np.mean(np_y_pred - np_y)
            sdep = (np.mean((np_y_pred - np_y - bias)**2))**0.5

            r2 = '{:.3f}'.format(r2)
            rmsd = '{:.3f}'.format(rmsd)
            bias = '{:.3f}'.format(bias)
            sdep = '{:.3f}'.format(sdep)
                                        
            r2 = float(r2)
            rmsd = float(rmsd)
            bias = float(bias)
            sdep = float(sdep)

            if solvent == 'charged':
                charged_r2_list.append(r2)
                charged_rmsd_list.append(rmsd)
                charged_bias_list.append(bias)
                charged_sdep_list.append(sdep)
                txt_file = open(f'{root_dir}/resample_{resample}/resample_{resample}/individual_solvent_predictions/test/resample_{resample}_test_stats_{solvent}.txt', 'w')
                txt_file.write(f'r2_test_{solvent}: {r2}')
                txt_file.write('\n')
                txt_file.write(f'rmsd_test_{solvent}: {rmsd}')
                txt_file.write('\n')
                txt_file.write(f'bias_test_{solvent}: {bias}')
                txt_file.write('\n')
                txt_file.write(f'sdep_test_{solvent}: {sdep}')
                txt_file.close()
            elif solvent == 'water':
                water_r2_list.append(r2)
                water_rmsd_list.append(rmsd)
                water_bias_list.append(bias)
                water_sdep_list.append(sdep)
                txt_file = open(f'{root_dir}/resample_{resample}/resample_{resample}/individual_solvent_predictions/test/resample_{resample}_test_stats_{solvent}.txt', 'w')
                txt_file.write(f'r2_test_{solvent}: {r2}')
                txt_file.write('\n')
                txt_file.write(f'rmsd_test_{solvent}: {rmsd}')
                txt_file.write('\n')
                txt_file.write(f'bias_test_{solvent}: {bias}')
                txt_file.write('\n')
                txt_file.write(f'sdep_test_{solvent}: {sdep}')
                txt_file.close()
            elif solvent == 'methanol':
                methanol_r2_list.append(r2)
                methanol_rmsd_list.append(rmsd)
                methanol_bias_list.append(bias)
                methanol_sdep_list.append(sdep)
                txt_file = open(f'{root_dir}/resample_{resample}/resample_{resample}/individual_solvent_predictions/test/resample_{resample}_test_stats_{solvent}.txt', 'w')
                txt_file.write(f'r2_test_{solvent}: {r2}')
                txt_file.write('\n')
                txt_file.write(f'rmsd_test_{solvent}: {rmsd}')
                txt_file.write('\n')
                txt_file.write(f'bias_test_{solvent}: {bias}')
                txt_file.write('\n')
                txt_file.write(f'sdep_test_{solvent}: {sdep}')
                txt_file.close()
            elif solvent == 'chloroform':
                chloroform_r2_list.append(r2)
                chloroform_rmsd_list.append(rmsd)
                chloroform_bias_list.append(bias)
                chloroform_sdep_list.append(sdep)
                txt_file = open(f'{root_dir}/resample_{resample}/resample_{resample}/individual_solvent_predictions/test/resample_{resample}_test_stats_{solvent}.txt', 'w')
                txt_file.write(f'r2_test_{solvent}: {r2}')
                txt_file.write('\n')
                txt_file.write(f'rmsd_test_{solvent}: {rmsd}')
                txt_file.write('\n')
                txt_file.write(f'bias_test_{solvent}: {bias}')
                txt_file.write('\n')
                txt_file.write(f'sdep_test_{solvent}: {sdep}')
                txt_file.close()
            elif solvent == 'carbontet':
                carbontet_r2_list.append(r2)
                carbontet_rmsd_list.append(rmsd)
                carbontet_bias_list.append(bias)
                carbontet_sdep_list.append(sdep)
                txt_file = open(f'{root_dir}/resample_{resample}/resample_{resample}/individual_solvent_predictions/test/resample_{resample}_test_stats_{solvent}.txt', 'w')
                txt_file.write(f'r2_test_{solvent}: {r2}')
                txt_file.write('\n')
                txt_file.write(f'rmsd_test_{solvent}: {rmsd}')
                txt_file.write('\n')
                txt_file.write(f'bias_test_{solvent}: {bias}')
                txt_file.write('\n')
                txt_file.write(f'sdep_test_{solvent}: {sdep}')
                txt_file.close()               
            
water_r2 = sum(water_r2_list) / len(water_r2_list)
water_rmsd = sum(water_rmsd_list) / len(water_rmsd_list)
water_bias = sum(water_bias_list) / len(water_bias_list)
water_sdep = sum(water_sdep_list) / len(water_sdep_list)
methanol_r2 = sum(methanol_r2_list) / len(methanol_r2_list)
methanol_rmsd = sum(methanol_rmsd_list) / len(methanol_rmsd_list)
methanol_bias = sum(methanol_bias_list) / len(methanol_bias_list)
methanol_sdep = sum(methanol_sdep_list) / len(methanol_sdep_list)
chloroform_r2 = sum(chloroform_r2_list) / len(chloroform_r2_list)
chloroform_rmsd = sum(chloroform_rmsd_list) / len(chloroform_rmsd_list)
chloroform_bias = sum(chloroform_bias_list) / len(chloroform_bias_list)
chloroform_sdep = sum(chloroform_sdep_list) / len(chloroform_sdep_list)
carbontet_r2 = sum(carbontet_r2_list) / len(carbontet_r2_list)
carbontet_rmsd = sum(carbontet_rmsd_list) / len(carbontet_rmsd_list)
carbontet_bias = sum(carbontet_bias_list) / len(carbontet_bias_list)
carbontet_sdep = sum(carbontet_sdep_list) / len(carbontet_sdep_list)
charged_r2 = sum(charged_r2_list) / len(charged_r2_list)
charged_rmsd = sum(charged_rmsd_list) / len(charged_rmsd_list)
charged_bias = sum(charged_bias_list) / len(charged_bias_list)
charged_sdep = sum(charged_sdep_list) / len(charged_sdep_list)

water_r2 = '{:.3f}'.format(water_r2)
water_rmsd = '{:.3f}'.format(water_rmsd)
water_bias = '{:.3f}'.format(water_bias)
water_sdep = '{:.3f}'.format(water_sdep)
methanol_r2 = '{:.3f}'.format(methanol_r2)
methanol_rmsd = '{:.3f}'.format(methanol_rmsd)
methanol_bias = '{:.3f}'.format(methanol_bias)
methanol_sdep = '{:.3f}'.format(methanol_sdep)
chloroform_r2 = '{:.3f}'.format(chloroform_r2)
chloroform_rmsd = '{:.3f}'.format(chloroform_rmsd)
chloroform_bias = '{:.3f}'.format(chloroform_bias)
chloroform_sdep = '{:.3f}'.format(chloroform_sdep)
carbontet_r2 = '{:.3f}'.format(carbontet_r2)
carbontet_rmsd = '{:.3f}'.format(carbontet_rmsd)
carbontet_bias = '{:.3f}'.format(carbontet_bias)
carbontet_sdep = '{:.3f}'.format(carbontet_sdep)
charged_r2 = '{:.3f}'.format(charged_r2)
charged_rmsd = '{:.3f}'.format(charged_rmsd)
charged_bias = '{:.3f}'.format(charged_bias)
charged_sdep = '{:.3f}'.format(charged_sdep)

avg_water_file = open(f'{root_dir}/avg_resample_test_stats_water.txt', 'w')
avg_water_file.write(f'r2_test: {water_r2}')
avg_water_file.write('\n')
avg_water_file.write(f'rmsd_test: {water_rmsd}')
avg_water_file.write('\n')
avg_water_file.write(f'bias_test: {water_bias}')
avg_water_file.write('\n')
avg_water_file.write(f'sdep_test: {water_sdep}')
avg_water_file.close()

avg_methanol_file = open(f'{root_dir}/avg_resample_test_stats_methanol.txt', 'w')
avg_methanol_file.write(f'r2_test: {methanol_r2}')
avg_methanol_file.write('\n')
avg_methanol_file.write(f'rmsd_test: {methanol_rmsd}')
avg_methanol_file.write('\n')
avg_methanol_file.write(f'bias_test: {methanol_bias}')
avg_methanol_file.write('\n')
avg_methanol_file.write(f'sdep_test: {methanol_sdep}')
avg_methanol_file.close()

avg_chloroform_file = open(f'{root_dir}/avg_resample_test_stats_chloroform.txt', 'w')
avg_chloroform_file.write(f'r2_test: {chloroform_r2}')
avg_chloroform_file.write('\n')
avg_chloroform_file.write(f'rmsd_test: {chloroform_rmsd}')
avg_chloroform_file.write('\n')
avg_chloroform_file.write(f'bias_test: {chloroform_bias}')
avg_chloroform_file.write('\n')
avg_chloroform_file.write(f'sdep_test: {chloroform_sdep}')
avg_chloroform_file.close()

avg_carbontet_file = open(f'{root_dir}/avg_resample_test_stats_carbontet.txt', 'w')
avg_carbontet_file.write(f'r2_test: {carbontet_r2}')
avg_carbontet_file.write('\n')
avg_carbontet_file.write(f'rmsd_test: {carbontet_rmsd}')
avg_carbontet_file.write('\n')
avg_carbontet_file.write(f'bias_test: {carbontet_bias}')
avg_carbontet_file.write('\n')
avg_carbontet_file.write(f'sdep_test: {carbontet_sdep}')
avg_carbontet_file.close()

avg_charged_file = open(f'{root_dir}/avg_resample_test_stats_charged.txt', 'w')
avg_charged_file.write(f'r2_test: {charged_r2}')
avg_charged_file.write('\n')
avg_charged_file.write(f'rmsd_test: {charged_rmsd}')
avg_charged_file.write('\n')
avg_charged_file.write(f'bias_test: {charged_bias}')
avg_charged_file.write('\n')
avg_charged_file.write(f'sdep_test: {charged_sdep}')
avg_charged_file.close()

### Output standard deviation of average test set stats

r2_test = []
rmsd_test = []
bias_test = []
sdep_test = []         

for resample in range(1, no_resamples + 1):

    with open(f'{root_dir}/resample_{resample}/resample_{resample}/resample_{resample}_stats.txt', 'r') as f:
        lines = f.readlines()
        r2_line_test = [line for line in lines if line.startswith('r2_test')]
        rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
        bias_line_test = [line for line in lines if line.startswith('bias_test')]		
        sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
        r2_test.append(float(r2_line_test[0].split()[1]))
        rmsd_test.append(float(rmsd_line_test[0].split()[1]))
        bias_test.append(float(bias_line_test[0].split()[1]))
        sdep_test.append(float(sdep_line_test[0].split()[1]))

sd_r2_test = stats.stdev(r2_test)
sd_rmsd_test = stats.stdev(rmsd_test)
sd_bias_test = stats.stdev(bias_test)
sd_sdep_test = stats.stdev(sdep_test)
sd_r2_test = '{:.3f}'.format(sd_r2_test)
sd_rmsd_test = '{:.3f}'.format(sd_rmsd_test)
sd_bias_test = '{:.3f}'.format(sd_bias_test)
sd_sdep_test = '{:.3f}'.format(sd_sdep_test)
sd_file = open(f'{root_dir}/sd_test_stats.txt', 'w')
sd_file.write(f'sd_r2_test: {sd_r2_test}')
sd_file.write('\n')
sd_file.write(f'sd_rmsd_test: {sd_rmsd_test}')
sd_file.write('\n')
sd_file.write(f'sd_bias_test: {sd_bias_test}')
sd_file.write('\n')
sd_file.write(f'sd_sdep_test: {sd_sdep_test}')
sd_file.close()

### Output average SFE prediction over resamples for each solute (instead of over the full set of SFE predictions)

input_dataset['Mol_Temp'] = input_dataset['Mol'] + '_' + input_dataset['Temp'].astype(str)
mol_temp_list = input_dataset['Mol_Temp'].tolist()
len_list = len(mol_temp_list)
summary_df = pd.DataFrame(columns = range(len_list))
summary_df.set_axis(mol_temp_list, axis='columns', inplace=True)
                                
df_list = []
for column in summary_df:
                            
    tmp_pred_list = []
    tmp_exp_list = []
    for resample in os.listdir(f'{root_dir}'):
        if os.path.isdir(f'{root_dir}'+'/'+resample):
            resample_test_pred_df = pd.read_csv(f'{root_dir}/{resample}/{resample}/{resample}_test_plot.csv')
            resample_test_pred_df['Mol_Temp'] = resample_test_pred_df['Mol'] + '_' + resample_test_pred_df['Temp'].astype(str)
            pred = (resample_test_pred_df.loc[resample_test_pred_df['Mol_Temp'] == column, 'y_pred'])
            exp = (resample_test_pred_df.loc[resample_test_pred_df['Mol_Temp'] == column, 'y'])
                                    
            if pred.shape == (0,):
                pass
            else:
                pred_df = pred.to_frame()
                pred_value = pred_df['y_pred'].iloc[0]
                tmp_pred_list.append(pred_value)
                exp_df = exp.to_frame()
                exp_value = exp_df['y'].iloc[0]
                tmp_exp_list.append(exp_value)
                         
    try:
        tmp_exp_list = tmp_exp_list[0]
    except IndexError:
         pass
    if not tmp_exp_list:
        continue
    avg_tmp_pred = sum(tmp_pred_list) / len(tmp_pred_list)
    save_to_csv = [column, tmp_exp_list, avg_tmp_pred]
    csv_header = ['Mol', 'y', 'y_pred']
    df = pd.DataFrame([save_to_csv], columns=csv_header)
    df_list.append(df)

df2 = pd.concat(df_list)
df2.to_csv(f'{root_dir}/avg_pred_over_resamples.csv')
                                    
y_test = df2['y']
y_pred_test = df2['y_pred']
y_test_np = y_test.to_numpy()
y_pred_test_np = y_pred_test.to_numpy()
r2_test = r2_score(y_test_np, y_pred_test_np)
rmsd_test = (mean_squared_error(y_test_np, y_pred_test_np))**0.5
bias_test = np.mean(y_pred_test_np - y_test_np)
sdep_test = (np.mean((y_pred_test_np - y_test_np - bias_test)**2))**0.5
r2_test = '{:.3f}'.format(r2_test)
rmsd_test = '{:.3f}'.format(rmsd_test)
bias_test = '{:.3f}'.format(bias_test)
sdep_test = '{:.3f}'.format(sdep_test)
results_file = open(f'{root_dir}/avg_pred_over_resamples_test_stats.txt', 'w')
results_file.write(f'r2_test: {r2_test}')
results_file.write('\n')
results_file.write(f'rmsd_test: {rmsd_test}')
results_file.write('\n')
results_file.write(f'bias_test: {bias_test}')
results_file.write('\n')
results_file.write(f'sdep_test: {sdep_test}')
results_file.close()
fig, ax = plt.subplots()   
    
# Line of best fit
try:
    a, b = np.polyfit(df2['y'], df2['y_pred'], 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    plt.plot([], [], ' ', label='Test set '+r'$ΔG^{neutral, ionic}_{solv}$')
    plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_test}')
    plt.plot([], [], ' ', label=f'RMSD : {rmsd_test}')
    plt.plot([], [], ' ', label=f'Bias : {bias_test}')
    plt.plot([], [], ' ', label=f'SDEP : {sdep_test}')
    plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(df2['y'], df2['y_pred'])
    plt.plot(df2['y'], a * df2['y'] + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
handles, labels = plt.gca().get_legend_handles_labels()
leg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/avg_pred_over_resamples_plot_with_pred_avg_stats.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

with open(f'{root_dir}/avg_resample_stats.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2_test_avg = float(r2_line_test[0].split()[1])
    rmsd_test_avg = float(rmsd_line_test[0].split()[1])
    bias_test_avg = float(bias_line_test[0].split()[1])
    sdep_test_avg = float(sdep_line_test[0].split()[1])

fig, ax = plt.subplots()    
   
# Line of best fit
try:
    a, b = np.polyfit(df2['y'], df2['y_pred'], 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
   plt.plot([], [], ' ', label='Test set '+r'$ΔG^{neutral, ionic}_{solv}$')
   plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_test_avg}')
   plt.plot([], [], ' ', label=f'RMSD : {rmsd_test_avg}')
   plt.plot([], [], ' ', label=f'Bias : {bias_test_avg}')
   plt.plot([], [], ' ', label=f'SDEP : {sdep_test_avg}')
   plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
   plt.scatter(df2['y'], df2['y_pred'])
   plt.plot(df2['y'], a * df2['y'] + b, color='purple')
   order = [0,1,2,3,4,5]
except NameError:
   pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
handles, labels = plt.gca().get_legend_handles_labels()
leg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/avg_pred_over_resamples_plot_with_resample_avg_stats.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# Individual solvent plots of average predictions per solute over all resamples. Stats calculated from average over all resamples.

# chloroform

df_chloroform = pd.read_csv(f'{root_dir}/avg_pred_over_resamples.csv')

for solvent in df_chloroform['Mol']:
    if 'chloroform' not in solvent:
        df_chloroform.drop(df_chloroform.loc[df_chloroform['Mol'] == solvent].index, inplace=True)

y_test_np = df_chloroform['y'].to_numpy()
y_pred_test_np = df_chloroform['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_chloroform.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Chloroform '+r'$ΔG^{neutral}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/chloroform_ind_plot.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# carbontet

df_carbontet = pd.read_csv(f'{root_dir}/avg_pred_over_resamples.csv')

for solvent in df_carbontet['Mol']:
    if 'carbontet' not in solvent:
        df_carbontet.drop(df_carbontet.loc[df_carbontet['Mol'] == solvent].index, inplace=True)

y_test_np = df_carbontet['y'].to_numpy()
y_pred_test_np = df_carbontet['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_carbontet.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Carbon Tetrachloride '+r'$ΔG^{neutral}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/carbontet_ind_plot.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# charged

df_charged = pd.read_csv(f'{root_dir}/avg_pred_over_resamples.csv')

for solvent in df_charged['Mol']:
    if 'ion' not in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)

y_test_np = df_charged['y'].to_numpy()
y_pred_test_np = df_charged['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_charged.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set '+r'$ΔG^{ionic}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/charged_ind_plot.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# methanol

df_methanol = pd.read_csv(f'{root_dir}/avg_pred_over_resamples.csv')

for solvent in df_methanol['Mol']:
    if 'ion' in solvent:
        df_methanol.drop(df_methanol.loc[df_methanol['Mol'] == solvent].index, inplace=True)

    if 'methanol' not in solvent:
        df_methanol.drop(df_methanol.loc[df_methanol['Mol'] == solvent].index, inplace=True)

y_test_np = df_methanol['y'].to_numpy()
y_pred_test_np = df_methanol['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_methanol.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Methanol '+r'$ΔG^{neutral}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/methanol_ind_plot.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# water

df_water = pd.read_csv(f'{root_dir}/avg_pred_over_resamples.csv')

for solvent in df_water['Mol']:
    if 'ion' in solvent:
        df_water.drop(df_water.loc[df_water['Mol'] == solvent].index, inplace=True)

    if 'methanol' in solvent:
        df_water.drop(df_water.loc[df_water['Mol'] == solvent].index, inplace=True)

    if 'chloroform' in solvent:
        df_water.drop(df_water.loc[df_water['Mol'] == solvent].index, inplace=True)

    if 'carbontet' in solvent:
        df_water.drop(df_water.loc[df_water['Mol'] == solvent].index, inplace=True)

y_test_np = df_water['y'].to_numpy()
y_pred_test_np = df_water['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_water.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Water '+r'$ΔG^{neutral}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/water_ind_plot.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

### Individual plots with all predictions per solute, instead of average. Stats calculated from average over resamples

df = pd.read_csv(f'{root_dir}/resample_pred_test_avg.csv')

mol_list = []

for columns in df:
	if 'Mol' in columns:
		mol_list.append(columns)

mol_values_list = []

for values in mol_list:
	for mol_values in df[f'{values}']:
		mol_values_list.append(mol_values)

no_nan_mol_list = [x for x in mol_values_list if str(x) != 'nan']

###

exp_list = []

for columns in df:
	if 'HFE' in columns:
		exp_list.append(columns)

exp_values_list = []

for values in exp_list:
	for exp_values in df[f'{values}']:
		exp_values_list.append(exp_values)

no_nan_exp_list = [x for x in exp_values_list if str(x) != 'nan']

###

calc_list = []

for columns in df:
	if 'resample' in columns:
		calc_list.append(columns)

calc_values_list = []

for values in calc_list:
	for calc_values in df[f'{values}']:
		calc_values_list.append(calc_values)

no_nan_calc_list = [x for x in calc_values_list if str(x) != 'nan']

###

df2 = pd.DataFrame({'Mol':no_nan_mol_list, 'y':no_nan_exp_list, 'y_pred':no_nan_calc_list})
df2.to_csv(f'{root_dir}/full_df.csv')

###

df_full = pd.read_csv(f'{root_dir}/full_df.csv')

exp = df_full['y'].to_numpy()
calc = df_full['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_stats.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))


fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(exp, calc, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set '+r'$ΔG^{neutral, ionic}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(exp, calc)
    plt.plot(exp, a * exp + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/full_resample_exp_pred_plot_no_avg_with_avg_stats.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# Individual solvent plots

# chloroform

df_chloroform = pd.read_csv(f'{root_dir}/full_df.csv')

for solvent in df_chloroform['Mol']:
    if 'chloroform' not in solvent:
        df_chloroform.drop(df_chloroform.loc[df_chloroform['Mol'] == solvent].index, inplace=True)

y_test_np = df_chloroform['y'].to_numpy()
y_pred_test_np = df_chloroform['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_chloroform.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Chloroform '+r'$ΔG^{neutral}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/chloroform_ind_plot_no_avg_with_avg_stats.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# carbontet

df_carbontet = pd.read_csv(f'{root_dir}/full_df.csv')

for solvent in df_carbontet['Mol']:
    if 'carbontet' not in solvent:
        df_carbontet.drop(df_carbontet.loc[df_carbontet['Mol'] == solvent].index, inplace=True)

y_test_np = df_carbontet['y'].to_numpy()
y_pred_test_np = df_carbontet['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_carbontet.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Carbon Tetrachloride '+r'$ΔG^{neutral}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/carbontet_ind_plot_no_avg_with_avg_stats.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# charged

df_charged = pd.read_csv(f'{root_dir}/full_df.csv')

for solvent in df_charged['Mol']:
    if 'ion' not in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)

y_test_np = df_charged['y'].to_numpy()
y_pred_test_np = df_charged['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_charged.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set '+r'$ΔG^{ionic}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/charged_ind_plot_no_avg_with_avg_stats.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# methanol

df_methanol = pd.read_csv(f'{root_dir}/full_df.csv')

for solvent in df_methanol['Mol']:
    if 'ion' in solvent:
        df_methanol.drop(df_methanol.loc[df_methanol['Mol'] == solvent].index, inplace=True)

    if 'methanol' not in solvent:
        df_methanol.drop(df_methanol.loc[df_methanol['Mol'] == solvent].index, inplace=True)

y_test_np = df_methanol['y'].to_numpy()
y_pred_test_np = df_methanol['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_methanol.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Methanol '+r'$ΔG^{neutral}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/methanol_ind_plot_no_avg_with_avg_stats.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# water

df_water = pd.read_csv(f'{root_dir}/full_df.csv')

for solvent in df_water['Mol']:
    if 'ion' in solvent:
        df_water.drop(df_water.loc[df_water['Mol'] == solvent].index, inplace=True)

    if 'methanol' in solvent:
        df_water.drop(df_water.loc[df_water['Mol'] == solvent].index, inplace=True)

    if 'chloroform' in solvent:
        df_water.drop(df_water.loc[df_water['Mol'] == solvent].index, inplace=True)

    if 'carbontet' in solvent:
        df_water.drop(df_water.loc[df_water['Mol'] == solvent].index, inplace=True)

y_test_np = df_water['y'].to_numpy()
y_pred_test_np = df_water['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_water.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Water '+r'$ΔG^{neutral}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/water_ind_plot_no_avg_with_avg_stats.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# Extra plots to split charged results into methanol and water

charged_methanol_r2_list = []
charged_methanol_rmsd_list = []
charged_methanol_bias_list = []
charged_methanol_sdep_list = []

charged_water_r2_list = []
charged_water_rmsd_list = []
charged_water_bias_list = []
charged_water_sdep_list = []

for resample in range(1, no_resamples + 1):
    for csv in os.listdir(f'{root_dir}/resample_{resample}/resample_{resample}/individual_solvent_predictions/test'):
        if csv.endswith('.csv'):
            if 'charged' in csv:

                df_methanol = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}/individual_solvent_predictions/test/{csv}')
                df_water = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}/individual_solvent_predictions/test/{csv}')

                for solvent in df_methanol['Mol']:
                    if 'methanol' not in solvent:
                        df_methanol.drop(df_methanol.loc[df_methanol['Mol'] == solvent].index, inplace=True)
                       
                np_y = df_methanol['y'].to_numpy()
                np_y_pred = df_methanol['y_pred'].to_numpy()

                r2 = r2_score(np_y, np_y_pred)
                rmsd = (mean_squared_error(np_y, np_y_pred))**0.5
                bias = np.mean(np_y_pred - np_y)
                sdep = (np.mean((np_y_pred - np_y - bias)**2))**0.5

                r2 = '{:.3f}'.format(r2)
                rmsd = '{:.3f}'.format(rmsd)
                bias = '{:.3f}'.format(bias)
                sdep = '{:.3f}'.format(sdep)
                                        
                r2 = float(r2)
                rmsd = float(rmsd)
                bias = float(bias)
                sdep = float(sdep)

                charged_methanol_r2_list.append(r2)
                charged_methanol_rmsd_list.append(rmsd)
                charged_methanol_bias_list.append(bias)
                charged_methanol_sdep_list.append(sdep)

                txt_file = open(f'{root_dir}/resample_{resample}/resample_{resample}/individual_solvent_predictions/test/resample_{resample}_test_stats_charged_methanol.txt', 'w')
                txt_file.write(f'r2_test_charged_methanol: {r2}')
                txt_file.write('\n')
                txt_file.write(f'rmsd_test_charged_methanol: {rmsd}')
                txt_file.write('\n')
                txt_file.write(f'bias_test_charged_methanol: {bias}')
                txt_file.write('\n')
                txt_file.write(f'sdep_test_charged_methanol: {sdep}')
                txt_file.close()

                for solvent in df_water['Mol']:
                    if 'water' not in solvent:
                        df_water.drop(df_water.loc[df_water['Mol'] == solvent].index, inplace=True)
                       
                np_y = df_water['y'].to_numpy()
                np_y_pred = df_water['y_pred'].to_numpy()

                r2 = r2_score(np_y, np_y_pred)
                rmsd = (mean_squared_error(np_y, np_y_pred))**0.5
                bias = np.mean(np_y_pred - np_y)
                sdep = (np.mean((np_y_pred - np_y - bias)**2))**0.5

                r2 = '{:.3f}'.format(r2)
                rmsd = '{:.3f}'.format(rmsd)
                bias = '{:.3f}'.format(bias)
                sdep = '{:.3f}'.format(sdep)
                                        
                r2 = float(r2)
                rmsd = float(rmsd)
                bias = float(bias)
                sdep = float(sdep)

                charged_water_r2_list.append(r2)
                charged_water_rmsd_list.append(rmsd)
                charged_water_bias_list.append(bias)
                charged_water_sdep_list.append(sdep)

                txt_file = open(f'{root_dir}/resample_{resample}/resample_{resample}/individual_solvent_predictions/test/resample_{resample}_test_stats_charged_water.txt', 'w')
                txt_file.write(f'r2_test_charged_water: {r2}')
                txt_file.write('\n')
                txt_file.write(f'rmsd_test_charged_water: {rmsd}')
                txt_file.write('\n')
                txt_file.write(f'bias_test_charged_water: {bias}')
                txt_file.write('\n')
                txt_file.write(f'sdep_test_charged_water: {sdep}')
                txt_file.close()

charged_methanol_r2 = sum(charged_methanol_r2_list) / len(charged_methanol_r2_list)
charged_methanol_rmsd = sum(charged_methanol_rmsd_list) / len(charged_methanol_rmsd_list)
charged_methanol_bias = sum(charged_methanol_bias_list) / len(charged_methanol_bias_list)
charged_methanol_sdep = sum(charged_methanol_sdep_list) / len(charged_methanol_sdep_list)

charged_water_r2 = sum(charged_water_r2_list) / len(charged_water_r2_list)
charged_water_rmsd = sum(charged_water_rmsd_list) / len(charged_water_rmsd_list)
charged_water_bias = sum(charged_water_bias_list) / len(charged_water_bias_list)
charged_water_sdep = sum(charged_water_sdep_list) / len(charged_water_sdep_list)

charged_methanol_r2 = '{:.3f}'.format(charged_methanol_r2)
charged_methanol_rmsd = '{:.3f}'.format(charged_methanol_rmsd)
charged_methanol_bias = '{:.3f}'.format(charged_methanol_bias)
charged_methanol_sdep = '{:.3f}'.format(charged_methanol_sdep)

charged_water_r2 = '{:.3f}'.format(charged_water_r2)
charged_water_rmsd = '{:.3f}'.format(charged_water_rmsd)
charged_water_bias = '{:.3f}'.format(charged_water_bias)
charged_water_sdep = '{:.3f}'.format(charged_water_sdep)

avg_charged_methanol_file = open(f'{root_dir}/avg_resample_test_stats_charged_methanol.txt', 'w')
avg_charged_methanol_file.write(f'r2_test: {charged_methanol_r2}')
avg_charged_methanol_file.write('\n')
avg_charged_methanol_file.write(f'rmsd_test: {charged_methanol_rmsd}')
avg_charged_methanol_file.write('\n')
avg_charged_methanol_file.write(f'bias_test: {charged_methanol_bias}')
avg_charged_methanol_file.write('\n')
avg_charged_methanol_file.write(f'sdep_test: {charged_methanol_sdep}')
avg_charged_methanol_file.close()

avg_charged_water_file = open(f'{root_dir}/avg_resample_test_stats_charged_water.txt', 'w')
avg_charged_water_file.write(f'r2_test: {charged_water_r2}')
avg_charged_water_file.write('\n')
avg_charged_water_file.write(f'rmsd_test: {charged_water_rmsd}')
avg_charged_water_file.write('\n')
avg_charged_water_file.write(f'bias_test: {charged_water_bias}')
avg_charged_water_file.write('\n')
avg_charged_water_file.write(f'sdep_test: {charged_water_sdep}')
avg_charged_water_file.close()

# Individual solvent plots of average predictions per solute over all resamples. Stats calculated from average over all resamples.

# charged methanol

df_charged = pd.read_csv(f'{root_dir}/avg_pred_over_resamples.csv')

for solvent in df_charged['Mol']:
    if 'ion' not in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)
    if 'methanol' not in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)

y_test_np = df_charged['y'].to_numpy()
y_pred_test_np = df_charged['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_charged_methanol.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Methanol '+r'$ΔG^{ionic}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='upper left')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor((0, 0, 0, 0))
fig.savefig(f'{root_dir}/charged_methanol_ind_plot.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# charged water

df_charged = pd.read_csv(f'{root_dir}/avg_pred_over_resamples.csv')

for solvent in df_charged['Mol']:
    if 'ion' not in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)
    if 'methanol' in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)

y_test_np = df_charged['y'].to_numpy()
y_pred_test_np = df_charged['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_charged_water.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Water '+r'$ΔG^{ionic}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='upper left')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor((0, 0, 0, 0))
fig.savefig(f'{root_dir}/charged_water_ind_plot.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

### Individual plots with all predictions per solute, instead of average. Stats calculated from average over resamples

# charged methanol

df_charged = pd.read_csv(f'{root_dir}/full_df.csv')

for solvent in df_charged['Mol']:
    if 'ion' not in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)
    if 'methanol' not in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)

y_test_np = df_charged['y'].to_numpy()
y_pred_test_np = df_charged['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_charged_methanol.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Methanol '+r'$ΔG^{ionic}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='lower right')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor((0, 0, 0, 0))
fig.savefig(f'{root_dir}/charged_methanol_ind_plot_no_avg_with_avg_stats.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# charged water

df_charged = pd.read_csv(f'{root_dir}/full_df.csv')

for solvent in df_charged['Mol']:
    if 'ion' not in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)
    if 'methanol' in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)

y_test_np = df_charged['y'].to_numpy()
y_pred_test_np = df_charged['y_pred'].to_numpy()

with open(f'{root_dir}/avg_resample_test_stats_charged_water.txt', 'r') as f:
    lines = f.readlines()
    r2_line_test = [line for line in lines if line.startswith('r2_test')]
    rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
    bias_line_test = [line for line in lines if line.startswith('bias_test')]		
    sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
    r2 = (float(r2_line_test[0].split()[1]))
    rmsd = (float(rmsd_line_test[0].split()[1]))
    bias = (float(bias_line_test[0].split()[1]))
    sdep = (float(sdep_line_test[0].split()[1]))

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Water '+r'$ΔG^{ionic}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='lower right')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor((0, 0, 0, 0))
fig.savefig(f'{root_dir}/charged_water_ind_plot_no_avg_with_avg_stats.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# Individual solvent plots of average predictions per solute over all resamples. Stats calculated from verage predictions per solute over all resamples.

# charged methanol

df_charged = pd.read_csv(f'{root_dir}/avg_pred_over_resamples.csv')

for solvent in df_charged['Mol']:
    if 'ion' not in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)
    if 'methanol' not in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)

y_test_np = df_charged['y'].to_numpy()
y_pred_test_np = df_charged['y_pred'].to_numpy()

r2_test = r2_score(y_test_np, y_pred_test_np)
rmsd_test = (mean_squared_error(y_test_np, y_pred_test_np))**0.5
bias_test = np.mean(y_pred_test_np - y_test_np)
sdep_test = (np.mean((y_pred_test_np - y_test_np - bias_test)**2))**0.5
r2_test = '{:.3f}'.format(r2_test)
rmsd_test = '{:.3f}'.format(rmsd_test)
bias_test = '{:.3f}'.format(bias_test)
sdep_test = '{:.3f}'.format(sdep_test)
results_file = open(f'{root_dir}/avg_pred_over_resamples_test_stats_charged.txt', 'w')
results_file.write(f'r2_test: {r2_test}')
results_file.write('\n')
results_file.write(f'rmsd_test: {rmsd_test}')
results_file.write('\n')
results_file.write(f'bias_test: {bias_test}')
results_file.write('\n')
results_file.write(f'sdep_test: {sdep_test}')
results_file.close()

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Methanol '+r'$ΔG^{ionic}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/charged_methanol_ind_plot_avg_with_avg_stats.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# charged water

df_charged = pd.read_csv(f'{root_dir}/avg_pred_over_resamples.csv')

for solvent in df_charged['Mol']:
    if 'ion' not in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)
    if 'methanol' in solvent:
        df_charged.drop(df_charged.loc[df_charged['Mol'] == solvent].index, inplace=True)

y_test_np = df_charged['y'].to_numpy()
y_pred_test_np = df_charged['y_pred'].to_numpy()

r2_test = r2_score(y_test_np, y_pred_test_np)
rmsd_test = (mean_squared_error(y_test_np, y_pred_test_np))**0.5
bias_test = np.mean(y_pred_test_np - y_test_np)
sdep_test = (np.mean((y_pred_test_np - y_test_np - bias_test)**2))**0.5
r2_test = '{:.3f}'.format(r2_test)
rmsd_test = '{:.3f}'.format(rmsd_test)
bias_test = '{:.3f}'.format(bias_test)
sdep_test = '{:.3f}'.format(sdep_test)
results_file = open(f'{root_dir}/avg_pred_over_resamples_test_stats_charged.txt', 'w')
results_file.write(f'r2_test: {r2_test}')
results_file.write('\n')
results_file.write(f'rmsd_test: {rmsd_test}')
results_file.write('\n')
results_file.write(f'bias_test: {bias_test}')
results_file.write('\n')
results_file.write(f'sdep_test: {sdep_test}')
results_file.close()

fig, ax = plt.subplots()   

# Line of best fit
try:
    a, b = np.polyfit(y_test_np, y_pred_test_np, 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    name = plt.plot([], [], ' ', label='Test set: '+'Water '+r'$ΔG^{ionic}_{solv}$')
    r2 = plt.plot([], [], ' ', label=f'$R^{2}$ : {r2}')
    rmsd = plt.plot([], [], ' ', label=f'RMSD : {rmsd}')
    bias = plt.plot([], [], ' ', label=f'Bias : {bias}')
    sdep = plt.plot([], [], ' ', label=f'SDEP : {sdep}')
    line = plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(y_test_np, y_pred_test_np)
    plt.plot(y_test_np, a * y_test_np + b, color='purple')
    order = [0,1,2,3,4,5]
except NameError:
    pass

# x=y line
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Legend
plt.xlabel('$ΔG^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔG^{calc}_{solv}$ (kcal/mol)')
leg = plt.legend(handles=name+r2+rmsd+bias+sdep+line, loc='best')
ax = plt.gca().add_artist(leg)
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/charged_water_ind_plot_avg_with_avg_stats.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)













