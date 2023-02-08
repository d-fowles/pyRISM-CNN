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
from tensorflow import keras
from tensorflow.keras import Input, Model
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
    inputs = keras.Input(shape=(160,1))
    conv1d_1 = keras.layers.Conv1D(32, 
                        kernel_size=(3), 
                        strides=(2), 
                        padding='valid', 
                        activation='relu', 
                        input_shape=(160,1),
                        name = 'conv1d_1'
                        )(inputs)
    maxpooling_1 = keras.layers.MaxPooling1D((2), name = 'maxpooling_1')(conv1d_1)
    batchnorm_1 = keras.layers.BatchNormalization(name = 'batchnorm_1')(maxpooling_1)

    conv1d_2 = keras.layers.Conv1D(32, 
                        kernel_size=(3), 
                        strides=(2), 
                        padding='valid', 
                        activation='relu',
                        name = 'conv1d_2'
                        )(batchnorm_1)
    maxpooling_2 = keras.layers.MaxPooling1D((2), name = 'maxpooling_2')(conv1d_2)
    batchnorm_2 = keras.layers.BatchNormalization(name = 'batchnorm_2')(maxpooling_2)

    conv1d_3 = keras.layers.Conv1D(32, 
                        kernel_size=(3), 
                        strides=(2), 
                        padding='valid', 
                        activation='relu',
                        name = 'conv1d_3'
                        )(batchnorm_2)
    maxpooling_3 = keras.layers.MaxPooling1D((2), name = 'maxpooling_3')(conv1d_3)
    batchnorm_3 = keras.layers.BatchNormalization(name = 'batchnorm_3')(maxpooling_3)

    flatten = keras.layers.Flatten(name = 'flatten')(batchnorm_3)

    output_1 = keras.layers.Dense(1, name='enthalpy_pred')(flatten)
    output_2 = keras.layers.Dense(1, name='entropy_pred')(flatten)
    output_3 = keras.layers.Dense(1, name='free_energy_pred')(flatten)

    model = Model(inputs=inputs, outputs=[output_1, output_2, output_3])
    model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(lr=0.01), metrics = ["mse"])

    return model


# Initialise the model and print model summary:
model = build_model()

### Prepare input data with a unique id for each molecule and its corresponding descr.
input_dataset = pd.read_csv(f'{root_dir}/thermo_dataset_gf.csv')
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
r2_train_sum_resample_enthalpy = 0
rmsd_train_sum_resample_enthalpy = 0
bias_train_sum_resample_enthalpy = 0
sdep_train_sum_resample_enthalpy = 0

r2_train_sum_resample_entropy = 0
rmsd_train_sum_resample_entropy = 0
bias_train_sum_resample_entropy = 0
sdep_train_sum_resample_entropy = 0

r2_train_sum_resample_free_energy = 0
rmsd_train_sum_resample_free_energy = 0
bias_train_sum_resample_free_energy = 0
sdep_train_sum_resample_free_energy = 0

r2_val_sum_resample_enthalpy = 0
rmsd_val_sum_resample_enthalpy = 0
bias_val_sum_resample_enthalpy = 0
sdep_val_sum_resample_enthalpy = 0

r2_val_sum_resample_entropy = 0
rmsd_val_sum_resample_entropy = 0
bias_val_sum_resample_entropy = 0
sdep_val_sum_resample_entropy = 0

r2_val_sum_resample_free_energy = 0
rmsd_val_sum_resample_free_energy = 0
bias_val_sum_resample_free_energy = 0
sdep_val_sum_resample_free_energy = 0

r2_test_sum_resample_enthalpy = 0
rmsd_test_sum_resample_enthalpy = 0
bias_test_sum_resample_enthalpy = 0
sdep_test_sum_resample_enthalpy = 0

r2_test_sum_resample_entropy = 0
rmsd_test_sum_resample_entropy = 0
bias_test_sum_resample_entropy = 0
sdep_test_sum_resample_entropy = 0

r2_test_sum_resample_free_energy = 0
rmsd_test_sum_resample_free_energy = 0
bias_test_sum_resample_free_energy = 0
sdep_test_sum_resample_free_energy = 0

# Lists to save resamples predictions
all_preds_train_resample_enthalpy = []
all_preds_val_resample_enthalpy = []
all_preds_test_resample_enthalpy = []

all_preds_train_resample_entropy = []
all_preds_val_resample_entropy = []
all_preds_test_resample_entropy = []

all_preds_train_resample_free_energy = []
all_preds_val_resample_free_energy = []
all_preds_test_resample_free_energy = []

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
    r2_train_sum_cv_enthalpy = 0
    rmsd_train_sum_cv_enthalpy = 0
    bias_train_sum_cv_enthalpy = 0
    sdep_train_sum_cv_enthalpy = 0

    r2_val_sum_cv_enthalpy = 0
    rmsd_val_sum_cv_enthalpy = 0
    bias_val_sum_cv_enthalpy = 0
    sdep_val_sum_cv_enthalpy = 0

    r2_test_sum_cv_enthalpy = 0
    rmsd_test_sum_cv_enthalpy= 0
    bias_test_sum_cv_enthalpy = 0
    sdep_test_sum_cv_enthalpy = 0

    r2_train_sum_cv_entropy = 0
    rmsd_train_sum_cv_entropy = 0
    bias_train_sum_cv_entropy = 0
    sdep_train_sum_cv_entropy = 0

    r2_val_sum_cv_entropy = 0
    rmsd_val_sum_cv_entropy = 0
    bias_val_sum_cv_entropy = 0
    sdep_val_sum_cv_entropy = 0

    r2_test_sum_cv_entropy = 0
    rmsd_test_sum_cv_entropy= 0
    bias_test_sum_cv_entropy = 0
    sdep_test_sum_cv_entropy = 0

    r2_train_sum_cv_free_energy = 0
    rmsd_train_sum_cv_free_energy = 0
    bias_train_sum_cv_free_energy = 0
    sdep_train_sum_cv_free_energy = 0

    r2_val_sum_cv_free_energy = 0
    rmsd_val_sum_cv_free_energy = 0
    bias_val_sum_cv_free_energy = 0
    sdep_val_sum_cv_free_energy = 0

    r2_test_sum_cv_free_energy = 0
    rmsd_test_sum_cv_free_energy= 0
    bias_test_sum_cv_free_energy = 0
    sdep_test_sum_cv_free_energy = 0

    # Lists to save CV fold predictions
    all_preds_train_cv_enthalpy = []
    all_preds_val_cv_enthalpy = []
    all_preds_test_cv_enthalpy = []

    all_preds_train_cv_entropy = []
    all_preds_val_cv_entropy = []
    all_preds_test_cv_entropy = []

    all_preds_train_cv_free_energy = []
    all_preds_val_cv_free_energy = []
    all_preds_test_cv_free_energy = []

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
        y_train_1_fold = train_fold[['DH']].copy()
        y_train_2_fold = train_fold[['TDS']].copy()
        y_train_3_fold = train_fold[['DG']].copy()

        y_train_fold_output = pd.DataFrame()
        y_train_fold_output['Mol'] = train_fold['Mol']
        y_train_fold_output['Temp'] = train_fold['Temp']
        y_train_fold_output['y_enthalpy'] = train_fold['DH']
        y_train_fold_output['y_entropy'] = train_fold['TDS']
        y_train_fold_output['y_free_energy'] = train_fold['DG']
        y_train_fold_output.to_csv(f'resample_{resample}_fold_{i}_train_plot.csv', index=False)

        y_val_1_fold = val_fold[['DH']].copy()
        y_val_2_fold = val_fold[['TDS']].copy()
        y_val_3_fold = val_fold[['DG']].copy()

        y_val_fold_output = pd.DataFrame()
        y_val_fold_output['Mol'] = val_fold['Mol']
        y_val_fold_output['Temp'] = val_fold['Temp']
        y_val_fold_output['y_enthalpy'] = val_fold['DH']
        y_val_fold_output['y_entropy'] = val_fold['TDS']
        y_val_fold_output['y_free_energy'] = val_fold['DG']
        y_val_fold_output.to_csv(f'resample_{resample}_fold_{i}_val_plot.csv', index=False)

        y_test_1_fold = test_fold[['DH']].copy()
        y_test_2_fold = test_fold[['TDS']].copy()
        y_test_3_fold = test_fold[['DG']].copy()

        y_test_fold_output = pd.DataFrame()
        y_test_fold_output['Mol'] = test_fold['Mol']
        y_test_fold_output['Temp'] = test_fold['Temp']
        y_test_fold_output['y_enthalpy'] = test_fold['DH']
        y_test_fold_output['y_entropy'] = test_fold['TDS']
        y_test_fold_output['y_free_energy'] = test_fold['DG']
        y_test_fold_output.to_csv(f'resample_{resample}_fold_{i}_test_plot.csv', index=False)

        # Convert y data into required input shape
        y_train_1_cv = y_train_1_fold.to_numpy()
        y_train_1_cv = y_train_1_cv.reshape(y_train_1_cv.shape[0])
        y_train_2_cv = y_train_2_fold.to_numpy()
        y_train_2_cv = y_train_2_cv.reshape(y_train_2_cv.shape[0])
        y_train_3_cv = y_train_3_fold.to_numpy()
        y_train_3_cv = y_train_3_cv.reshape(y_train_3_cv.shape[0])

        y_val_1_cv = y_val_1_fold.to_numpy()
        y_val_1_cv = y_val_1_cv.reshape(y_val_1_cv.shape[0])
        y_val_2_cv = y_val_2_fold.to_numpy()
        y_val_2_cv = y_val_2_cv.reshape(y_val_2_cv.shape[0])
        y_val_3_cv = y_val_3_fold.to_numpy()
        y_val_3_cv = y_val_3_cv.reshape(y_val_3_cv.shape[0])

        y_test_1_cv = y_test_1_fold.to_numpy()
        y_test_1_cv = y_test_1_cv.reshape(y_test_1_cv.shape[0])
        y_test_2_cv = y_test_2_fold.to_numpy()
        y_test_2_cv = y_test_2_cv.reshape(y_test_2_cv.shape[0])
        y_test_3_cv = y_test_3_fold.to_numpy()
        y_test_3_cv = y_test_3_cv.reshape(y_test_3_cv.shape[0])

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
        history = model.fit(X_train_cv, [y_train_1_cv,y_train_2_cv,y_train_3_cv],
                            epochs = epochs,
                            verbose = 1,
                            validation_data =(X_val_cv, [y_val_1_cv,y_val_2_cv,y_val_3_cv]),
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
        y_pred_train_cv_enthalpy = y_pred_train_cv[0]
        y_pred_train_cv_entropy = y_pred_train_cv[1]
        y_pred_train_cv_free_energy = y_pred_train_cv[2]

        y_pred_val_cv = model.predict(X_val_cv)
        y_pred_val_cv_enthalpy = y_pred_val_cv[0]
        y_pred_val_cv_entropy = y_pred_val_cv[1]
        y_pred_val_cv_free_energy = y_pred_val_cv[2]

        y_pred_test_cv = model.predict(X_test_cv)
        y_pred_test_cv_enthalpy = y_pred_test_cv[0]
        y_pred_test_cv_entropy = y_pred_test_cv[1]
        y_pred_test_cv_free_energy = y_pred_test_cv[2]

        # Assess performace of model based on predictions:

        # Reduce dimensionality of predcition arrays
        y_pred_train_cv_enthalpy = y_pred_train_cv_enthalpy.squeeze()
        y_pred_train_cv_entropy = y_pred_train_cv_entropy.squeeze()
        y_pred_train_cv_free_energy = y_pred_train_cv_free_energy.squeeze()

        y_pred_val_cv_enthalpy = y_pred_val_cv_enthalpy.squeeze()
        y_pred_val_cv_entropy = y_pred_val_cv_entropy.squeeze()
        y_pred_val_cv_free_energy = y_pred_val_cv_free_energy.squeeze()

        y_pred_test_cv_enthalpy = y_pred_test_cv_enthalpy.squeeze()
        y_pred_test_cv_entropy = y_pred_test_cv_entropy.squeeze()
        y_pred_test_cv_free_energy = y_pred_test_cv_free_energy.squeeze()

        # Coefficient of determination
        r2_train_cv_enthalpy = r2_score(y_train_1_cv, y_pred_train_cv_enthalpy)
        r2_train_cv_entropy = r2_score(y_train_2_cv, y_pred_train_cv_entropy)
        r2_train_cv_free_energy = r2_score(y_train_3_cv, y_pred_train_cv_free_energy)

        r2_val_cv_enthalpy = r2_score(y_val_1_cv, y_pred_val_cv_enthalpy)
        r2_val_cv_entropy = r2_score(y_val_2_cv, y_pred_val_cv_entropy)
        r2_val_cv_free_energy = r2_score(y_val_3_cv, y_pred_val_cv_free_energy)

        r2_test_cv_enthalpy = r2_score(y_test_1_cv, y_pred_test_cv_enthalpy)
        r2_test_cv_entropy = r2_score(y_test_2_cv, y_pred_test_cv_entropy)
        r2_test_cv_free_energy = r2_score(y_test_3_cv, y_pred_test_cv_free_energy)

        # Root mean squared error
        rmsd_train_cv_enthalpy = (mean_squared_error(y_train_1_cv, y_pred_train_cv_enthalpy))**0.5
        rmsd_train_cv_entropy = (mean_squared_error(y_train_2_cv, y_pred_train_cv_entropy))**0.5
        rmsd_train_cv_free_energy = (mean_squared_error(y_train_3_cv, y_pred_train_cv_free_energy))**0.5

        rmsd_val_cv_enthalpy = (mean_squared_error(y_val_1_cv, y_pred_val_cv_enthalpy))**0.5
        rmsd_val_cv_entropy = (mean_squared_error(y_val_2_cv, y_pred_val_cv_entropy))**0.5
        rmsd_val_cv_free_energy = (mean_squared_error(y_val_3_cv, y_pred_val_cv_free_energy))**0.5

        rmsd_test_cv_enthalpy = (mean_squared_error(y_test_1_cv, y_pred_test_cv_enthalpy))**0.5
        rmsd_test_cv_entropy = (mean_squared_error(y_test_2_cv, y_pred_test_cv_entropy))**0.5
        rmsd_test_cv_free_energy = (mean_squared_error(y_test_3_cv, y_pred_test_cv_free_energy))**0.5

        # Bias
        bias_train_cv_enthalpy = np.mean(y_pred_train_cv_enthalpy - y_train_1_cv)
        bias_train_cv_entropy = np.mean(y_pred_train_cv_entropy - y_train_2_cv)
        bias_train_cv_free_energy = np.mean(y_pred_train_cv_free_energy - y_train_3_cv)

        bias_val_cv_enthalpy = np.mean(y_pred_val_cv_enthalpy - y_val_1_cv)
        bias_val_cv_entropy = np.mean(y_pred_val_cv_entropy - y_val_2_cv)
        bias_val_cv_free_energy = np.mean(y_pred_val_cv_free_energy - y_val_3_cv)

        bias_test_cv_enthalpy = np.mean(y_pred_test_cv_enthalpy - y_test_1_cv)
        bias_test_cv_entropy = np.mean(y_pred_test_cv_entropy - y_test_2_cv)
        bias_test_cv_free_energy = np.mean(y_pred_test_cv_free_energy - y_test_3_cv)

        # Standard deviation of the error of prediction
        sdep_train_cv_enthalpy = (np.mean((y_pred_train_cv_enthalpy - y_train_1_cv - bias_train_cv_enthalpy)**2))**0.5
        sdep_train_cv_entropy = (np.mean((y_pred_train_cv_entropy - y_train_2_cv - bias_train_cv_entropy)**2))**0.5
        sdep_train_cv_free_energy = (np.mean((y_pred_train_cv_free_energy - y_train_3_cv - bias_train_cv_free_energy)**2))**0.5

        sdep_val_cv_enthalpy = (np.mean((y_pred_val_cv_enthalpy - y_val_1_cv - bias_val_cv_enthalpy)**2))**0.5
        sdep_val_cv_entropy = (np.mean((y_pred_val_cv_entropy - y_val_2_cv - bias_val_cv_entropy)**2))**0.5
        sdep_val_cv_free_energy = (np.mean((y_pred_val_cv_free_energy - y_val_3_cv - bias_val_cv_free_energy)**2))**0.5

        sdep_test_cv_enthalpy = (np.mean(((y_pred_test_cv_enthalpy - y_test_1_cv) - bias_test_cv_enthalpy)**2))**0.5
        sdep_test_cv_entropy = (np.mean(((y_pred_test_cv_entropy - y_test_2_cv) - bias_test_cv_entropy)**2))**0.5
        sdep_test_cv_free_energy = (np.mean(((y_pred_test_cv_free_energy - y_test_3_cv) - bias_test_cv_free_energy)**2))**0.5
        
        # Save stats for each individual fold 
        r2_train_ind_cv_enthalpy = '{:.3f}'.format(r2_train_cv_enthalpy)
        rmsd_train_ind_cv_enthalpy = '{:.3f}'.format(rmsd_train_cv_enthalpy)
        bias_train_ind_cv_enthalpy = '{:.3f}'.format(bias_train_cv_enthalpy)
        sdep_train_ind_cv_enthalpy = '{:.3f}'.format(sdep_train_cv_enthalpy)

        r2_val_ind_cv_enthalpy = '{:.3f}'.format(r2_val_cv_enthalpy)
        rmsd_val_ind_cv_enthalpy = '{:.3f}'.format(rmsd_val_cv_enthalpy)
        bias_val_ind_cv_enthalpy = '{:.3f}'.format(bias_val_cv_enthalpy)
        sdep_val_ind_cv_enthalpy = '{:.3f}'.format(sdep_val_cv_enthalpy)

        r2_test_ind_cv_enthalpy = '{:.3f}'.format(r2_test_cv_enthalpy)
        rmsd_test_ind_cv_enthalpy = '{:.3f}'.format(rmsd_test_cv_enthalpy)
        bias_test_ind_cv_enthalpy = '{:.3f}'.format(bias_test_cv_enthalpy)
        sdep_test_ind_cv_enthalpy = '{:.3f}'.format(sdep_test_cv_enthalpy)

        results_file = open(f'fold_{i}_enthalpy_CV_stats.txt', 'w')
        results_file.write(f'r2_train: {r2_train_ind_cv_enthalpy}')
        results_file.write('\n')
        results_file.write(f'rmsd_train: {rmsd_train_ind_cv_enthalpy}')
        results_file.write('\n')
        results_file.write(f'bias_train: {bias_train_ind_cv_enthalpy}')
        results_file.write('\n')
        results_file.write(f'sdep_train: {sdep_train_ind_cv_enthalpy}')
        results_file.write('\n')
        results_file.write('\n')
        results_file.write(f'r2_val: {r2_val_ind_cv_enthalpy}')
        results_file.write('\n')
        results_file.write(f'rmsd_val: {rmsd_val_ind_cv_enthalpy}')
        results_file.write('\n')
        results_file.write(f'bias_val: {bias_val_ind_cv_enthalpy}')
        results_file.write('\n')
        results_file.write(f'sdep_val: {sdep_val_ind_cv_enthalpy}')
        results_file.write('\n')
        results_file.write('\n')
        results_file.write(f'r2_test: {r2_test_ind_cv_enthalpy}')
        results_file.write('\n')
        results_file.write(f'rmsd_test: {rmsd_test_ind_cv_enthalpy}')
        results_file.write('\n')
        results_file.write(f'bias_test: {bias_test_ind_cv_enthalpy}')
        results_file.write('\n')
        results_file.write(f'sdep_test: {sdep_test_ind_cv_enthalpy}')
        results_file.close()

        r2_train_ind_cv_entropy = '{:.3f}'.format(r2_train_cv_entropy)
        rmsd_train_ind_cv_entropy = '{:.3f}'.format(rmsd_train_cv_entropy)
        bias_train_ind_cv_entropy = '{:.3f}'.format(bias_train_cv_entropy)
        sdep_train_ind_cv_entropy = '{:.3f}'.format(sdep_train_cv_entropy)

        r2_val_ind_cv_entropy = '{:.3f}'.format(r2_val_cv_entropy)
        rmsd_val_ind_cv_entropy = '{:.3f}'.format(rmsd_val_cv_entropy)
        bias_val_ind_cv_entropy = '{:.3f}'.format(bias_val_cv_entropy)
        sdep_val_ind_cv_entropy = '{:.3f}'.format(sdep_val_cv_entropy)

        r2_test_ind_cv_entropy = '{:.3f}'.format(r2_test_cv_entropy)
        rmsd_test_ind_cv_entropy = '{:.3f}'.format(rmsd_test_cv_entropy)
        bias_test_ind_cv_entropy = '{:.3f}'.format(bias_test_cv_entropy)
        sdep_test_ind_cv_entropy = '{:.3f}'.format(sdep_test_cv_entropy)

        results_file = open(f'fold_{i}_entropy_CV_stats.txt', 'w')
        results_file.write(f'r2_train: {r2_train_ind_cv_entropy}')
        results_file.write('\n')
        results_file.write(f'rmsd_train: {rmsd_train_ind_cv_entropy}')
        results_file.write('\n')
        results_file.write(f'bias_train: {bias_train_ind_cv_entropy}')
        results_file.write('\n')
        results_file.write(f'sdep_train: {sdep_train_ind_cv_entropy}')
        results_file.write('\n')
        results_file.write('\n')
        results_file.write(f'r2_val: {r2_val_ind_cv_entropy}')
        results_file.write('\n')
        results_file.write(f'rmsd_val: {rmsd_val_ind_cv_entropy}')
        results_file.write('\n')
        results_file.write(f'bias_val: {bias_val_ind_cv_entropy}')
        results_file.write('\n')
        results_file.write(f'sdep_val: {sdep_val_ind_cv_entropy}')
        results_file.write('\n')
        results_file.write('\n')
        results_file.write(f'r2_test: {r2_test_ind_cv_entropy}')
        results_file.write('\n')
        results_file.write(f'rmsd_test: {rmsd_test_ind_cv_entropy}')
        results_file.write('\n')
        results_file.write(f'bias_test: {bias_test_ind_cv_entropy}')
        results_file.write('\n')
        results_file.write(f'sdep_test: {sdep_test_ind_cv_entropy}')
        results_file.close()

        r2_train_ind_cv_free_energy = '{:.3f}'.format(r2_train_cv_free_energy)
        rmsd_train_ind_cv_free_energy = '{:.3f}'.format(rmsd_train_cv_free_energy)
        bias_train_ind_cv_free_energy = '{:.3f}'.format(bias_train_cv_free_energy)
        sdep_train_ind_cv_free_energy = '{:.3f}'.format(sdep_train_cv_free_energy)

        r2_val_ind_cv_free_energy = '{:.3f}'.format(r2_val_cv_free_energy)
        rmsd_val_ind_cv_free_energy = '{:.3f}'.format(rmsd_val_cv_free_energy)
        bias_val_ind_cv_free_energy = '{:.3f}'.format(bias_val_cv_free_energy)
        sdep_val_ind_cv_free_energy = '{:.3f}'.format(sdep_val_cv_free_energy)

        r2_test_ind_cv_free_energy = '{:.3f}'.format(r2_test_cv_free_energy)
        rmsd_test_ind_cv_free_energy = '{:.3f}'.format(rmsd_test_cv_free_energy)
        bias_test_ind_cv_free_energy = '{:.3f}'.format(bias_test_cv_free_energy)
        sdep_test_ind_cv_free_energy = '{:.3f}'.format(sdep_test_cv_free_energy)

        results_file = open(f'fold_{i}_free_energy_CV_stats.txt', 'w')
        results_file.write(f'r2_train: {r2_train_ind_cv_free_energy}')
        results_file.write('\n')
        results_file.write(f'rmsd_train: {rmsd_train_ind_cv_free_energy}')
        results_file.write('\n')
        results_file.write(f'bias_train: {bias_train_ind_cv_free_energy}')
        results_file.write('\n')
        results_file.write(f'sdep_train: {sdep_train_ind_cv_free_energy}')
        results_file.write('\n')
        results_file.write('\n')
        results_file.write(f'r2_val: {r2_val_ind_cv_free_energy}')
        results_file.write('\n')
        results_file.write(f'rmsd_val: {rmsd_val_ind_cv_free_energy}')
        results_file.write('\n')
        results_file.write(f'bias_val: {bias_val_ind_cv_free_energy}')
        results_file.write('\n')
        results_file.write(f'sdep_val: {sdep_val_ind_cv_free_energy}')
        results_file.write('\n')
        results_file.write('\n')
        results_file.write(f'r2_test: {r2_test_ind_cv_free_energy}')
        results_file.write('\n')
        results_file.write(f'rmsd_test: {rmsd_test_ind_cv_free_energy}')
        results_file.write('\n')
        results_file.write(f'bias_test: {bias_test_ind_cv_free_energy}')
        results_file.write('\n')
        results_file.write(f'sdep_test: {sdep_test_ind_cv_free_energy}')
        results_file.close()

        # Save running sum of results:
        r2_train_sum_cv_enthalpy += r2_train_cv_enthalpy
        rmsd_train_sum_cv_enthalpy += rmsd_train_cv_enthalpy
        bias_train_sum_cv_enthalpy += bias_train_cv_enthalpy
        sdep_train_sum_cv_enthalpy += sdep_train_cv_enthalpy
 
        r2_val_sum_cv_enthalpy += r2_val_cv_enthalpy
        rmsd_val_sum_cv_enthalpy += rmsd_val_cv_enthalpy
        bias_val_sum_cv_enthalpy += bias_val_cv_enthalpy
        sdep_val_sum_cv_enthalpy += sdep_val_cv_enthalpy

        r2_test_sum_cv_enthalpy += r2_test_cv_enthalpy
        rmsd_test_sum_cv_enthalpy += rmsd_test_cv_enthalpy
        bias_test_sum_cv_enthalpy += bias_test_cv_enthalpy
        sdep_test_sum_cv_enthalpy += sdep_test_cv_enthalpy

        r2_train_sum_cv_entropy += r2_train_cv_entropy
        rmsd_train_sum_cv_entropy += rmsd_train_cv_entropy
        bias_train_sum_cv_entropy += bias_train_cv_entropy
        sdep_train_sum_cv_entropy += sdep_train_cv_entropy
 
        r2_val_sum_cv_entropy += r2_val_cv_entropy
        rmsd_val_sum_cv_entropy += rmsd_val_cv_entropy
        bias_val_sum_cv_entropy += bias_val_cv_entropy
        sdep_val_sum_cv_entropy += sdep_val_cv_entropy

        r2_test_sum_cv_entropy += r2_test_cv_entropy
        rmsd_test_sum_cv_entropy += rmsd_test_cv_entropy
        bias_test_sum_cv_entropy += bias_test_cv_entropy
        sdep_test_sum_cv_entropy += sdep_test_cv_entropy

        r2_train_sum_cv_free_energy += r2_train_cv_free_energy
        rmsd_train_sum_cv_free_energy += rmsd_train_cv_free_energy
        bias_train_sum_cv_free_energy += bias_train_cv_free_energy
        sdep_train_sum_cv_free_energy += sdep_train_cv_free_energy
 
        r2_val_sum_cv_free_energy += r2_val_cv_free_energy
        rmsd_val_sum_cv_free_energy += rmsd_val_cv_free_energy
        bias_val_sum_cv_free_energy += bias_val_cv_free_energy
        sdep_val_sum_cv_free_energy += sdep_val_cv_free_energy

        r2_test_sum_cv_free_energy += r2_test_cv_free_energy
        rmsd_test_sum_cv_free_energy += rmsd_test_cv_free_energy
        bias_test_sum_cv_free_energy += bias_test_cv_free_energy
        sdep_test_sum_cv_free_energy += sdep_test_cv_free_energy

        # Save individual predictions for fold in dataframes
        y_pred_train_cv_enthalpy = y_pred_train_cv_enthalpy.reshape(y_pred_train_cv_enthalpy.shape[0])
        y_pred_train_cv_enthalpy = pd.DataFrame(y_pred_train_cv_enthalpy, columns=[f'fold_{i}'])
        y_pred_train_cv_enthalpy_labeled = pd.concat([train_fold['Mol'], train_fold['Temp'], train_fold['DH'], y_pred_train_cv_enthalpy], axis=1)
        y_pred_train_cv_enthalpy_labeled[''] = ''
        all_preds_train_cv_enthalpy.append(y_pred_train_cv_enthalpy_labeled)

        y_pred_val_cv_enthalpy = y_pred_val_cv_enthalpy.reshape(y_pred_val_cv_enthalpy.shape[0])
        y_pred_val_cv_enthalpy = pd.DataFrame(y_pred_val_cv_enthalpy, columns=[f'fold_{i}'])
        y_pred_val_cv_enthalpy_labeled = pd.concat([val_fold['Mol'], val_fold['Temp'], val_fold['DH'], y_pred_val_cv_enthalpy], axis=1)
        y_pred_val_cv_enthalpy_labeled[''] = ''
        all_preds_val_cv_enthalpy.append(y_pred_val_cv_enthalpy_labeled)

        y_pred_test_cv_enthalpy = y_pred_test_cv_enthalpy.reshape(y_pred_test_cv_enthalpy.shape[0])
        y_pred_test_cv_enthalpy = pd.DataFrame(y_pred_test_cv_enthalpy, columns=[f'fold_{i}'])
        y_pred_test_cv_enthalpy_labeled = pd.concat([test_fold['Mol'], test_fold['Temp'], test_fold['DH'], y_pred_test_cv_enthalpy], axis=1)
        y_pred_test_cv_enthalpy_labeled[''] = ''
        all_preds_test_cv_enthalpy.append(y_pred_test_cv_enthalpy_labeled)

        y_pred_train_cv_entropy = y_pred_train_cv_entropy.reshape(y_pred_train_cv_entropy.shape[0])
        y_pred_train_cv_entropy = pd.DataFrame(y_pred_train_cv_entropy, columns=[f'fold_{i}'])
        y_pred_train_cv_entropy_labeled = pd.concat([train_fold['Mol'], train_fold['Temp'], train_fold['TDS'], y_pred_train_cv_entropy], axis=1)
        y_pred_train_cv_entropy_labeled[''] = ''
        all_preds_train_cv_entropy.append(y_pred_train_cv_entropy_labeled)

        y_pred_val_cv_entropy = y_pred_val_cv_entropy.reshape(y_pred_val_cv_entropy.shape[0])
        y_pred_val_cv_entropy = pd.DataFrame(y_pred_val_cv_entropy, columns=[f'fold_{i}'])
        y_pred_val_cv_entropy_labeled = pd.concat([val_fold['Mol'], val_fold['Temp'], val_fold['TDS'], y_pred_val_cv_entropy], axis=1)
        y_pred_val_cv_entropy_labeled[''] = ''
        all_preds_val_cv_entropy.append(y_pred_val_cv_entropy_labeled)

        y_pred_test_cv_entropy = y_pred_test_cv_entropy.reshape(y_pred_test_cv_entropy.shape[0])
        y_pred_test_cv_entropy = pd.DataFrame(y_pred_test_cv_entropy, columns=[f'fold_{i}'])
        y_pred_test_cv_entropy_labeled = pd.concat([test_fold['Mol'], test_fold['Temp'], test_fold['TDS'], y_pred_test_cv_entropy], axis=1)
        y_pred_test_cv_entropy_labeled[''] = ''
        all_preds_test_cv_entropy.append(y_pred_test_cv_entropy_labeled)

        y_pred_train_cv_free_energy = y_pred_train_cv_free_energy.reshape(y_pred_train_cv_free_energy.shape[0])
        y_pred_train_cv_free_energy = pd.DataFrame(y_pred_train_cv_free_energy, columns=[f'fold_{i}'])
        y_pred_train_cv_free_energy_labeled = pd.concat([train_fold['Mol'], train_fold['Temp'], train_fold['DG'], y_pred_train_cv_free_energy], axis=1)
        y_pred_train_cv_free_energy_labeled[''] = ''
        all_preds_train_cv_free_energy.append(y_pred_train_cv_free_energy_labeled)

        y_pred_val_cv_free_energy = y_pred_val_cv_free_energy.reshape(y_pred_val_cv_free_energy.shape[0])
        y_pred_val_cv_free_energy = pd.DataFrame(y_pred_val_cv_free_energy, columns=[f'fold_{i}'])
        y_pred_val_cv_free_energy_labeled = pd.concat([val_fold['Mol'], val_fold['Temp'], val_fold['DG'], y_pred_val_cv_free_energy], axis=1)
        y_pred_val_cv_free_energy_labeled[''] = ''
        all_preds_val_cv_free_energy.append(y_pred_val_cv_free_energy_labeled)

        y_pred_test_cv_free_energy = y_pred_test_cv_free_energy.reshape(y_pred_test_cv_free_energy.shape[0])
        y_pred_test_cv_free_energy = pd.DataFrame(y_pred_test_cv_free_energy, columns=[f'fold_{i}'])
        y_pred_test_cv_free_energy_labeled = pd.concat([test_fold['Mol'], test_fold['Temp'], test_fold['DG'], y_pred_test_cv_free_energy], axis=1)
        y_pred_test_cv_free_energy_labeled[''] = ''
        all_preds_test_cv_free_energy.append(y_pred_test_cv_free_energy_labeled)

    # Average results over CV folds:
    r2_train_av_cv_enthalpy = r2_train_sum_cv_enthalpy/no_folds
    rmsd_train_av_cv_enthalpy = rmsd_train_sum_cv_enthalpy/no_folds
    bias_train_av_cv_enthalpy = bias_train_sum_cv_enthalpy/no_folds
    sdep_train_av_cv_enthalpy = sdep_train_sum_cv_enthalpy/no_folds

    r2_train_av_cv_enthalpy = '{:.3f}'.format(r2_train_av_cv_enthalpy)
    rmsd_train_av_cv_enthalpy = '{:.3f}'.format(rmsd_train_av_cv_enthalpy)
    bias_train_av_cv_enthalpy = '{:.3f}'.format(bias_train_av_cv_enthalpy)
    sdep_train_av_cv_enthalpy = '{:.3f}'.format(sdep_train_av_cv_enthalpy)

    r2_val_av_cv_enthalpy = r2_val_sum_cv_enthalpy/no_folds
    rmsd_val_av_cv_enthalpy = rmsd_val_sum_cv_enthalpy/no_folds
    bias_val_av_cv_enthalpy = bias_val_sum_cv_enthalpy/no_folds
    sdep_val_av_cv_enthalpy = sdep_val_sum_cv_enthalpy/no_folds

    r2_val_av_cv_enthalpy = '{:.3f}'.format(r2_val_av_cv_enthalpy)
    rmsd_val_av_cv_enthalpy = '{:.3f}'.format(rmsd_val_av_cv_enthalpy)
    bias_val_av_cv_enthalpy = '{:.3f}'.format(bias_val_av_cv_enthalpy)
    sdep_val_av_cv_enthalpy = '{:.3f}'.format(sdep_val_av_cv_enthalpy)

    r2_test_av_cv_enthalpy = r2_test_sum_cv_enthalpy/no_folds
    rmsd_test_av_cv_enthalpy = rmsd_test_sum_cv_enthalpy/no_folds
    bias_test_av_cv_enthalpy = bias_test_sum_cv_enthalpy/no_folds
    sdep_test_av_cv_enthalpy = sdep_test_sum_cv_enthalpy/no_folds

    r2_test_av_cv_enthalpy = '{:.3f}'.format(r2_test_av_cv_enthalpy)
    rmsd_test_av_cv_enthalpy = '{:.3f}'.format(rmsd_test_av_cv_enthalpy)
    bias_test_av_cv_enthalpy = '{:.3f}'.format(bias_test_av_cv_enthalpy)
    sdep_test_av_cv_enthalpy = '{:.3f}'.format(sdep_test_av_cv_enthalpy)
    #
    r2_train_av_cv_entropy = r2_train_sum_cv_entropy/no_folds
    rmsd_train_av_cv_entropy = rmsd_train_sum_cv_entropy/no_folds
    bias_train_av_cv_entropy = bias_train_sum_cv_entropy/no_folds
    sdep_train_av_cv_entropy = sdep_train_sum_cv_entropy/no_folds

    r2_train_av_cv_entropy = '{:.3f}'.format(r2_train_av_cv_entropy)
    rmsd_train_av_cv_entropy = '{:.3f}'.format(rmsd_train_av_cv_entropy)
    bias_train_av_cv_entropy = '{:.3f}'.format(bias_train_av_cv_entropy)
    sdep_train_av_cv_entropy = '{:.3f}'.format(sdep_train_av_cv_entropy)

    r2_val_av_cv_entropy = r2_val_sum_cv_entropy/no_folds
    rmsd_val_av_cv_entropy = rmsd_val_sum_cv_entropy/no_folds
    bias_val_av_cv_entropy = bias_val_sum_cv_entropy/no_folds
    sdep_val_av_cv_entropy = sdep_val_sum_cv_entropy/no_folds

    r2_val_av_cv_entropy = '{:.3f}'.format(r2_val_av_cv_entropy)
    rmsd_val_av_cv_entropy = '{:.3f}'.format(rmsd_val_av_cv_entropy)
    bias_val_av_cv_entropy = '{:.3f}'.format(bias_val_av_cv_entropy)
    sdep_val_av_cv_entropy = '{:.3f}'.format(sdep_val_av_cv_entropy)

    r2_test_av_cv_entropy = r2_test_sum_cv_entropy/no_folds
    rmsd_test_av_cv_entropy = rmsd_test_sum_cv_entropy/no_folds
    bias_test_av_cv_entropy = bias_test_sum_cv_entropy/no_folds
    sdep_test_av_cv_entropy = sdep_test_sum_cv_entropy/no_folds

    r2_test_av_cv_entropy = '{:.3f}'.format(r2_test_av_cv_entropy)
    rmsd_test_av_cv_entropy = '{:.3f}'.format(rmsd_test_av_cv_entropy)
    bias_test_av_cv_entropy = '{:.3f}'.format(bias_test_av_cv_entropy)
    sdep_test_av_cv_entropy = '{:.3f}'.format(sdep_test_av_cv_entropy)
    #
    r2_train_av_cv_free_energy = r2_train_sum_cv_free_energy/no_folds
    rmsd_train_av_cv_free_energy = rmsd_train_sum_cv_free_energy/no_folds
    bias_train_av_cv_free_energy = bias_train_sum_cv_free_energy/no_folds
    sdep_train_av_cv_free_energy = sdep_train_sum_cv_free_energy/no_folds

    r2_train_av_cv_free_energy = '{:.3f}'.format(r2_train_av_cv_free_energy)
    rmsd_train_av_cv_free_energy = '{:.3f}'.format(rmsd_train_av_cv_free_energy)
    bias_train_av_cv_free_energy = '{:.3f}'.format(bias_train_av_cv_free_energy)
    sdep_train_av_cv_free_energy = '{:.3f}'.format(sdep_train_av_cv_free_energy)

    r2_val_av_cv_free_energy = r2_val_sum_cv_free_energy/no_folds
    rmsd_val_av_cv_free_energy = rmsd_val_sum_cv_free_energy/no_folds
    bias_val_av_cv_free_energy = bias_val_sum_cv_free_energy/no_folds
    sdep_val_av_cv_free_energy = sdep_val_sum_cv_free_energy/no_folds

    r2_val_av_cv_free_energy = '{:.3f}'.format(r2_val_av_cv_free_energy)
    rmsd_val_av_cv_free_energy = '{:.3f}'.format(rmsd_val_av_cv_free_energy)
    bias_val_av_cv_free_energy = '{:.3f}'.format(bias_val_av_cv_free_energy)
    sdep_val_av_cv_free_energy = '{:.3f}'.format(sdep_val_av_cv_free_energy)

    r2_test_av_cv_free_energy = r2_test_sum_cv_free_energy/no_folds
    rmsd_test_av_cv_free_energy = rmsd_test_sum_cv_free_energy/no_folds
    bias_test_av_cv_free_energy = bias_test_sum_cv_free_energy/no_folds
    sdep_test_av_cv_free_energy = sdep_test_sum_cv_free_energy/no_folds

    r2_test_av_cv_free_energy = '{:.3f}'.format(r2_test_av_cv_free_energy)
    rmsd_test_av_cv_free_energy = '{:.3f}'.format(rmsd_test_av_cv_free_energy)
    bias_test_av_cv_free_energy = '{:.3f}'.format(bias_test_av_cv_free_energy)
    sdep_test_av_cv_free_energy = '{:.3f}'.format(sdep_test_av_cv_free_energy)

    # Write average results to a file:
    results_file = open(f'{root_dir}/resample_{resample}/resample_{resample}_CV_avg_stats_enthalpy.txt', 'w')
    results_file.write(f'r2_train: {r2_train_av_cv_enthalpy}')
    results_file.write('\n')
    results_file.write(f'rmsd_train: {rmsd_train_av_cv_enthalpy}')
    results_file.write('\n')
    results_file.write(f'bias_train: {bias_train_av_cv_enthalpy}')
    results_file.write('\n')
    results_file.write(f'sdep_train: {sdep_train_av_cv_enthalpy}')
    results_file.write('\n')
    results_file.write('\n')
    results_file.write(f'r2_val: {r2_val_av_cv_enthalpy}')
    results_file.write('\n')
    results_file.write(f'rmsd_val: {rmsd_val_av_cv_enthalpy}')
    results_file.write('\n')
    results_file.write(f'bias_val: {bias_val_av_cv_enthalpy}')
    results_file.write('\n')
    results_file.write(f'sdep_val: {sdep_val_av_cv_enthalpy}')
    results_file.write('\n')
    results_file.write('\n')
    results_file.write(f'r2_test: {r2_test_av_cv_enthalpy}')
    results_file.write('\n')
    results_file.write(f'rmsd_test: {rmsd_test_av_cv_enthalpy}')
    results_file.write('\n')
    results_file.write(f'bias_test: {bias_test_av_cv_enthalpy}')
    results_file.write('\n')
    results_file.write(f'sdep_test: {sdep_test_av_cv_enthalpy}')
    results_file.close()

    results_file = open(f'{root_dir}/resample_{resample}/resample_{resample}_CV_avg_stats_entropy.txt', 'w')
    results_file.write(f'r2_train: {r2_train_av_cv_entropy}')
    results_file.write('\n')
    results_file.write(f'rmsd_train: {rmsd_train_av_cv_entropy}')
    results_file.write('\n')
    results_file.write(f'bias_train: {bias_train_av_cv_entropy}')
    results_file.write('\n')
    results_file.write(f'sdep_train: {sdep_train_av_cv_entropy}')
    results_file.write('\n')
    results_file.write('\n')
    results_file.write(f'r2_val: {r2_val_av_cv_entropy}')
    results_file.write('\n')
    results_file.write(f'rmsd_val: {rmsd_val_av_cv_entropy}')
    results_file.write('\n')
    results_file.write(f'bias_val: {bias_val_av_cv_entropy}')
    results_file.write('\n')
    results_file.write(f'sdep_val: {sdep_val_av_cv_entropy}')
    results_file.write('\n')
    results_file.write('\n')
    results_file.write(f'r2_test: {r2_test_av_cv_entropy}')
    results_file.write('\n')
    results_file.write(f'rmsd_test: {rmsd_test_av_cv_entropy}')
    results_file.write('\n')
    results_file.write(f'bias_test: {bias_test_av_cv_entropy}')
    results_file.write('\n')
    results_file.write(f'sdep_test: {sdep_test_av_cv_entropy}')
    results_file.close()

    results_file = open(f'{root_dir}/resample_{resample}/resample_{resample}_CV_avg_stats_free_energy.txt', 'w')
    results_file.write(f'r2_train: {r2_train_av_cv_free_energy}')
    results_file.write('\n')
    results_file.write(f'rmsd_train: {rmsd_train_av_cv_free_energy}')
    results_file.write('\n')
    results_file.write(f'bias_train: {bias_train_av_cv_free_energy}')
    results_file.write('\n')
    results_file.write(f'sdep_train: {sdep_train_av_cv_free_energy}')
    results_file.write('\n')
    results_file.write('\n')
    results_file.write(f'r2_val: {r2_val_av_cv_free_energy}')
    results_file.write('\n')
    results_file.write(f'rmsd_val: {rmsd_val_av_cv_free_energy}')
    results_file.write('\n')
    results_file.write(f'bias_val: {bias_val_av_cv_free_energy}')
    results_file.write('\n')
    results_file.write(f'sdep_val: {sdep_val_av_cv_free_energy}')
    results_file.write('\n')
    results_file.write('\n')
    results_file.write(f'r2_test: {r2_test_av_cv_free_energy}')
    results_file.write('\n')
    results_file.write(f'rmsd_test: {rmsd_test_av_cv_free_energy}')
    results_file.write('\n')
    results_file.write(f'bias_test: {bias_test_av_cv_free_energy}')
    results_file.write('\n')
    results_file.write(f'sdep_test: {sdep_test_av_cv_free_energy}')
    results_file.close()

    # Save all individual predictions to train file:
    all_preds_train_cv_enthalpy = pd.concat(all_preds_train_cv_enthalpy, axis=1)
    all_preds_train_cv_enthalpy['Mol'].replace('', np.nan, inplace=True)
    all_preds_train_cv_enthalpy = all_preds_train_cv_enthalpy.apply(lambda x: pd.Series(x.dropna().values))
    all_preds_train_cv_enthalpy.to_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_train_CV_avg_enthalpy.csv')

    all_preds_val_cv_enthalpy = pd.concat(all_preds_val_cv_enthalpy, axis=1)
    all_preds_val_cv_enthalpy['Mol'].replace('', np.nan, inplace=True)
    all_preds_val_cv_enthalpy = all_preds_val_cv_enthalpy.apply(lambda x: pd.Series(x.dropna().values))
    all_preds_val_cv_enthalpy.to_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_val_CV_avg_enthalpy.csv')

    all_preds_test_cv_enthalpy = pd.concat(all_preds_test_cv_enthalpy, axis=1)
    all_preds_test_cv_enthalpy['Mol'].replace('', np.nan, inplace=True)
    all_preds_test_cv_enthalpy = all_preds_test_cv_enthalpy.apply(lambda x: pd.Series(x.dropna().values))
    all_preds_test_cv_enthalpy.to_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_test_CV_avg_enthalpy.csv')

    all_preds_train_cv_entropy = pd.concat(all_preds_train_cv_entropy, axis=1)
    all_preds_train_cv_entropy['Mol'].replace('', np.nan, inplace=True)
    all_preds_train_cv_entropy = all_preds_train_cv_entropy.apply(lambda x: pd.Series(x.dropna().values))
    all_preds_train_cv_entropy.to_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_train_CV_avg_entropy.csv')

    all_preds_val_cv_entropy = pd.concat(all_preds_val_cv_entropy, axis=1)
    all_preds_val_cv_entropy['Mol'].replace('', np.nan, inplace=True)
    all_preds_val_cv_entropy = all_preds_val_cv_entropy.apply(lambda x: pd.Series(x.dropna().values))
    all_preds_val_cv_entropy.to_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_val_CV_avg_entropy.csv')

    all_preds_test_cv_entropy = pd.concat(all_preds_test_cv_entropy, axis=1)
    all_preds_test_cv_entropy['Mol'].replace('', np.nan, inplace=True)
    all_preds_test_cv_entropy = all_preds_test_cv_entropy.apply(lambda x: pd.Series(x.dropna().values))
    all_preds_test_cv_entropy.to_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_test_CV_avg_entropy.csv')

    all_preds_train_cv_free_energy = pd.concat(all_preds_train_cv_free_energy, axis=1)
    all_preds_train_cv_free_energy['Mol'].replace('', np.nan, inplace=True)
    all_preds_train_cv_free_energy = all_preds_train_cv_free_energy.apply(lambda x: pd.Series(x.dropna().values))
    all_preds_train_cv_free_energy.to_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_train_CV_avg_free_energy.csv')

    all_preds_val_cv_free_energy = pd.concat(all_preds_val_cv_free_energy, axis=1)
    all_preds_val_cv_free_energy['Mol'].replace('', np.nan, inplace=True)
    all_preds_val_cv_free_energy = all_preds_val_cv_free_energy.apply(lambda x: pd.Series(x.dropna().values))
    all_preds_val_cv_free_energy.to_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_val_CV_avg_free_energy.csv')

    all_preds_test_cv_free_energy = pd.concat(all_preds_test_cv_free_energy, axis=1)
    all_preds_test_cv_free_energy['Mol'].replace('', np.nan, inplace=True)
    all_preds_test_cv_free_energy = all_preds_test_cv_free_energy.apply(lambda x: pd.Series(x.dropna().values))
    all_preds_test_cv_free_energy.to_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_test_CV_avg_free_energy.csv')
    
    # Output for each individual fold

    for num in range(0, 5):

        os.chdir(f'{root_dir}/resample_{resample}/fold_{num}')

        ### Train
        df_enthalpy = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_train_CV_avg_enthalpy.csv', index_col=False)
        df_entropy = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_train_CV_avg_entropy.csv', index_col=False)
        df_free_energy = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_train_CV_avg_free_energy.csv', index_col=False)
        df2 = pd.read_csv(f'resample_{resample}_fold_{num}_train_plot.csv', index_col=False)
        df2.columns = ['Mol', 'Temp', 'y_enthalpy', 'y_entropy', 'y_free_energy']
        df2['y_enthalpy_pred'] = df_enthalpy[f'fold_{num}']
        df2['y_entropy_pred'] = df_entropy[f'fold_{num}']
        df2['y_free_energy_pred'] = df_free_energy[f'fold_{num}']
        df2.to_csv(f'resample_{resample}_fold_{num}_train_plot.csv', index=False)

        ### Val
        df_enthalpy = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_val_CV_avg_enthalpy.csv', index_col=False)
        df_entropy = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_val_CV_avg_entropy.csv', index_col=False)
        df_free_energy = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_val_CV_avg_free_energy.csv', index_col=False)
        df2 = pd.read_csv(f'resample_{resample}_fold_{num}_val_plot.csv', index_col=False)
        df2.columns = ['Mol', 'Temp', 'y_enthalpy', 'y_entropy', 'y_free_energy']
        df2['y_enthalpy_pred'] = df_enthalpy[f'fold_{num}']
        df2['y_entropy_pred'] = df_entropy[f'fold_{num}']
        df2['y_free_energy_pred'] = df_free_energy[f'fold_{num}']
        df2.to_csv(f'resample_{resample}_fold_{num}_val_plot.csv', index=False)

        ### Test
        df_enthalpy = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_test_CV_avg_enthalpy.csv', index_col=False)
        df_entropy = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_test_CV_avg_entropy.csv', index_col=False)
        df_free_energy = pd.read_csv(f'{root_dir}/resample_{resample}/resample_{resample}_pred_test_CV_avg_free_energy.csv', index_col=False)
        df2 = pd.read_csv(f'resample_{resample}_fold_{num}_test_plot.csv', index_col=False)
        df2.columns = ['Mol', 'Temp', 'y_enthalpy', 'y_entropy', 'y_free_energy']
        df2['y_enthalpy_pred'] = df_enthalpy[f'fold_{num}']
        df2['y_entropy_pred'] = df_entropy[f'fold_{num}']
        df2['y_free_energy_pred'] = df_free_energy[f'fold_{num}']
        df2.to_csv(f'resample_{resample}_fold_{num}_test_plot.csv', index=False)

    ### End of CV loop for this resample

    # Start data prep and model training for this resample 
       
    # Make a resample folder within each resample directory  
    try:
        os.mkdir(f'{root_dir}/resample_{resample}/resample_{resample}')
    except FileExistsError:
        pass

    os.chdir(f'{root_dir}/resample_{resample}/resample_{resample}')

    ### Prepare data into X and y for each resample
    # Take exp as y input in dataframe format
    y_train_1_resample = train_resample[['DH']].copy()
    y_train_2_resample = train_resample[['TDS']].copy()
    y_train_3_resample = train_resample[['DG']].copy()

    y_train_resample_output = pd.DataFrame()
    y_train_resample_output['Mol'] = train_resample['Mol']
    y_train_resample_output['Temp'] = train_resample['Temp']
    y_train_resample_output['y_enthalpy'] = train_resample['DH']
    y_train_resample_output['y_entropy'] = train_resample['TDS']
    y_train_resample_output['y_free_energy'] = train_resample['DG']
    y_train_resample_output.to_csv(f'resample_{resample}_train_plot.csv', index=False)

    y_val_1_resample = val_resample[['DH']].copy()
    y_val_2_resample = val_resample[['TDS']].copy()
    y_val_3_resample = val_resample[['DG']].copy()

    y_val_resample_output = pd.DataFrame()
    y_val_resample_output['Mol'] = val_resample['Mol']
    y_val_resample_output['Temp'] = val_resample['Temp']
    y_val_resample_output['y_enthalpy'] = val_resample['DH']
    y_val_resample_output['y_entropy'] = val_resample['TDS']
    y_val_resample_output['y_free_energy'] = val_resample['DG']
    y_val_resample_output.to_csv(f'resample_{resample}_val_plot.csv', index=False)

    y_test_1_resample = test_resample[['DH']].copy()
    y_test_2_resample = test_resample[['TDS']].copy()
    y_test_3_resample = test_resample[['DG']].copy()

    y_test_resample_output = pd.DataFrame()
    y_test_resample_output['Mol'] = test_resample['Mol']
    y_test_resample_output['Temp'] = test_resample['Temp']
    y_test_resample_output['y_enthalpy'] = test_resample['DH']
    y_test_resample_output['y_entropy'] = test_resample['TDS']
    y_test_resample_output['y_free_energy'] = test_resample['DG']
    y_test_resample_output.to_csv(f'resample_{resample}_test_plot.csv', index=False)

    # Convert y data into required input shape
    y_train_1_resample = y_train_1_resample.to_numpy()
    y_train_1_resample = y_train_1_resample.reshape(y_train_1_resample.shape[0])
    y_train_2_resample = y_train_2_resample.to_numpy()
    y_train_2_resample = y_train_2_resample.reshape(y_train_2_resample.shape[0])
    y_train_3_resample = y_train_3_resample.to_numpy()
    y_train_3_resample = y_train_3_resample.reshape(y_train_3_resample.shape[0])

    y_val_1_resample = y_val_1_resample.to_numpy()
    y_val_1_resample = y_val_1_resample.reshape(y_val_1_resample.shape[0])
    y_val_2_resample = y_val_2_resample.to_numpy()
    y_val_2_resample = y_val_2_resample.reshape(y_val_2_resample.shape[0])
    y_val_3_resample = y_val_3_resample.to_numpy()
    y_val_3_resample = y_val_3_resample.reshape(y_val_3_resample.shape[0])

    y_test_1_resample = y_test_1_resample.to_numpy()
    y_test_1_resample = y_test_1_resample.reshape(y_test_1_resample.shape[0])
    y_test_2_resample = y_test_2_resample.to_numpy()
    y_test_2_resample = y_test_2_resample.reshape(y_test_2_resample.shape[0])
    y_test_3_resample = y_test_3_resample.to_numpy()
    y_test_3_resample = y_test_3_resample.reshape(y_test_3_resample.shape[0])

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
    #means = X_descr_train.mean(axis=0)
    #sds = X_descr_train.std(axis=0)

    #X_descr_train_scaled = (X_descr_train - means) / sds
    #X_descr_val_scaled = (X_descr_val - means) / sds
    #X_descr_test_scaled = (X_descr_test - means) / sds

    # Convert X data into required input shape
    # No temp
    X_train_resample = X_descr_train.to_numpy()
    X_train_resample = X_train_resample.reshape(X_train_resample.shape[0], X_train_resample.shape[1],1)

    X_val_resample = X_descr_val.to_numpy()
    X_val_resample = X_val_resample.reshape(X_val_resample.shape[0], X_val_resample.shape[1],1)

    X_test_resample = X_descr_test.to_numpy()
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
    history = model.fit(X_train_resample, [y_train_1_resample,y_train_2_resample,y_train_3_resample],
                        epochs = epochs,
                        verbose = 1,
                        validation_data =(X_val_resample, [y_val_1_resample,y_val_2_resample,y_val_3_resample]),
                        callbacks=[es, csv_logger, cp_callback, tensorboard_callback])
        
    # Get weights
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    np.savetxt('weights.csv' , weights , fmt='%s', delimiter=',')

    with open('weights.txt', 'w') as f:

        for name, weight in zip(names, weights):
            print(name, weight.shape, weight, file=f)

    # Save Model
    #model.save('saved_model/my_model')

    # Make predictions on train, val and test set using trained model:
    y_pred_train_resample = model.predict(X_train_resample)
    y_pred_train_resample_enthalpy = y_pred_train_resample[0]
    y_pred_train_resample_entropy = y_pred_train_resample[1]
    y_pred_train_resample_free_energy = y_pred_train_resample[2]

    y_pred_val_resample = model.predict(X_val_resample)
    y_pred_val_resample_enthalpy = y_pred_val_resample[0]
    y_pred_val_resample_entropy = y_pred_val_resample[1]
    y_pred_val_resample_free_energy = y_pred_val_resample[2]

    y_pred_test_resample = model.predict(X_test_resample)
    y_pred_test_resample_enthalpy = y_pred_test_resample[0]
    y_pred_test_resample_entropy = y_pred_test_resample[1]
    y_pred_test_resample_free_energy = y_pred_test_resample[2]

    # Assess performace of model based on predictions:

    # Reduce dimensionality of predcition arrays
    y_pred_train_resample_enthalpy = y_pred_train_resample_enthalpy.squeeze()
    y_pred_train_resample_entropy = y_pred_train_resample_entropy.squeeze()
    y_pred_train_resample_free_energy = y_pred_train_resample_free_energy.squeeze()

    y_pred_val_resample_enthalpy = y_pred_val_resample_enthalpy.squeeze()
    y_pred_val_resample_entropy = y_pred_val_resample_entropy.squeeze()
    y_pred_val_resample_free_energy = y_pred_val_resample_free_energy.squeeze()

    y_pred_test_resample_enthalpy = y_pred_test_resample_enthalpy.squeeze()
    y_pred_test_resample_entropy = y_pred_test_resample_entropy.squeeze()
    y_pred_test_resample_free_energy = y_pred_test_resample_free_energy.squeeze()

    # Coefficient of determination
    r2_train_resample_enthalpy = r2_score(y_train_1_resample, y_pred_train_resample_enthalpy)
    r2_train_resample_entropy = r2_score(y_train_2_resample, y_pred_train_resample_entropy)
    r2_train_resample_free_energy = r2_score(y_train_3_resample, y_pred_train_resample_free_energy)
    r2_val_resample_enthalpy = r2_score(y_val_1_resample, y_pred_val_resample_enthalpy)
    r2_val_resample_entropy = r2_score(y_val_2_resample, y_pred_val_resample_entropy)
    r2_val_resample_free_energy = r2_score(y_val_3_resample, y_pred_val_resample_free_energy)
    r2_test_resample_enthalpy = r2_score(y_test_1_resample, y_pred_test_resample_enthalpy)
    r2_test_resample_entropy = r2_score(y_test_2_resample, y_pred_test_resample_entropy)
    r2_test_resample_free_energy = r2_score(y_test_3_resample, y_pred_test_resample_free_energy)
    # Root mean squared error
    rmsd_train_resample_enthalpy = (mean_squared_error(y_train_1_resample, y_pred_train_resample_enthalpy))**0.5
    rmsd_train_resample_entropy = (mean_squared_error(y_train_2_resample, y_pred_train_resample_entropy))**0.5
    rmsd_train_resample_free_energy = (mean_squared_error(y_train_3_resample, y_pred_train_resample_free_energy))**0.5
    rmsd_val_resample_enthalpy = (mean_squared_error(y_val_1_resample, y_pred_val_resample_enthalpy))**0.5
    rmsd_val_resample_entropy = (mean_squared_error(y_val_2_resample, y_pred_val_resample_entropy))**0.5
    rmsd_val_resample_free_energy = (mean_squared_error(y_val_3_resample, y_pred_val_resample_free_energy))**0.5
    rmsd_test_resample_enthalpy = (mean_squared_error(y_test_1_resample, y_pred_test_resample_enthalpy))**0.5
    rmsd_test_resample_entropy = (mean_squared_error(y_test_2_resample, y_pred_test_resample_entropy))**0.5
    rmsd_test_resample_free_energy = (mean_squared_error(y_test_3_resample, y_pred_test_resample_free_energy))**0.5
    # Bias
    bias_train_resample_enthalpy = np.mean(y_pred_train_resample_enthalpy - y_train_1_resample)
    bias_train_resample_entropy = np.mean(y_pred_train_resample_entropy - y_train_2_resample)
    bias_train_resample_free_energy = np.mean(y_pred_train_resample_free_energy - y_train_3_resample)
    bias_val_resample_enthalpy = np.mean(y_pred_val_resample_enthalpy - y_val_1_resample)
    bias_val_resample_entropy = np.mean(y_pred_val_resample_entropy - y_val_2_resample)
    bias_val_resample_free_energy = np.mean(y_pred_val_resample_free_energy - y_val_3_resample)
    bias_test_resample_enthalpy = np.mean(y_pred_test_resample_enthalpy - y_test_1_resample)
    bias_test_resample_entropy = np.mean(y_pred_test_resample_entropy - y_test_2_resample)
    bias_test_resample_free_energy = np.mean(y_pred_test_resample_free_energy - y_test_3_resample)
    # Standard deviation of the error of prediction
    sdep_train_resample_enthalpy = (np.mean((y_pred_train_resample_enthalpy - y_train_1_resample - bias_train_resample_enthalpy)**2))**0.5
    sdep_train_resample_entropy = (np.mean((y_pred_train_resample_entropy - y_train_2_resample - bias_train_resample_entropy)**2))**0.5
    sdep_train_resample_free_energy = (np.mean((y_pred_train_resample_free_energy - y_train_3_resample - bias_train_resample_free_energy)**2))**0.5
    sdep_val_resample_enthalpy = (np.mean((y_pred_val_resample_enthalpy - y_val_1_resample - bias_val_resample_enthalpy)**2))**0.5
    sdep_val_resample_entropy = (np.mean((y_pred_val_resample_entropy - y_val_2_resample - bias_val_resample_entropy)**2))**0.5
    sdep_val_resample_free_energy = (np.mean((y_pred_val_resample_free_energy - y_val_3_resample - bias_val_resample_free_energy)**2))**0.5
    sdep_test_resample_enthalpy = (np.mean((y_pred_test_resample_enthalpy - y_test_1_resample - bias_test_resample_enthalpy)**2))**0.5
    sdep_test_resample_entropy = (np.mean((y_pred_test_resample_entropy - y_test_2_resample - bias_test_resample_entropy)**2))**0.5
    sdep_test_resample_free_energy = (np.mean((y_pred_test_resample_free_energy - y_test_3_resample - bias_test_resample_free_energy)**2))**0.5

    # Save running sum of results:
    r2_train_sum_resample_enthalpy += r2_train_resample_enthalpy
    rmsd_train_sum_resample_enthalpy += rmsd_train_resample_enthalpy
    bias_train_sum_resample_enthalpy += bias_train_resample_enthalpy
    sdep_train_sum_resample_enthalpy += sdep_train_resample_enthalpy

    r2_train_sum_resample_entropy += r2_train_resample_entropy
    rmsd_train_sum_resample_entropy += rmsd_train_resample_entropy
    bias_train_sum_resample_entropy += bias_train_resample_entropy
    sdep_train_sum_resample_entropy += sdep_train_resample_entropy

    r2_train_sum_resample_free_energy += r2_train_resample_free_energy
    rmsd_train_sum_resample_free_energy += rmsd_train_resample_free_energy
    bias_train_sum_resample_free_energy += bias_train_resample_free_energy
    sdep_train_sum_resample_free_energy += sdep_train_resample_free_energy

    r2_val_sum_resample_enthalpy += r2_val_resample_enthalpy
    rmsd_val_sum_resample_enthalpy += rmsd_val_resample_enthalpy
    bias_val_sum_resample_enthalpy += bias_val_resample_enthalpy
    sdep_val_sum_resample_enthalpy += sdep_val_resample_enthalpy

    r2_val_sum_resample_entropy += r2_val_resample_entropy
    rmsd_val_sum_resample_entropy += rmsd_val_resample_entropy
    bias_val_sum_resample_entropy += bias_val_resample_entropy
    sdep_val_sum_resample_entropy += sdep_val_resample_entropy

    r2_val_sum_resample_free_energy += r2_val_resample_free_energy
    rmsd_val_sum_resample_free_energy += rmsd_val_resample_free_energy
    bias_val_sum_resample_free_energy += bias_val_resample_free_energy
    sdep_val_sum_resample_free_energy += sdep_val_resample_free_energy

    r2_test_sum_resample_enthalpy += r2_test_resample_enthalpy
    rmsd_test_sum_resample_enthalpy += rmsd_test_resample_enthalpy
    bias_test_sum_resample_enthalpy += bias_test_resample_enthalpy
    sdep_test_sum_resample_enthalpy += sdep_test_resample_enthalpy

    r2_test_sum_resample_entropy += r2_test_resample_entropy
    rmsd_test_sum_resample_entropy += rmsd_test_resample_entropy
    bias_test_sum_resample_entropy += bias_test_resample_entropy
    sdep_test_sum_resample_entropy += sdep_test_resample_entropy

    r2_test_sum_resample_free_energy += r2_test_resample_free_energy
    rmsd_test_sum_resample_free_energy += rmsd_test_resample_free_energy
    bias_test_sum_resample_free_energy += bias_test_resample_free_energy
    sdep_test_sum_resample_free_energy += sdep_test_resample_free_energy

    # Save individual predictions for resample in dataframes
    y_pred_train_resample_enthalpy = y_pred_train_resample_enthalpy.reshape(y_pred_train_resample_enthalpy.shape[0])
    y_pred_train_resample_enthalpy = pd.DataFrame(y_pred_train_resample_enthalpy, columns=[f'resample_{resample}'])
    y_pred_train_resample_enthalpy_labeled = pd.concat([train_resample['Mol'], train_resample['Temp'], train_resample['DH'], y_pred_train_resample_enthalpy], axis=1)
    y_pred_train_resample_enthalpy_labeled[''] = ''
    all_preds_train_resample_enthalpy.append(y_pred_train_resample_enthalpy_labeled)

    y_pred_val_resample_enthalpy = y_pred_val_resample_enthalpy.reshape(y_pred_val_resample_enthalpy.shape[0])
    y_pred_val_resample_enthalpy = pd.DataFrame(y_pred_val_resample_enthalpy, columns=[f'resample_{resample}'])
    y_pred_val_resample_enthalpy_labeled = pd.concat([val_resample['Mol'], val_resample['Temp'], val_resample['DH'], y_pred_val_resample_enthalpy], axis=1)
    y_pred_val_resample_enthalpy_labeled[''] = ''
    all_preds_val_resample_enthalpy.append(y_pred_val_resample_enthalpy_labeled)

    y_pred_test_resample_enthalpy = y_pred_test_resample_enthalpy.reshape(y_pred_test_resample_enthalpy.shape[0])
    y_pred_test_resample_enthalpy = pd.DataFrame(y_pred_test_resample_enthalpy, columns=[f'resample_{resample}'])
    y_pred_test_resample_enthalpy_labeled = pd.concat([test_resample['Mol'], test_resample['Temp'], test_resample['DH'], y_pred_test_resample_enthalpy], axis=1)
    y_pred_test_resample_enthalpy_labeled[''] = ''
    all_preds_test_resample_enthalpy.append(y_pred_test_resample_enthalpy_labeled)

    y_pred_train_resample_entropy = y_pred_train_resample_entropy.reshape(y_pred_train_resample_entropy.shape[0])
    y_pred_train_resample_entropy = pd.DataFrame(y_pred_train_resample_entropy, columns=[f'resample_{resample}'])
    y_pred_train_resample_entropy_labeled = pd.concat([train_resample['Mol'], train_resample['Temp'], train_resample['TDS'], y_pred_train_resample_entropy], axis=1)
    y_pred_train_resample_entropy_labeled[''] = ''
    all_preds_train_resample_entropy.append(y_pred_train_resample_entropy_labeled)

    y_pred_val_resample_entropy = y_pred_val_resample_entropy.reshape(y_pred_val_resample_entropy.shape[0])
    y_pred_val_resample_entropy = pd.DataFrame(y_pred_val_resample_entropy, columns=[f'resample_{resample}'])
    y_pred_val_resample_entropy_labeled = pd.concat([val_resample['Mol'], val_resample['Temp'], val_resample['TDS'], y_pred_val_resample_entropy], axis=1)
    y_pred_val_resample_entropy_labeled[''] = ''
    all_preds_val_resample_entropy.append(y_pred_val_resample_entropy_labeled)

    y_pred_test_resample_entropy = y_pred_test_resample_entropy.reshape(y_pred_test_resample_entropy.shape[0])
    y_pred_test_resample_entropy = pd.DataFrame(y_pred_test_resample_entropy, columns=[f'resample_{resample}'])
    y_pred_test_resample_entropy_labeled = pd.concat([test_resample['Mol'], test_resample['Temp'], test_resample['TDS'], y_pred_test_resample_entropy], axis=1)
    y_pred_test_resample_entropy_labeled[''] = ''
    all_preds_test_resample_entropy.append(y_pred_test_resample_entropy_labeled)

    y_pred_train_resample_free_energy = y_pred_train_resample_free_energy.reshape(y_pred_train_resample_free_energy.shape[0])
    y_pred_train_resample_free_energy = pd.DataFrame(y_pred_train_resample_free_energy, columns=[f'resample_{resample}'])
    y_pred_train_resample_free_energy_labeled = pd.concat([train_resample['Mol'], train_resample['Temp'], train_resample['DG'], y_pred_train_resample_free_energy], axis=1)
    y_pred_train_resample_free_energy_labeled[''] = ''
    all_preds_train_resample_free_energy.append(y_pred_train_resample_free_energy_labeled)

    y_pred_val_resample_free_energy = y_pred_val_resample_free_energy.reshape(y_pred_val_resample_free_energy.shape[0])
    y_pred_val_resample_free_energy = pd.DataFrame(y_pred_val_resample_free_energy, columns=[f'resample_{resample}'])
    y_pred_val_resample_free_energy_labeled = pd.concat([val_resample['Mol'], val_resample['Temp'], val_resample['DG'], y_pred_val_resample_free_energy], axis=1)
    y_pred_val_resample_free_energy_labeled[''] = ''
    all_preds_val_resample_free_energy.append(y_pred_val_resample_free_energy_labeled)

    y_pred_test_resample_free_energy = y_pred_test_resample_free_energy.reshape(y_pred_test_resample_free_energy.shape[0])
    y_pred_test_resample_free_energy = pd.DataFrame(y_pred_test_resample_free_energy, columns=[f'resample_{resample}'])
    y_pred_test_resample_free_energy_labeled = pd.concat([test_resample['Mol'], test_resample['Temp'], test_resample['DG'], y_pred_test_resample_free_energy], axis=1)
    y_pred_test_resample_free_energy_labeled[''] = ''
    all_preds_test_resample_free_energy.append(y_pred_test_resample_free_energy_labeled)

# Average results over resamples:
r2_train_av_resample_enthalpy = r2_train_sum_resample_enthalpy/no_resamples
rmsd_train_av_resample_enthalpy = rmsd_train_sum_resample_enthalpy/no_resamples
bias_train_av_resample_enthalpy = bias_train_sum_resample_enthalpy/no_resamples
sdep_train_av_resample_enthalpy = sdep_train_sum_resample_enthalpy/no_resamples

r2_train_av_resample_enthalpy = '{:.3f}'.format(r2_train_av_resample_enthalpy)
rmsd_train_av_resample_enthalpy = '{:.3f}'.format(rmsd_train_av_resample_enthalpy)
bias_train_av_resample_enthalpy = '{:.3f}'.format(bias_train_av_resample_enthalpy)
sdep_train_av_resample_enthalpy = '{:.3f}'.format(sdep_train_av_resample_enthalpy)

r2_train_av_resample_entropy = r2_train_sum_resample_entropy/no_resamples
rmsd_train_av_resample_entropy = rmsd_train_sum_resample_entropy/no_resamples
bias_train_av_resample_entropy = bias_train_sum_resample_entropy/no_resamples
sdep_train_av_resample_entropy = sdep_train_sum_resample_entropy/no_resamples

r2_train_av_resample_entropy = '{:.3f}'.format(r2_train_av_resample_entropy)
rmsd_train_av_resample_entropy = '{:.3f}'.format(rmsd_train_av_resample_entropy)
bias_train_av_resample_entropy = '{:.3f}'.format(bias_train_av_resample_entropy)
sdep_train_av_resample_entropy = '{:.3f}'.format(sdep_train_av_resample_entropy)

r2_train_av_resample_free_energy = r2_train_sum_resample_free_energy/no_resamples
rmsd_train_av_resample_free_energy = rmsd_train_sum_resample_free_energy/no_resamples
bias_train_av_resample_free_energy = bias_train_sum_resample_free_energy/no_resamples
sdep_train_av_resample_free_energy = sdep_train_sum_resample_free_energy/no_resamples

r2_train_av_resample_free_energy = '{:.3f}'.format(r2_train_av_resample_free_energy)
rmsd_train_av_resample_free_energy = '{:.3f}'.format(rmsd_train_av_resample_free_energy)
bias_train_av_resample_free_energy = '{:.3f}'.format(bias_train_av_resample_free_energy)
sdep_train_av_resample_free_energy = '{:.3f}'.format(sdep_train_av_resample_free_energy)
#
r2_val_av_resample_enthalpy = r2_val_sum_resample_enthalpy/no_resamples
rmsd_val_av_resample_enthalpy = rmsd_val_sum_resample_enthalpy/no_resamples
bias_val_av_resample_enthalpy = bias_val_sum_resample_enthalpy/no_resamples
sdep_val_av_resample_enthalpy = sdep_val_sum_resample_enthalpy/no_resamples

r2_val_av_resample_enthalpy = '{:.3f}'.format(r2_val_av_resample_enthalpy)
rmsd_val_av_resample_enthalpy = '{:.3f}'.format(rmsd_val_av_resample_enthalpy)
bias_val_av_resample_enthalpy = '{:.3f}'.format(bias_val_av_resample_enthalpy)
sdep_val_av_resample_enthalpy = '{:.3f}'.format(sdep_val_av_resample_enthalpy)

r2_val_av_resample_entropy = r2_val_sum_resample_entropy/no_resamples
rmsd_val_av_resample_entropy = rmsd_val_sum_resample_entropy/no_resamples
bias_val_av_resample_entropy = bias_val_sum_resample_entropy/no_resamples
sdep_val_av_resample_entropy = sdep_val_sum_resample_entropy/no_resamples

r2_val_av_resample_entropy = '{:.3f}'.format(r2_val_av_resample_entropy)
rmsd_val_av_resample_entropy = '{:.3f}'.format(rmsd_val_av_resample_entropy)
bias_val_av_resample_entropy = '{:.3f}'.format(bias_val_av_resample_entropy)
sdep_val_av_resample_entropy = '{:.3f}'.format(sdep_val_av_resample_entropy)

r2_val_av_resample_free_energy = r2_val_sum_resample_free_energy/no_resamples
rmsd_val_av_resample_free_energy = rmsd_val_sum_resample_free_energy/no_resamples
bias_val_av_resample_free_energy = bias_val_sum_resample_free_energy/no_resamples
sdep_val_av_resample_free_energy = sdep_val_sum_resample_free_energy/no_resamples

r2_val_av_resample_free_energy = '{:.3f}'.format(r2_val_av_resample_free_energy)
rmsd_val_av_resample_free_energy = '{:.3f}'.format(rmsd_val_av_resample_free_energy)
bias_val_av_resample_free_energy = '{:.3f}'.format(bias_val_av_resample_free_energy)
sdep_val_av_resample_free_energy = '{:.3f}'.format(sdep_val_av_resample_free_energy)
#
r2_test_av_resample_enthalpy = r2_test_sum_resample_enthalpy/no_resamples
rmsd_test_av_resample_enthalpy = rmsd_test_sum_resample_enthalpy/no_resamples
bias_test_av_resample_enthalpy = bias_test_sum_resample_enthalpy/no_resamples
sdep_test_av_resample_enthalpy = sdep_test_sum_resample_enthalpy/no_resamples

r2_test_av_resample_enthalpy = '{:.3f}'.format(r2_test_av_resample_enthalpy)
rmsd_test_av_resample_enthalpy = '{:.3f}'.format(rmsd_test_av_resample_enthalpy)
bias_test_av_resample_enthalpy = '{:.3f}'.format(bias_test_av_resample_enthalpy)
sdep_test_av_resample_enthalpy = '{:.3f}'.format(sdep_test_av_resample_enthalpy)

r2_test_av_resample_entropy = r2_test_sum_resample_entropy/no_resamples
rmsd_test_av_resample_entropy = rmsd_test_sum_resample_entropy/no_resamples
bias_test_av_resample_entropy = bias_test_sum_resample_entropy/no_resamples
sdep_test_av_resample_entropy = sdep_test_sum_resample_entropy/no_resamples

r2_test_av_resample_entropy = '{:.3f}'.format(r2_test_av_resample_entropy)
rmsd_test_av_resample_entropy = '{:.3f}'.format(rmsd_test_av_resample_entropy)
bias_test_av_resample_entropy = '{:.3f}'.format(bias_test_av_resample_entropy)
sdep_test_av_resample_entropy = '{:.3f}'.format(sdep_test_av_resample_entropy)

r2_test_av_resample_free_energy = r2_test_sum_resample_free_energy/no_resamples
rmsd_test_av_resample_free_energy = rmsd_test_sum_resample_free_energy/no_resamples
bias_test_av_resample_free_energy = bias_test_sum_resample_free_energy/no_resamples
sdep_test_av_resample_free_energy = sdep_test_sum_resample_free_energy/no_resamples

r2_test_av_resample_free_energy = '{:.3f}'.format(r2_test_av_resample_free_energy)
rmsd_test_av_resample_free_energy = '{:.3f}'.format(rmsd_test_av_resample_free_energy)
bias_test_av_resample_free_energy = '{:.3f}'.format(bias_test_av_resample_free_energy)
sdep_test_av_resample_free_energy = '{:.3f}'.format(sdep_test_av_resample_free_energy)

# Write average enthalpy results to a file:
results_file = open(f'{root_dir}/resample_avg_stats_enthalpy.txt', 'w')
results_file.write(f'r2_train: {r2_train_av_resample_enthalpy}')
results_file.write('\n')
results_file.write(f'rmsd_train: {rmsd_train_av_resample_enthalpy}')
results_file.write('\n')
results_file.write(f'bias_train: {bias_train_av_resample_enthalpy}')
results_file.write('\n')
results_file.write(f'sdep_train: {sdep_train_av_resample_enthalpy}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_val: {r2_val_av_resample_enthalpy}')
results_file.write('\n')
results_file.write(f'rmsd_val: {rmsd_val_av_resample_enthalpy}')
results_file.write('\n')
results_file.write(f'bias_val: {bias_val_av_resample_enthalpy}')
results_file.write('\n')
results_file.write(f'sdep_val: {sdep_val_av_resample_enthalpy}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_test: {r2_test_av_resample_enthalpy}')
results_file.write('\n')
results_file.write(f'rmsd_test: {rmsd_test_av_resample_enthalpy}')
results_file.write('\n')
results_file.write(f'bias_test: {bias_test_av_resample_enthalpy}')
results_file.write('\n')
results_file.write(f'sdep_test: {sdep_test_av_resample_enthalpy}')
results_file.close()

# Write average entropy results to a file:
results_file = open(f'{root_dir}/resample_avg_stats_entropy.txt', 'w')
results_file.write(f'r2_train: {r2_train_av_resample_entropy}')
results_file.write('\n')
results_file.write(f'rmsd_train: {rmsd_train_av_resample_entropy}')
results_file.write('\n')
results_file.write(f'bias_train: {bias_train_av_resample_entropy}')
results_file.write('\n')
results_file.write(f'sdep_train: {sdep_train_av_resample_entropy}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_val: {r2_val_av_resample_entropy}')
results_file.write('\n')
results_file.write(f'rmsd_val: {rmsd_val_av_resample_entropy}')
results_file.write('\n')
results_file.write(f'bias_val: {bias_val_av_resample_entropy}')
results_file.write('\n')
results_file.write(f'sdep_val: {sdep_val_av_resample_entropy}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_test: {r2_test_av_resample_entropy}')
results_file.write('\n')
results_file.write(f'rmsd_test: {rmsd_test_av_resample_entropy}')
results_file.write('\n')
results_file.write(f'bias_test: {bias_test_av_resample_entropy}')
results_file.write('\n')
results_file.write(f'sdep_test: {sdep_test_av_resample_entropy}')
results_file.close()

# Write average free energy results to a file:
results_file = open(f'{root_dir}/resample_avg_stats_free_energy.txt', 'w')
results_file.write(f'r2_train: {r2_train_av_resample_free_energy}')
results_file.write('\n')
results_file.write(f'rmsd_train: {rmsd_train_av_resample_free_energy}')
results_file.write('\n')
results_file.write(f'bias_train: {bias_train_av_resample_free_energy}')
results_file.write('\n')
results_file.write(f'sdep_train: {sdep_train_av_resample_free_energy}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_val: {r2_val_av_resample_free_energy}')
results_file.write('\n')
results_file.write(f'rmsd_val: {rmsd_val_av_resample_free_energy}')
results_file.write('\n')
results_file.write(f'bias_val: {bias_val_av_resample_free_energy}')
results_file.write('\n')
results_file.write(f'sdep_val: {sdep_val_av_resample_free_energy}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_test: {r2_test_av_resample_free_energy}')
results_file.write('\n')
results_file.write(f'rmsd_test: {rmsd_test_av_resample_free_energy}')
results_file.write('\n')
results_file.write(f'bias_test: {bias_test_av_resample_free_energy}')
results_file.write('\n')
results_file.write(f'sdep_test: {sdep_test_av_resample_free_energy}')
results_file.close()

# Save all individual predictions to train file:
all_preds_train_resample_enthalpy = pd.concat(all_preds_train_resample_enthalpy, axis=1)
all_preds_train_resample_enthalpy['Mol'].replace('', np.nan, inplace=True)
all_preds_train_resample_enthalpy = all_preds_train_resample_enthalpy.apply(lambda x: pd.Series(x.dropna().values))
all_preds_train_resample_enthalpy.to_csv(f'{root_dir}/resample_pred_train_avg_enthalpy.csv')

all_preds_val_resample_enthalpy = pd.concat(all_preds_val_resample_enthalpy, axis=1)
all_preds_val_resample_enthalpy['Mol'].replace('', np.nan, inplace=True)
all_preds_val_resample_enthalpy = all_preds_val_resample_enthalpy.apply(lambda x: pd.Series(x.dropna().values))
all_preds_val_resample_enthalpy.to_csv(f'{root_dir}/resample_pred_val_avg_enthalpy.csv')

all_preds_test_resample_enthalpy = pd.concat(all_preds_test_resample_enthalpy, axis=1)
all_preds_test_resample_enthalpy['Mol'].replace('', np.nan, inplace=True)
all_preds_test_resample_enthalpy = all_preds_test_resample_enthalpy.apply(lambda x: pd.Series(x.dropna().values))
all_preds_test_resample_enthalpy.to_csv(f'{root_dir}/resample_pred_test_avg_enthalpy.csv')

all_preds_train_resample_entropy = pd.concat(all_preds_train_resample_entropy, axis=1)
all_preds_train_resample_entropy['Mol'].replace('', np.nan, inplace=True)
all_preds_train_resample_entropy = all_preds_train_resample_entropy.apply(lambda x: pd.Series(x.dropna().values))
all_preds_train_resample_entropy.to_csv(f'{root_dir}/resample_pred_train_avg_entropy.csv')

all_preds_val_resample_entropy = pd.concat(all_preds_val_resample_entropy, axis=1)
all_preds_val_resample_entropy['Mol'].replace('', np.nan, inplace=True)
all_preds_val_resample_entropy = all_preds_val_resample_entropy.apply(lambda x: pd.Series(x.dropna().values))
all_preds_val_resample_entropy.to_csv(f'{root_dir}/resample_pred_val_avg_entropy.csv')

all_preds_test_resample_entropy = pd.concat(all_preds_test_resample_entropy, axis=1)
all_preds_test_resample_entropy['Mol'].replace('', np.nan, inplace=True)
all_preds_test_resample_entropy = all_preds_test_resample_entropy.apply(lambda x: pd.Series(x.dropna().values))
all_preds_test_resample_entropy.to_csv(f'{root_dir}/resample_pred_test_avg_entropy.csv')

all_preds_train_resample_free_energy = pd.concat(all_preds_train_resample_free_energy, axis=1)
all_preds_train_resample_free_energy['Mol'].replace('', np.nan, inplace=True)
all_preds_train_resample_free_energy = all_preds_train_resample_free_energy.apply(lambda x: pd.Series(x.dropna().values))
all_preds_train_resample_free_energy.to_csv(f'{root_dir}/resample_pred_train_avg_free_energy.csv')

all_preds_val_resample_free_energy = pd.concat(all_preds_val_resample_free_energy, axis=1)
all_preds_val_resample_free_energy['Mol'].replace('', np.nan, inplace=True)
all_preds_val_resample_free_energy = all_preds_val_resample_free_energy.apply(lambda x: pd.Series(x.dropna().values))
all_preds_val_resample_free_energy.to_csv(f'{root_dir}/resample_pred_val_avg_free_energy.csv')

all_preds_test_resample_free_energy = pd.concat(all_preds_test_resample_free_energy, axis=1)
all_preds_test_resample_free_energy['Mol'].replace('', np.nan, inplace=True)
all_preds_test_resample_free_energy = all_preds_test_resample_free_energy.apply(lambda x: pd.Series(x.dropna().values))
all_preds_test_resample_free_energy.to_csv(f'{root_dir}/resample_pred_test_avg_free_energy.csv')

# Stats for each individual resample
for resample in range(1, no_resamples + 1):

    os.chdir(f'{root_dir}/resample_{resample}/resample_{resample}')

    ### Test enthalpy
    df_enthalpy = pd.read_csv(f'{root_dir}/resample_pred_test_avg_enthalpy.csv', index_col=False)
    df_entropy = pd.read_csv(f'{root_dir}/resample_pred_test_avg_entropy.csv', index_col=False)
    df_free_energy = pd.read_csv(f'{root_dir}/resample_pred_test_avg_free_energy.csv', index_col=False)
    df2 = pd.read_csv(f'resample_{resample}_test_plot.csv', index_col=False)
    df2.columns = ['Mol', 'Temp', 'y_enthalpy', 'y_entropy', 'y_free_energy']
    df2['y_enthalpy_pred'] = df_enthalpy[f'resample_{resample}']
    df2['y_entropy_pred'] = df_entropy[f'resample_{resample}']
    df2['y_free_energy_pred'] = df_free_energy[f'resample_{resample}']
    df2.to_csv(f'resample_{resample}_test_plot.csv', index=False)

    np_y_enthalpy = df2['y_enthalpy'].to_numpy()
    np_y_enthalpy_pred = df2['y_enthalpy_pred'].to_numpy()

    r2_enthalpy = r2_score(np_y_enthalpy, np_y_enthalpy_pred)
    rmsd_enthalpy = (mean_squared_error(np_y_enthalpy, np_y_enthalpy_pred))**0.5
    bias_enthalpy = np.mean(np_y_enthalpy_pred - np_y_enthalpy)
    sdep_enthalpy = (np.mean((np_y_enthalpy_pred - np_y_enthalpy - bias_enthalpy)**2))**0.5

    r2_enthalpy = '{:.3f}'.format(r2_enthalpy)
    rmsd_enthalpy = '{:.3f}'.format(rmsd_enthalpy)
    bias_enthalpy = '{:.3f}'.format(bias_enthalpy)
    sdep_enthalpy = '{:.3f}'.format(sdep_enthalpy)

    results_file = open(f'resample_{resample}_stats_enthalpy.txt', 'w')
    results_file.write(f'r2_test: {r2_enthalpy}')
    results_file.write('\n')
    results_file.write(f'rmsd_test: {rmsd_enthalpy}')
    results_file.write('\n')
    results_file.write(f'bias_test: {bias_enthalpy}')
    results_file.write('\n')
    results_file.write(f'sdep_test: {sdep_enthalpy}')
    results_file.write('\n')
    results_file.write('\n')
    results_file.close()

    fig, ax = plt.subplots()

    # Line of best fit
    try:
        a, b = np.polyfit(df2['y_enthalpy'], df2['y_enthalpy_pred'], 1)
        plot_a = '{:.3f}'.format(a)
        plot_b = '{:.3f}'.format(b)
        plt.plot(df2['y_enthalpy'], a * df2['y_enthalpy'] + b, color='purple')
    except np.linalg.LinAlgError:
        pass

    # Plot everything
    try:
        plt.plot([], [], ' ', label='Test set')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_enthalpy}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd_enthalpy}')
        plt.plot([], [], ' ', label=f'Bias : {bias_enthalpy}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep_enthalpy}')
        plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
        plt.scatter(df2['y_enthalpy'], df2['y_enthalpy_pred'])
        order = [0,1,2,3,4,5]
    except NameError:
        plt.plot([], [], ' ', label='Test set')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_enthalpy}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd_enthalpy}')
        plt.plot([], [], ' ', label=f'Bias : {bias_enthalpy}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep_enthalpy}')
        plt.scatter(df2['y_enthalpy'], df2['y_enthalpy_pred'])
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
    plt.xlabel('$ΔH^{exp}_{solv}$ (kcal/mol)')
    plt.ylabel('$ΔH^{calc}_{solv}$ (kcal/mol)')
    handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    leg.get_frame().set_linewidth(0.0)
    fig.savefig(f'resample_{resample}_test_pred_enthalpy.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.cla()
    plt.close(fig)

    ### Test entropy
    np_y_entropy = df2['y_entropy'].to_numpy()
    np_y_entropy_pred = df2['y_entropy_pred'].to_numpy()

    r2_entropy = r2_score(np_y_entropy, np_y_entropy_pred)
    rmsd_entropy = (mean_squared_error(np_y_entropy, np_y_entropy_pred))**0.5
    bias_entropy = np.mean(np_y_entropy_pred - np_y_entropy)
    sdep_entropy = (np.mean((np_y_entropy_pred - np_y_entropy - bias_entropy)**2))**0.5

    r2_entropy = '{:.3f}'.format(r2_entropy)
    rmsd_entropy = '{:.3f}'.format(rmsd_entropy)
    bias_entropy = '{:.3f}'.format(bias_entropy)
    sdep_entropy = '{:.3f}'.format(sdep_entropy)

    results_file = open(f'resample_{resample}_stats_entropy.txt', 'w')
    results_file.write(f'r2_test: {r2_entropy}')
    results_file.write('\n')
    results_file.write(f'rmsd_test: {rmsd_entropy}')
    results_file.write('\n')
    results_file.write(f'bias_test: {bias_entropy}')
    results_file.write('\n')
    results_file.write(f'sdep_test: {sdep_entropy}')
    results_file.write('\n')
    results_file.write('\n')
    results_file.close()

    fig, ax = plt.subplots()

    # Line of best fit
    try:
        a, b = np.polyfit(df2['y_entropy'], df2['y_entropy_pred'], 1)
        plot_a = '{:.3f}'.format(a)
        plot_b = '{:.3f}'.format(b)
        plt.plot(df2['y_entropy'], a * df2['y_entropy'] + b, color='purple')
    except np.linalg.LinAlgError:
        pass

    # Plot everything
    try:
        plt.plot([], [], ' ', label='Test set')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_entropy}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd_entropy}')
        plt.plot([], [], ' ', label=f'Bias : {bias_entropy}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep_entropy}')
        plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
        plt.scatter(df2['y_entropy'], df2['y_entropy_pred'])
        order = [0,1,2,3,4,5]
    except NameError:
        plt.plot([], [], ' ', label='Test set')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_entropy}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd_entropy}')
        plt.plot([], [], ' ', label=f'Bias : {bias_entropy}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep_entropy}')
        plt.scatter(df2['y_entropy'], df2['y_entropy_pred'])
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
    plt.xlabel('$TΔS^{exp}_{solv}$ (kcal/mol)')
    plt.ylabel('$TΔS^{calc}_{solv}$ (kcal/mol)')
    handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    leg.get_frame().set_linewidth(0.0)
    fig.savefig(f'resample_{resample}_test_pred_entropy.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.cla()
    plt.close(fig)

    ### Test free energy
    np_y_free_energy = df2['y_free_energy'].to_numpy()
    np_y_free_energy_pred = df2['y_free_energy_pred'].to_numpy()

    r2_free_energy = r2_score(np_y_free_energy, np_y_free_energy_pred)
    rmsd_free_energy = (mean_squared_error(np_y_free_energy, np_y_free_energy_pred))**0.5
    bias_free_energy = np.mean(np_y_free_energy_pred - np_y_free_energy)
    sdep_free_energy = (np.mean((np_y_free_energy_pred - np_y_free_energy - bias_free_energy)**2))**0.5

    r2_free_energy = '{:.3f}'.format(r2_free_energy)
    rmsd_free_energy = '{:.3f}'.format(rmsd_free_energy)
    bias_free_energy = '{:.3f}'.format(bias_free_energy)
    sdep_free_energy = '{:.3f}'.format(sdep_free_energy)

    results_file = open(f'resample_{resample}_stats_free_energy.txt', 'w')
    results_file.write(f'r2_test: {r2_free_energy}')
    results_file.write('\n')
    results_file.write(f'rmsd_test: {rmsd_free_energy}')
    results_file.write('\n')
    results_file.write(f'bias_test: {bias_free_energy}')
    results_file.write('\n')
    results_file.write(f'sdep_test: {sdep_free_energy}')
    results_file.write('\n')
    results_file.write('\n')
    results_file.close()

    fig, ax = plt.subplots()

    # Line of best fit
    try:
        a, b = np.polyfit(df2['y_free_energy'], df2['y_free_energy_pred'], 1)
        plot_a = '{:.3f}'.format(a)
        plot_b = '{:.3f}'.format(b)
        plt.plot(df2['y_free_energy'], a * df2['y_free_energy'] + b, color='purple')
    except np.linalg.LinAlgError:
        pass

    # Plot everything
    try:
        plt.plot([], [], ' ', label='Test set')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_free_energy}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd_free_energy}')
        plt.plot([], [], ' ', label=f'Bias : {bias_free_energy}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep_free_energy}')
        plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
        plt.scatter(df2['y_free_energy'], df2['y_free_energy_pred'])
        order = [0,1,2,3,4,5]
    except NameError:
        plt.plot([], [], ' ', label='Test set')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_free_energy}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd_free_energy}')
        plt.plot([], [], ' ', label=f'Bias : {bias_free_energy}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep_free_energy}')
        plt.scatter(df2['y_free_energy'], df2['y_free_energy_pred'])
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
    fig.savefig(f'resample_{resample}_test_pred_free_energy.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.cla()
    plt.close(fig)

### Compare CV average results against resample average results, should be similar

r2_train_enthalpy = []
rmsd_train_enthalpy = []
bias_train_enthalpy = []
sdep_train_enthalpy = []

r2_val_enthalpy = []
rmsd_val_enthalpy = []
bias_val_enthalpy = []
sdep_val_enthalpy = []

r2_test_enthalpy = []
rmsd_test_enthalpy = []
bias_test_enthalpy = []
sdep_test_enthalpy = []

for folders in glob.glob(f'{root_dir}/*/*_enthalpy.txt'):

    with open(f'{folders}', 'r') as f:
        lines = f.readlines()		
        r2_line_test = [line for line in lines if line.startswith('r2_test')]
        rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
        bias_line_test = [line for line in lines if line.startswith('bias_test')]		
        sdep_line_test = [line for line in lines if line.startswith('sdep_test')]

        r2_test_enthalpy.append(float(r2_line_test[0].split()[1]))
        rmsd_test_enthalpy.append(float(rmsd_line_test[0].split()[1]))
        bias_test_enthalpy.append(float(bias_line_test[0].split()[1]))
        sdep_test_enthalpy.append(float(sdep_line_test[0].split()[1]))

        r2_line_train = [line for line in lines if line.startswith('r2_train')]
        rmsd_line_train = [line for line in lines if line.startswith('rmsd_train')]
        bias_line_train = [line for line in lines if line.startswith('bias_train')]		
        sdep_line_train = [line for line in lines if line.startswith('sdep_train')]

        r2_train_enthalpy.append(float(r2_line_train[0].split()[1]))
        rmsd_train_enthalpy.append(float(rmsd_line_train[0].split()[1]))
        bias_train_enthalpy.append(float(bias_line_train[0].split()[1]))
        sdep_train_enthalpy.append(float(sdep_line_train[0].split()[1]))

        r2_line_val = [line for line in lines if line.startswith('r2_val')]
        rmsd_line_val = [line for line in lines if line.startswith('rmsd_val')]
        bias_line_val = [line for line in lines if line.startswith('bias_val')]		
        sdep_line_val = [line for line in lines if line.startswith('sdep_val')]

        r2_val_enthalpy.append(float(r2_line_val[0].split()[1]))
        rmsd_val_enthalpy.append(float(rmsd_line_val[0].split()[1]))
        bias_val_enthalpy.append(float(bias_line_val[0].split()[1]))
        sdep_val_enthalpy.append(float(sdep_line_val[0].split()[1]))

r2_train_avg_enthalpy = round(sum(r2_train_enthalpy) / len(r2_train_enthalpy), 3)
rmsd_train_avg_enthalpy = round(sum(rmsd_train_enthalpy) / len(rmsd_train_enthalpy), 3)
bias_train_avg_enthalpy = round(sum(bias_train_enthalpy) / len(bias_train_enthalpy), 3)
sdep_train_avg_enthalpy = round(sum(sdep_train_enthalpy) / len(sdep_train_enthalpy), 3)

r2_test_avg_enthalpy = round(sum(r2_test_enthalpy) / len(r2_test_enthalpy), 3)
rmsd_test_avg_enthalpy = round(sum(rmsd_test_enthalpy) / len(rmsd_test_enthalpy), 3)
bias_test_avg_enthalpy = round(sum(bias_test_enthalpy) / len(bias_test_enthalpy), 3)
sdep_test_avg_enthalpy = round(sum(sdep_test_enthalpy) / len(sdep_test_enthalpy), 3)

r2_val_avg_enthalpy = round(sum(r2_val_enthalpy) / len(r2_val_enthalpy), 3)
rmsd_val_avg_enthalpy = round(sum(rmsd_val_enthalpy) / len(rmsd_val_enthalpy), 3)
bias_val_avg_enthalpy = round(sum(bias_val_enthalpy) / len(bias_val_enthalpy), 3)
sdep_val_avg_enthalpy = round(sum(sdep_val_enthalpy) / len(sdep_val_enthalpy), 3)

results_file = open(f'{root_dir}/CV_avg_stats_against_resamples_enthalpy.txt', 'w')
results_file.write(f'r2_train_avg: {r2_train_avg_enthalpy}')
results_file.write('\n')
results_file.write(f'rmsd_train_avg: {rmsd_train_avg_enthalpy}')
results_file.write('\n')
results_file.write(f'bias_train_avg: {bias_train_avg_enthalpy}')
results_file.write('\n')
results_file.write(f'sdep_train_avg: {sdep_train_avg_enthalpy}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_val_avg: {r2_val_avg_enthalpy}')
results_file.write('\n')
results_file.write(f'rmsd_val_avg: {rmsd_val_avg_enthalpy}')
results_file.write('\n')
results_file.write(f'bias_val_avg: {bias_val_avg_enthalpy}')
results_file.write('\n')
results_file.write(f'sdep_val_avg: {sdep_val_avg_enthalpy}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_test_avg: {r2_test_avg_enthalpy}')
results_file.write('\n')
results_file.write(f'rmsd_test_avg: {rmsd_test_avg_enthalpy}')
results_file.write('\n')
results_file.write(f'bias_test_avg: {bias_test_avg_enthalpy}')
results_file.write('\n')
results_file.write(f'sdep_test_avg: {sdep_test_avg_enthalpy}')

r2_train_entropy = []
rmsd_train_entropy = []
bias_train_entropy = []
sdep_train_entropy = []

r2_val_entropy = []
rmsd_val_entropy = []
bias_val_entropy = []
sdep_val_entropy = []

r2_test_entropy = []
rmsd_test_entropy = []
bias_test_entropy = []
sdep_test_entropy = []

for folders in glob.glob(f'{root_dir}/*/*_entropy.txt'):

    with open(f'{folders}', 'r') as f:
        lines = f.readlines()		
        r2_line_test = [line for line in lines if line.startswith('r2_test')]
        rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
        bias_line_test = [line for line in lines if line.startswith('bias_test')]		
        sdep_line_test = [line for line in lines if line.startswith('sdep_test')]

        r2_test_entropy.append(float(r2_line_test[0].split()[1]))
        rmsd_test_entropy.append(float(rmsd_line_test[0].split()[1]))
        bias_test_entropy.append(float(bias_line_test[0].split()[1]))
        sdep_test_entropy.append(float(sdep_line_test[0].split()[1]))

        r2_line_train = [line for line in lines if line.startswith('r2_train')]
        rmsd_line_train = [line for line in lines if line.startswith('rmsd_train')]
        bias_line_train = [line for line in lines if line.startswith('bias_train')]		
        sdep_line_train = [line for line in lines if line.startswith('sdep_train')]

        r2_train_entropy.append(float(r2_line_train[0].split()[1]))
        rmsd_train_entropy.append(float(rmsd_line_train[0].split()[1]))
        bias_train_entropy.append(float(bias_line_train[0].split()[1]))
        sdep_train_entropy.append(float(sdep_line_train[0].split()[1]))

        r2_line_val = [line for line in lines if line.startswith('r2_val')]
        rmsd_line_val = [line for line in lines if line.startswith('rmsd_val')]
        bias_line_val = [line for line in lines if line.startswith('bias_val')]		
        sdep_line_val = [line for line in lines if line.startswith('sdep_val')]

        r2_val_entropy.append(float(r2_line_val[0].split()[1]))
        rmsd_val_entropy.append(float(rmsd_line_val[0].split()[1]))
        bias_val_entropy.append(float(bias_line_val[0].split()[1]))
        sdep_val_entropy.append(float(sdep_line_val[0].split()[1]))

r2_train_avg_entropy = round(sum(r2_train_entropy) / len(r2_train_entropy), 3)
rmsd_train_avg_entropy = round(sum(rmsd_train_entropy) / len(rmsd_train_entropy), 3)
bias_train_avg_entropy = round(sum(bias_train_entropy) / len(bias_train_entropy), 3)
sdep_train_avg_entropy = round(sum(sdep_train_entropy) / len(sdep_train_entropy), 3)

r2_test_avg_entropy = round(sum(r2_test_entropy) / len(r2_test_entropy), 3)
rmsd_test_avg_entropy = round(sum(rmsd_test_entropy) / len(rmsd_test_entropy), 3)
bias_test_avg_entropy = round(sum(bias_test_entropy) / len(bias_test_entropy), 3)
sdep_test_avg_entropy = round(sum(sdep_test_entropy) / len(sdep_test_entropy), 3)

r2_val_avg_entropy = round(sum(r2_val_entropy) / len(r2_val_entropy), 3)
rmsd_val_avg_entropy = round(sum(rmsd_val_entropy) / len(rmsd_val_entropy), 3)
bias_val_avg_entropy = round(sum(bias_val_entropy) / len(bias_val_entropy), 3)
sdep_val_avg_entropy = round(sum(sdep_val_entropy) / len(sdep_val_entropy), 3)

results_file = open(f'{root_dir}/CV_avg_stats_against_resamples_entropy.txt', 'w')
results_file.write(f'r2_train_avg: {r2_train_avg_entropy}')
results_file.write('\n')
results_file.write(f'rmsd_train_avg: {rmsd_train_avg_entropy}')
results_file.write('\n')
results_file.write(f'bias_train_avg: {bias_train_avg_entropy}')
results_file.write('\n')
results_file.write(f'sdep_train_avg: {sdep_train_avg_entropy}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_val_avg: {r2_val_avg_entropy}')
results_file.write('\n')
results_file.write(f'rmsd_val_avg: {rmsd_val_avg_entropy}')
results_file.write('\n')
results_file.write(f'bias_val_avg: {bias_val_avg_entropy}')
results_file.write('\n')
results_file.write(f'sdep_val_avg: {sdep_val_avg_entropy}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_test_avg: {r2_test_avg_entropy}')
results_file.write('\n')
results_file.write(f'rmsd_test_avg: {rmsd_test_avg_entropy}')
results_file.write('\n')
results_file.write(f'bias_test_avg: {bias_test_avg_entropy}')
results_file.write('\n')
results_file.write(f'sdep_test_avg: {sdep_test_avg_entropy}')

r2_train_free_energy = []
rmsd_train_free_energy = []
bias_train_free_energy = []
sdep_train_free_energy = []

r2_val_free_energy = []
rmsd_val_free_energy = []
bias_val_free_energy = []
sdep_val_free_energy = []

r2_test_free_energy = []
rmsd_test_free_energy = []
bias_test_free_energy = []
sdep_test_free_energy = []

for folders in glob.glob(f'{root_dir}/*/*_free_energy.txt'):

    with open(f'{folders}', 'r') as f:
        lines = f.readlines()		
        r2_line_test = [line for line in lines if line.startswith('r2_test')]
        rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
        bias_line_test = [line for line in lines if line.startswith('bias_test')]		
        sdep_line_test = [line for line in lines if line.startswith('sdep_test')]

        r2_test_free_energy.append(float(r2_line_test[0].split()[1]))
        rmsd_test_free_energy.append(float(rmsd_line_test[0].split()[1]))
        bias_test_free_energy.append(float(bias_line_test[0].split()[1]))
        sdep_test_free_energy.append(float(sdep_line_test[0].split()[1]))

        r2_line_train = [line for line in lines if line.startswith('r2_train')]
        rmsd_line_train = [line for line in lines if line.startswith('rmsd_train')]
        bias_line_train = [line for line in lines if line.startswith('bias_train')]		
        sdep_line_train = [line for line in lines if line.startswith('sdep_train')]

        r2_train_free_energy.append(float(r2_line_train[0].split()[1]))
        rmsd_train_free_energy.append(float(rmsd_line_train[0].split()[1]))
        bias_train_free_energy.append(float(bias_line_train[0].split()[1]))
        sdep_train_free_energy.append(float(sdep_line_train[0].split()[1]))

        r2_line_val = [line for line in lines if line.startswith('r2_val')]
        rmsd_line_val = [line for line in lines if line.startswith('rmsd_val')]
        bias_line_val = [line for line in lines if line.startswith('bias_val')]		
        sdep_line_val = [line for line in lines if line.startswith('sdep_val')]

        r2_val_free_energy.append(float(r2_line_val[0].split()[1]))
        rmsd_val_free_energy.append(float(rmsd_line_val[0].split()[1]))
        bias_val_free_energy.append(float(bias_line_val[0].split()[1]))
        sdep_val_free_energy.append(float(sdep_line_val[0].split()[1]))

r2_train_avg_free_energy = round(sum(r2_train_free_energy) / len(r2_train_free_energy), 3)
rmsd_train_avg_free_energy = round(sum(rmsd_train_free_energy) / len(rmsd_train_free_energy), 3)
bias_train_avg_free_energy = round(sum(bias_train_free_energy) / len(bias_train_free_energy), 3)
sdep_train_avg_free_energy = round(sum(sdep_train_free_energy) / len(sdep_train_free_energy), 3)

r2_test_avg_free_energy = round(sum(r2_test_free_energy) / len(r2_test_free_energy), 3)
rmsd_test_avg_free_energy = round(sum(rmsd_test_free_energy) / len(rmsd_test_free_energy), 3)
bias_test_avg_free_energy = round(sum(bias_test_free_energy) / len(bias_test_free_energy), 3)
sdep_test_avg_free_energy = round(sum(sdep_test_free_energy) / len(sdep_test_free_energy), 3)

r2_val_avg_free_energy = round(sum(r2_val_free_energy) / len(r2_val_free_energy), 3)
rmsd_val_avg_free_energy = round(sum(rmsd_val_free_energy) / len(rmsd_val_free_energy), 3)
bias_val_avg_free_energy = round(sum(bias_val_free_energy) / len(bias_val_free_energy), 3)
sdep_val_avg_free_energy = round(sum(sdep_val_free_energy) / len(sdep_val_free_energy), 3)

results_file = open(f'{root_dir}/CV_avg_stats_against_resamples_free_energy.txt', 'w')
results_file.write(f'r2_train_avg: {r2_train_avg_free_energy}')
results_file.write('\n')
results_file.write(f'rmsd_train_avg: {rmsd_train_avg_free_energy}')
results_file.write('\n')
results_file.write(f'bias_train_avg: {bias_train_avg_free_energy}')
results_file.write('\n')
results_file.write(f'sdep_train_avg: {sdep_train_avg_free_energy}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_val_avg: {r2_val_avg_free_energy}')
results_file.write('\n')
results_file.write(f'rmsd_val_avg: {rmsd_val_avg_free_energy}')
results_file.write('\n')
results_file.write(f'bias_val_avg: {bias_val_avg_free_energy}')
results_file.write('\n')
results_file.write(f'sdep_val_avg: {sdep_val_avg_free_energy}')
results_file.write('\n')
results_file.write('\n')
results_file.write(f'r2_test_avg: {r2_test_avg_free_energy}')
results_file.write('\n')
results_file.write(f'rmsd_test_avg: {rmsd_test_avg_free_energy}')
results_file.write('\n')
results_file.write(f'bias_test_avg: {bias_test_avg_free_energy}')
results_file.write('\n')
results_file.write(f'sdep_test_avg: {sdep_test_avg_free_energy}')

### Output standard deviation of average test set stats

r2_test_enthalpy = []
rmsd_test_enthalpy = []
bias_test_enthalpy = []
sdep_test_enthalpy = []         

for resample in range(1, no_resamples + 1):

    with open(f'{root_dir}/resample_{resample}/resample_{resample}/resample_{resample}_stats_enthalpy.txt', 'r') as f:
        lines = f.readlines()
        r2_line_test = [line for line in lines if line.startswith('r2_test')]
        rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
        bias_line_test = [line for line in lines if line.startswith('bias_test')]		
        sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
        r2_test_enthalpy.append(float(r2_line_test[0].split()[1]))
        rmsd_test_enthalpy.append(float(rmsd_line_test[0].split()[1]))
        bias_test_enthalpy.append(float(bias_line_test[0].split()[1]))
        sdep_test_enthalpy.append(float(sdep_line_test[0].split()[1]))

sd_r2_test = stats.stdev(r2_test_enthalpy)
sd_rmsd_test = stats.stdev(rmsd_test_enthalpy)
sd_bias_test = stats.stdev(bias_test_enthalpy)
sd_sdep_test = stats.stdev(sdep_test_enthalpy)
sd_r2_test = '{:.3f}'.format(sd_r2_test)
sd_rmsd_test = '{:.3f}'.format(sd_rmsd_test)
sd_bias_test = '{:.3f}'.format(sd_bias_test)
sd_sdep_test = '{:.3f}'.format(sd_sdep_test)
sd_file = open(f'{root_dir}/sd_test_stats_enthalpy.txt', 'w')
sd_file.write(f'sd_r2_test: {sd_r2_test}')
sd_file.write('\n')
sd_file.write(f'sd_rmsd_test: {sd_rmsd_test}')
sd_file.write('\n')
sd_file.write(f'sd_bias_test: {sd_bias_test}')
sd_file.write('\n')
sd_file.write(f'sd_sdep_test: {sd_sdep_test}')
sd_file.close()

r2_test_entropy = []
rmsd_test_entropy = []
bias_test_entropy = []
sdep_test_entropy = []         

for resample in range(1, no_resamples + 1):

    with open(f'{root_dir}/resample_{resample}/resample_{resample}/resample_{resample}_stats_entropy.txt', 'r') as f:
        lines = f.readlines()
        r2_line_test = [line for line in lines if line.startswith('r2_test')]
        rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
        bias_line_test = [line for line in lines if line.startswith('bias_test')]		
        sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
        r2_test_entropy.append(float(r2_line_test[0].split()[1]))
        rmsd_test_entropy.append(float(rmsd_line_test[0].split()[1]))
        bias_test_entropy.append(float(bias_line_test[0].split()[1]))
        sdep_test_entropy.append(float(sdep_line_test[0].split()[1]))

sd_r2_test = stats.stdev(r2_test_entropy)
sd_rmsd_test = stats.stdev(rmsd_test_entropy)
sd_bias_test = stats.stdev(bias_test_entropy)
sd_sdep_test = stats.stdev(sdep_test_entropy)
sd_r2_test = '{:.3f}'.format(sd_r2_test)
sd_rmsd_test = '{:.3f}'.format(sd_rmsd_test)
sd_bias_test = '{:.3f}'.format(sd_bias_test)
sd_sdep_test = '{:.3f}'.format(sd_sdep_test)
sd_file = open(f'{root_dir}/sd_test_stats_entropy.txt', 'w')
sd_file.write(f'sd_r2_test: {sd_r2_test}')
sd_file.write('\n')
sd_file.write(f'sd_rmsd_test: {sd_rmsd_test}')
sd_file.write('\n')
sd_file.write(f'sd_bias_test: {sd_bias_test}')
sd_file.write('\n')
sd_file.write(f'sd_sdep_test: {sd_sdep_test}')
sd_file.close()

r2_test_free_energy = []
rmsd_test_free_energy = []
bias_test_free_energy = []
sdep_test_free_energy = []         

for resample in range(1, no_resamples + 1):

    with open(f'{root_dir}/resample_{resample}/resample_{resample}/resample_{resample}_stats_free_energy.txt', 'r') as f:
        lines = f.readlines()
        r2_line_test = [line for line in lines if line.startswith('r2_test')]
        rmsd_line_test = [line for line in lines if line.startswith('rmsd_test')]
        bias_line_test = [line for line in lines if line.startswith('bias_test')]		
        sdep_line_test = [line for line in lines if line.startswith('sdep_test')]
        r2_test_free_energy.append(float(r2_line_test[0].split()[1]))
        rmsd_test_free_energy.append(float(rmsd_line_test[0].split()[1]))
        bias_test_free_energy.append(float(bias_line_test[0].split()[1]))
        sdep_test_free_energy.append(float(sdep_line_test[0].split()[1]))

sd_r2_test = stats.stdev(r2_test_free_energy)
sd_rmsd_test = stats.stdev(rmsd_test_free_energy)
sd_bias_test = stats.stdev(bias_test_free_energy)
sd_sdep_test = stats.stdev(sdep_test_free_energy)
sd_r2_test = '{:.3f}'.format(sd_r2_test)
sd_rmsd_test = '{:.3f}'.format(sd_rmsd_test)
sd_bias_test = '{:.3f}'.format(sd_bias_test)
sd_sdep_test = '{:.3f}'.format(sd_sdep_test)
sd_file = open(f'{root_dir}/sd_test_stats_free_energy.txt', 'w')
sd_file.write(f'sd_r2_test: {sd_r2_test}')
sd_file.write('\n')
sd_file.write(f'sd_rmsd_test: {sd_rmsd_test}')
sd_file.write('\n')
sd_file.write(f'sd_bias_test: {sd_bias_test}')
sd_file.write('\n')
sd_file.write(f'sd_sdep_test: {sd_sdep_test}')
sd_file.close()

### Output average predictions over resamples for each solute (instead of over the full set of SFE predictions)

input_dataset['Mol_Temp'] = input_dataset['Mol'] + '_' + input_dataset['Temp'].astype(str)
mol_temp_list = input_dataset['Mol_Temp'].tolist()
len_list = len(mol_temp_list)
summary_df = pd.DataFrame(columns = range(len_list))
summary_df.set_axis(mol_temp_list, axis='columns', inplace=True)
                                
df_list = []
for column in summary_df:
                            
    tmp_enthalpy_pred_list = []
    tmp_entropy_pred_list = []
    tmp_free_energy_pred_list = []
    tmp_enthalpy_exp_list = []
    tmp_entropy_exp_list = []
    tmp_free_energy_exp_list = []
    for resample in os.listdir(f'{root_dir}'):
        if os.path.isdir(f'{root_dir}'+'/'+resample):
            resample_test_pred_df = pd.read_csv(f'{root_dir}/{resample}/{resample}/{resample}_test_plot.csv')
            resample_test_pred_df['Mol_Temp'] = resample_test_pred_df['Mol'] + '_' + resample_test_pred_df['Temp'].astype(str)
            enthalpy_pred = (resample_test_pred_df.loc[resample_test_pred_df['Mol_Temp'] == column, 'y_enthalpy_pred'])
            enthalpy_exp = (resample_test_pred_df.loc[resample_test_pred_df['Mol_Temp'] == column, 'y_enthalpy'])
            entropy_pred = (resample_test_pred_df.loc[resample_test_pred_df['Mol_Temp'] == column, 'y_entropy_pred'])
            entropy_exp = (resample_test_pred_df.loc[resample_test_pred_df['Mol_Temp'] == column, 'y_entropy'])
            free_energy_pred = (resample_test_pred_df.loc[resample_test_pred_df['Mol_Temp'] == column, 'y_free_energy_pred'])
            free_energy_exp = (resample_test_pred_df.loc[resample_test_pred_df['Mol_Temp'] == column, 'y_free_energy'])
                                    
            if enthalpy_pred.shape == (0,):
                pass
            else:
                enthalpy_pred_df = enthalpy_pred.to_frame()
                entropy_pred_df = entropy_pred.to_frame()
                free_energy_pred_df = free_energy_pred.to_frame()

                enthalpy_pred_value = enthalpy_pred_df['y_enthalpy_pred'].iloc[0]
                entropy_pred_value = entropy_pred_df['y_entropy_pred'].iloc[0]
                free_energy_pred_value = free_energy_pred_df['y_free_energy_pred'].iloc[0]

                tmp_enthalpy_pred_list.append(enthalpy_pred_value)
                tmp_entropy_pred_list.append(entropy_pred_value)
                tmp_free_energy_pred_list.append(free_energy_pred_value)

                enthalpy_exp_df = enthalpy_exp.to_frame()
                entropy_exp_df = entropy_exp.to_frame()
                free_energy_exp_df = free_energy_exp.to_frame()

                enthalpy_exp_value = enthalpy_exp_df['y_enthalpy'].iloc[0]
                entropy_exp_value = entropy_exp_df['y_entropy'].iloc[0]
                free_energy_exp_value = free_energy_exp_df['y_free_energy'].iloc[0]

                tmp_enthalpy_exp_list.append(enthalpy_exp_value)
                tmp_entropy_exp_list.append(entropy_exp_value)
                tmp_free_energy_exp_list.append(free_energy_exp_value)
                         
    try:
        tmp_enthalpy_exp_list = tmp_enthalpy_exp_list[0]
        tmp_entropy_exp_list = tmp_entropy_exp_list[0]
        tmp_free_energy_exp_list = tmp_free_energy_exp_list[0]

    except IndexError:
         pass
    if not tmp_enthalpy_exp_list:
        continue
    avg_tmp_enthalpy_pred = sum(tmp_enthalpy_pred_list) / len(tmp_enthalpy_pred_list)
    avg_tmp_entropy_pred = sum(tmp_entropy_pred_list) / len(tmp_entropy_pred_list)
    avg_tmp_free_energy_pred = sum(tmp_free_energy_pred_list) / len(tmp_free_energy_pred_list)
    save_to_csv = [column, tmp_enthalpy_exp_list, avg_tmp_enthalpy_pred, tmp_entropy_exp_list, avg_tmp_entropy_pred, tmp_free_energy_exp_list, avg_tmp_free_energy_pred]
    csv_header = ['Mol', 'y_enthalpy', 'y_enthalpy_pred', 'y_entropy', 'y_entropy_pred', 'y_free_energy', 'y_free_energy_pred']
    df = pd.DataFrame([save_to_csv], columns=csv_header)
    df_list.append(df)

df2 = pd.concat(df_list)
df2.to_csv(f'{root_dir}/avg_preds_over_resamples.csv')

# Enthalpy
y_test = df2['y_enthalpy']
y_pred_test = df2['y_enthalpy_pred']
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
results_file = open(f'{root_dir}/avg_pred_over_resamples_stats_enthalpy.txt', 'w')
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
    a, b = np.polyfit(df2['y_enthalpy'], df2['y_enthalpy_pred'], 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    plt.plot([], [], ' ', label=f'Test set '+r'$ΔH_{solv}$')
    plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_test}')
    plt.plot([], [], ' ', label=f'RMSD : {rmsd_test}')
    plt.plot([], [], ' ', label=f'Bias : {bias_test}')
    plt.plot([], [], ' ', label=f'SDEP : {sdep_test}')
    plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(df2['y_enthalpy'], df2['y_enthalpy_pred'])
    plt.plot(df2['y_enthalpy'], a * df2['y_enthalpy'] + b, color='purple')
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
plt.xlabel('$ΔH^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔH^{calc}_{solv}$ (kcal/mol)')
handles, labels = plt.gca().get_legend_handles_labels()
leg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/avg_pred_over_resamples_plot_with_pred_avg_stats_enthalpy.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

with open(f'{root_dir}/resample_avg_stats_enthalpy.txt', 'r') as f:
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
    a, b = np.polyfit(df2['y_enthalpy'], df2['y_enthalpy_pred'], 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
   plt.plot([], [], ' ', label=f'Test set '+r'$ΔH_{solv}$')
   plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_test_avg}')
   plt.plot([], [], ' ', label=f'RMSD : {rmsd_test_avg}')
   plt.plot([], [], ' ', label=f'Bias : {bias_test_avg}')
   plt.plot([], [], ' ', label=f'SDEP : {sdep_test_avg}')
   plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
   plt.scatter(df2['y_enthalpy'], df2['y_enthalpy_pred'])
   plt.plot(df2['y_enthalpy'], a * df2['y_enthalpy'] + b, color='purple')
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
plt.xlabel('$ΔH^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$ΔH^{calc}_{solv}$ (kcal/mol)')
handles, labels = plt.gca().get_legend_handles_labels()
leg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/avg_pred_over_resamples_plot_with_resample_avg_stats_enthalpy.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# Entropy
y_test = df2['y_entropy']
y_pred_test = df2['y_entropy_pred']
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
results_file = open(f'{root_dir}/avg_pred_over_resamples_stats_entropy.txt', 'w')
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
    a, b = np.polyfit(df2['y_entropy'], df2['y_entropy_pred'], 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    plt.plot([], [], ' ', label=f'Test set '+r'$TΔS_{solv}$')
    plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_test}')
    plt.plot([], [], ' ', label=f'RMSD : {rmsd_test}')
    plt.plot([], [], ' ', label=f'Bias : {bias_test}')
    plt.plot([], [], ' ', label=f'SDEP : {sdep_test}')
    plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(df2['y_entropy'], df2['y_entropy_pred'])
    plt.plot(df2['y_entropy'], a * df2['y_entropy'] + b, color='purple')
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
plt.xlabel('$TΔS^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$TΔS^{calc}_{solv}$ (kcal/mol)')
handles, labels = plt.gca().get_legend_handles_labels()
leg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/avg_pred_over_resamples_plot_with_pred_avg_stats_entropy.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

with open(f'{root_dir}/resample_avg_stats_entropy.txt', 'r') as f:
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
    a, b = np.polyfit(df2['y_entropy'], df2['y_entropy_pred'], 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
   plt.plot([], [], ' ', label=f'Test set '+r'$TΔS_{solv}$')
   plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_test_avg}')
   plt.plot([], [], ' ', label=f'RMSD : {rmsd_test_avg}')
   plt.plot([], [], ' ', label=f'Bias : {bias_test_avg}')
   plt.plot([], [], ' ', label=f'SDEP : {sdep_test_avg}')
   plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
   plt.scatter(df2['y_entropy'], df2['y_entropy_pred'])
   plt.plot(df2['y_entropy'], a * df2['y_entropy'] + b, color='purple')
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
plt.xlabel('$TΔS^{exp}_{solv}$ (kcal/mol)')
plt.ylabel('$TΔS^{calc}_{solv}$ (kcal/mol)')
handles, labels = plt.gca().get_legend_handles_labels()
leg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
leg.get_frame().set_linewidth(0.0)
fig.savefig(f'{root_dir}/avg_pred_over_resamples_plot_with_resample_avg_stats_entropy.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

# Free energy
y_test = df2['y_free_energy']
y_pred_test = df2['y_free_energy_pred']
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
results_file = open(f'{root_dir}/avg_pred_over_resamples_stats_free_energy.txt', 'w')
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
    a, b = np.polyfit(df2['y_free_energy'], df2['y_free_energy_pred'], 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
    plt.plot([], [], ' ', label=f'Test set '+r'$ΔG_{solv}$')
    plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_test}')
    plt.plot([], [], ' ', label=f'RMSD : {rmsd_test}')
    plt.plot([], [], ' ', label=f'Bias : {bias_test}')
    plt.plot([], [], ' ', label=f'SDEP : {sdep_test}')
    plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
    plt.scatter(df2['y_free_energy'], df2['y_free_energy_pred'])
    plt.plot(df2['y_free_energy'], a * df2['y_free_energy'] + b, color='purple')
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
fig.savefig(f'{root_dir}/avg_pred_over_resamples_plot_with_pred_avg_stats_free_energy.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)

with open(f'{root_dir}/resample_avg_stats_free_energy.txt', 'r') as f:
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
    a, b = np.polyfit(df2['y_free_energy'], df2['y_free_energy_pred'], 1)
    plot_a = '{:.3f}'.format(a)
    plot_b = '{:.3f}'.format(b)
except np.linalg.LinAlgError:
    pass

# Plot everything
try:
   plt.plot([], [], ' ', label=f'Test set '+r'$ΔG_{solv}$')
   plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_test_avg}')
   plt.plot([], [], ' ', label=f'RMSD : {rmsd_test_avg}')
   plt.plot([], [], ' ', label=f'Bias : {bias_test_avg}')
   plt.plot([], [], ' ', label=f'SDEP : {sdep_test_avg}')
   plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
   plt.scatter(df2['y_free_energy'], df2['y_free_energy_pred'])
   plt.plot(df2['y_free_energy'], a * df2['y_free_energy'] + b, color='purple')
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
fig.savefig(f'{root_dir}/avg_pred_over_resamples_plot_with_resample_avg_stats_free_energy.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()
plt.close(fig)


