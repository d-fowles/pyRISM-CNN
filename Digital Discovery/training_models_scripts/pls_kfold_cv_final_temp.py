import os
os.environ['QT_QPA_PLATFORM']='offscreen'

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
#from sklearn.inspection import permutation_importance

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import csv
import math
import statistics as stats
from scipy.stats import pearsonr, spearmanr
import glob

root_dir = os.getcwd()

### Prepare input data with a unique id for each molecule and its corresponding descr.
input_dataset = pd.read_csv(f'{root_dir}/csv_to_replace')
input_dataset = input_dataset.sort_values('Mol')
input_dataset = input_dataset.reset_index()
input_dataset = input_dataset.drop(['index'], axis=1)
df_id = input_dataset.groupby(['Mol']).ngroup()
input_dataset.insert(0, 'id', df_id)

### Specify SFED type
sfed_type = '' #kh, hnc or gf

### Start resampling loop
no_resamples = 50

# Set up variables to save all resamples statistics for each resample loop
r2_train_sum_resample = 0
rmsd_train_sum_resample = 0
bias_train_sum_resample = 0
sdep_train_sum_resample = 0

r2_test_sum_resample = 0
rmsd_test_sum_resample = 0
bias_test_sum_resample = 0
sdep_test_sum_resample = 0

# Lists to save resamples predictions
all_preds_train_resample = []
all_preds_test_resample = []

# Component used for manual hyperparameter tuning
component = 10

for resample in range(1, no_resamples + 1):
    
    # Make directory for each resample loop and dump all related files into it
    try:
        os.mkdir(f'{root_dir}/resample_{resample}')
    except FileExistsError:
        pass

    # Shuffle and split dataset along unique id for each resample into train and test sets
    input_dataset_shuffle = input_dataset.sample(frac=1).reset_index(drop=True)

    train_resample_frac = 0.7
    unique_resample_id = input_dataset_shuffle['id'].unique()
    train_resample_idx = input_dataset_shuffle['id'].isin(unique_resample_id[:round(train_resample_frac * len(unique_resample_id))])
    train_resample = input_dataset_shuffle.loc[train_resample_idx, :]
    test_resample = input_dataset_shuffle.loc[~train_resample_idx, :]

    ### Start CV set up and loop
    no_folds = 5

    # Set up variables to save all CV fold statistics for this resample
    r2_train_sum_cv = 0
    rmsd_train_sum_cv = 0
    bias_train_sum_cv = 0
    sdep_train_sum_cv = 0

    r2_test_sum_cv = 0
    rmsd_test_sum_cv= 0
    bias_test_sum_cv = 0
    sdep_test_sum_cv = 0

    # Lists to save CV fold predictions
    all_preds_train_cv = []
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
        train_fold = other_folds
        test_fold = fold

        ### Prepare data into X and y for each fold
        # Take exp HFE as y input in dataframe format
        y_train_fold = train_fold[['HFE']].copy()

        y_train_fold_output = pd.DataFrame()
        y_train_fold_output['Mol'] = train_fold['Mol']
        y_train_fold_output['Temp'] = train_fold['Temp']
        y_train_fold_output['HFE'] = train_fold['HFE']
        y_train_fold_output.to_csv(f'resample_{resample}_fold_{i}_train_plot.csv', index=False)

        y_test_fold = test_fold[['HFE']].copy()

        y_test_fold_output = pd.DataFrame()
        y_test_fold_output['Mol'] = test_fold['Mol']
        y_test_fold_output['Temp'] = test_fold['Temp']
        y_test_fold_output['HFE'] = test_fold['HFE']
        y_test_fold_output.to_csv(f'resample_{resample}_fold_{i}_test_plot.csv', index=False)

        # Convert y data into required input shape
        y_train_cv = y_train_fold.to_numpy()

        y_test_cv = y_test_fold.to_numpy()

        # Take descr. columns as X data in dataframe format
        # Select all columns beginning with 'X_w_' as the X data
        # Uses temp as a descr.
        if sfed_type == 'hnc':
            X_descr_train = train_fold[[col for col in train_fold.columns if col[:6]==f'{sfed_type}_w_']]
            temp_train = train_fold['Temp']
            X_pd_train = pd.concat([temp_train, X_descr_train], axis=1)

            X_descr_test = test_fold[[col for col in test_fold.columns if col[:6]==f'{sfed_type}_w_']]
            temp_test = test_fold['Temp']
            X_pd_test = pd.concat([temp_test, X_descr_test], axis=1)

        else:
            X_descr_train = train_fold[[col for col in train_fold.columns if col[:5]==f'{sfed_type}_w_']]
            temp_train = train_fold['Temp']
            X_pd_train = pd.concat([temp_train, X_descr_train], axis=1)
            
            X_descr_test = test_fold[[col for col in test_fold.columns if col[:5]==f'{sfed_type}_w_']]
            temp_test = test_fold['Temp']
            X_pd_test = pd.concat([temp_test, X_descr_test], axis=1)

        # Scale train and use that to scale val and test
        # Temp
        means = X_pd_train.mean(axis=0)
        sds = X_pd_train.std(axis=0)

        X_pd_train_scaled = (X_pd_train - means) / sds
        X_pd_test_scaled = (X_pd_test - means) / sds

        # Convert X data into required input shape
        # Temp
        X_train_cv = X_pd_train_scaled.to_numpy()

        X_test_cv = X_pd_test_scaled.to_numpy()

        # Build model
        pls = PLSRegression(n_components=component, scale=False)

        # Train the model:
        # ----------------
        # Fit the model:
        pls.fit(X_train_cv, y_train_cv)       
            
        # Make predictions on train, val and test set using trained model:
        y_pred_train_cv = pls.predict(X_train_cv)
        y_pred_test_cv = pls.predict(X_test_cv)

        # Assess performace of model based on predictions:
        # Coefficient of determination
        r2_train_cv = r2_score(y_train_cv, y_pred_train_cv)
        r2_test_cv = r2_score(y_test_cv, y_pred_test_cv)
        # Root mean squared error
        rmsd_train_cv = (mean_squared_error(y_train_cv, y_pred_train_cv))**0.5
        rmsd_test_cv = (mean_squared_error(y_test_cv, y_pred_test_cv))**0.5
        # Bias
        bias_train_cv = np.mean(y_pred_train_cv - y_train_cv)
        bias_test_cv = np.mean(y_pred_test_cv - y_test_cv)
        # Standard deviation of the error of prediction
        sdep_train_cv = (np.mean((y_pred_train_cv - y_train_cv - bias_train_cv)**2))**0.5
        sdep_test_cv = (np.mean((y_pred_test_cv - y_test_cv - bias_test_cv)**2))**0.5

        # Save running sum of results:
        r2_train_sum_cv += r2_train_cv
        rmsd_train_sum_cv += rmsd_train_cv
        bias_train_sum_cv += bias_train_cv
        sdep_train_sum_cv += sdep_train_cv
 
        r2_test_sum_cv += r2_test_cv
        rmsd_test_sum_cv += rmsd_test_cv
        bias_test_sum_cv += bias_test_cv
        sdep_test_sum_cv+= sdep_test_cv

        # Save individual predictions for fold in dataframes
        y_pred_train_cv = y_pred_train_cv.reshape(y_pred_train_cv.shape[0])
        y_pred_train_cv = pd.DataFrame(y_pred_train_cv, columns=[f'fold_{i}'])
        y_pred_train_cv_labeled = pd.concat([train_fold['Mol'], y_pred_train_cv], axis=1)
        y_pred_train_cv_labeled[''] = ''
        all_preds_train_cv.append(y_pred_train_cv_labeled)

        y_pred_test_cv = y_pred_test_cv.reshape(y_pred_test_cv.shape[0])
        y_pred_test_cv = pd.DataFrame(y_pred_test_cv, columns=[f'fold_{i}'])
        y_pred_test_cv_labeled = pd.concat([test_fold['Mol'], y_pred_test_cv], axis=1)
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
            os.mkdir('individual_solvent_predictions/test')
        except FileExistsError:
            pass

        sets = ['train', 'test']

        for set in sets:

            df_chloroform = pd.read_csv(f'resample_{resample}_fold_{num}_{set}_plot.csv', index_col=False)
            df_carbontet = pd.read_csv(f'resample_{resample}_fold_{num}_{set}_plot.csv', index_col=False)
            df_water = pd.read_csv(f'resample_{resample}_fold_{num}_{set}_plot.csv', index_col=False)
                    
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
                if 'carbontet' in mol:
                    df_water.drop(df_water.loc[df_water['Mol']==mol].index, inplace=True)
                elif 'chloroform' in mol:
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

    y_test_resample = test_resample[['HFE']].copy()

    y_test_resample_output = pd.DataFrame()
    y_test_resample_output['Mol'] = test_resample['Mol']
    y_test_resample_output['Temp'] = test_resample['Temp']
    y_test_resample_output['HFE'] = test_resample['HFE']
    y_test_resample_output.to_csv(f'resample_{resample}_test_plot.csv', index=False)

    # Convert y data into required input shape
    y_train_resample = y_train_resample.to_numpy()

    y_test_resample = y_test_resample.to_numpy()

    # Take descr. columns as X data in dataframe format

    # Select all columns beginning with 'X_w_' as the X data
    # Uses temp as a descr.
    if sfed_type == 'hnc':
        X_descr_train = train_resample[[col for col in train_resample.columns if col[:6]==f'{sfed_type}_w_']]
        temp_train = train_resample['Temp']
        X_pd_train = pd.concat([temp_train, X_descr_train], axis=1)

        X_descr_test = test_resample[[col for col in test_resample.columns if col[:6]==f'{sfed_type}_w_']]
        temp_test = test_resample['Temp']
        X_pd_test = pd.concat([temp_test, X_descr_test], axis=1)

    else:
        X_descr_train = train_resample[[col for col in train_resample.columns if col[:5]==f'{sfed_type}_w_']]
        temp_train = train_resample['Temp']
        X_pd_train = pd.concat([temp_train, X_descr_train], axis=1)
        
        X_descr_test = test_resample[[col for col in test_resample.columns if col[:5]==f'{sfed_type}_w_']]
        temp_test = test_resample['Temp']
        X_pd_test = pd.concat([temp_test, X_descr_test], axis=1)

    # Scale train and use that to scale val and test
    # Temp
    means = X_pd_train.mean(axis=0)
    sds = X_pd_train.std(axis=0)

    X_pd_train_scaled = (X_pd_train - means) / sds
    X_pd_test_scaled = (X_pd_test - means) / sds

    # Convert X data into required input shape
    # Temp
    X_train_resample = X_pd_train_scaled.to_numpy()

    X_test_resample = X_pd_test_scaled.to_numpy()

    # Train the model:
    # ----------------
    # Fit the model:
    pls.fit(X_train_cv, y_train_cv)
        
    # Make predictions on train and test set using trained model:
    y_pred_train_resample = pls.predict(X_train_resample)
    y_pred_test_resample = pls.predict(X_test_resample)

    # Assess performace of model based on predictions:
    # Coefficient of determination
    r2_train_resample = r2_score(y_train_resample, y_pred_train_resample)
    r2_test_resample = r2_score(y_test_resample, y_pred_test_resample)
    # Root mean squared error
    rmsd_train_resample = (mean_squared_error(y_train_resample, y_pred_train_resample))**0.5
    rmsd_test_resample = (mean_squared_error(y_test_resample, y_pred_test_resample))**0.5
    # Bias
    bias_train_resample = np.mean(y_pred_train_resample - y_train_resample)
    bias_test_resample = np.mean(y_pred_test_resample - y_test_resample)
    # Standard deviation of the error of prediction
    sdep_train_resample = (np.mean((y_pred_train_resample - y_train_resample - bias_train_resample)**2))**0.5
    sdep_test_resample = (np.mean((y_pred_test_resample - y_test_resample - bias_test_resample)**2))**0.5

    # Save running sum of results:
    r2_train_sum_resample += r2_train_resample
    rmsd_train_sum_resample += rmsd_train_resample
    bias_train_sum_resample += bias_train_resample
    sdep_train_sum_resample += sdep_train_resample

    r2_test_sum_resample += r2_test_resample
    rmsd_test_sum_resample += rmsd_test_resample
    bias_test_sum_resample += bias_test_resample
    sdep_test_sum_resample += sdep_test_resample

    # Save individual predictions for resample in dataframes
    y_pred_train_resample = y_pred_train_resample.reshape(y_pred_train_resample.shape[0])
    y_pred_train_resample = pd.DataFrame(y_pred_train_resample, columns=[f'resample_{resample}'])
    y_pred_train_resample_labeled = pd.concat([train_resample['Mol'], y_pred_train_resample], axis=1)
    y_pred_train_resample_labeled[''] = ''
    all_preds_train_resample.append(y_pred_train_resample_labeled)

    y_pred_test_resample = y_pred_test_resample.reshape(y_pred_test_resample.shape[0])
    y_pred_test_resample = pd.DataFrame(y_pred_test_resample, columns=[f'resample_{resample}'])
    y_pred_test_resample_labeled = pd.concat([test_resample['Mol'], y_pred_test_resample], axis=1)
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

r2_test_av_resample = r2_test_sum_resample/no_resamples
rmsd_test_av_resample = rmsd_test_sum_resample/no_resamples
bias_test_av_resample = bias_test_sum_resample/no_resamples
sdep_test_av_resample = sdep_test_sum_resample/no_resamples

r2_test_av_resample = '{:.3f}'.format(r2_test_av_resample)
rmsd_test_av_resample = '{:.3f}'.format(rmsd_test_av_resample)
bias_test_av_resample = '{:.3f}'.format(bias_test_av_resample)
sdep_test_av_resample = '{:.3f}'.format(sdep_test_av_resample)

# Write average results to a file:
results_file = open(f'{root_dir}/resample_avg_stats.txt', 'w')
results_file.write(f'r2_train: {r2_train_av_resample}')
results_file.write('\n')
results_file.write(f'rmsd_train: {rmsd_train_av_resample}')
results_file.write('\n')
results_file.write(f'bias_train: {bias_train_av_resample}')
results_file.write('\n')
results_file.write(f'sdep_train: {sdep_train_av_resample}')
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
all_preds_train_resample = all_preds_train_resample.rename(columns={'Mol': f'Mol_{resample}'})
all_preds_train_resample = all_preds_train_resample.apply(lambda x: pd.Series(x.dropna().values))
all_preds_train_resample.to_csv(f'{root_dir}/resample_pred_train_avg.csv')

all_preds_test_resample = pd.concat(all_preds_test_resample, axis=1)
all_preds_test_resample['Mol'].replace('', np.nan, inplace=True)
all_preds_test_resample = all_preds_test_resample.rename(columns={'Mol': f'Mol_{resample}'})
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
        plt.plot([], [], ' ', label='Training set predictions')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_train_av_resample}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd_train_av_resample}')
        plt.plot([], [], ' ', label=f'Bias : {bias_train_av_resample}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep_train_av_resample}')
        plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
        plt.scatter(df2['y'], df2['y_pred'])
        order = [0,1,2,3,4,5]
    except NameError:
        plt.plot([], [], ' ', label='Training set predictions')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_train_av_resample}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd_train_av_resample}')
        plt.plot([], [], ' ', label=f'Bias : {bias_train_av_resample}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep_train_av_resample}')
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
        plt.plot([], [], ' ', label='Test set predictions')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_test_av_resample}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd_test_av_resample}')
        plt.plot([], [], ' ', label=f'Bias : {bias_test_av_resample}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep_test_av_resample}')
        plt.plot([], [], ' ', label=f'y = {plot_a}x + {plot_b}')
        plt.scatter(df2['y'], df2['y_pred'])
        order = [0,1,2,3,4,5]
    except NameError:
        plt.plot([], [], ' ', label='Test set predictions')
        plt.plot([], [], ' ', label=f'$R^{2}$ : {r2_test_av_resample}')
        plt.plot([], [], ' ', label=f'RMSD : {rmsd_test_av_resample}')
        plt.plot([], [], ' ', label=f'Bias : {bias_test_av_resample}')
        plt.plot([], [], ' ', label=f'SDEP : {sdep_test_av_resample}')
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
        os.mkdir('individual_solvent_predictions/test')
    except FileExistsError:
        pass

    sets = ['train', 'test']

    for set in sets:

        df_chloroform = pd.read_csv(f'resample_{resample}_{set}_plot.csv', index_col=False)
        df_carbontet = pd.read_csv(f'resample_{resample}_{set}_plot.csv', index_col=False)
        df_water = pd.read_csv(f'resample_{resample}_{set}_plot.csv', index_col=False)
                    
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
            if 'carbontet' in mol:
                df_water.drop(df_water.loc[df_water['Mol']==mol].index, inplace=True)
            elif 'chloroform' in mol:
                df_water.drop(df_water.loc[df_water['Mol']==mol].index, inplace=True)

        df_water.reset_index(inplace=True)
        df_water.drop(columns = ['index'], inplace=True)
        df_water.to_csv(f'individual_solvent_predictions/{set}/resample_{resample}_{set}_water_pred.csv')

### Compare CV average results against resample average results, should be similar

r2_train = []
rmsd_train = []
bias_train = []
sdep_train = []

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

r2_train_avg = round(sum(r2_train) / len(r2_train), 3)
rmsd_train_avg = round(sum(rmsd_train) / len(rmsd_train), 3)
bias_train_avg = round(sum(bias_train) / len(bias_train), 3)
sdep_train_avg = round(sum(sdep_train) / len(sdep_train), 3)

r2_test_avg = round(sum(r2_test) / len(r2_test), 3)
rmsd_test_avg = round(sum(rmsd_test) / len(rmsd_test), 3)
bias_test_avg = round(sum(bias_test) / len(bias_test), 3)
sdep_test_avg = round(sum(sdep_test) / len(sdep_test), 3)

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
chloroform_r2_list = []
chloroform_rmsd_list = []
chloroform_bias_list = []
chloroform_sdep_list = []
carbontet_r2_list = []
carbontet_rmsd_list = []
carbontet_bias_list = []
carbontet_sdep_list = []

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

            if solvent == 'chloroform':
                chloroform_r2_list.append(r2)
                chloroform_rmsd_list.append(rmsd)
                chloroform_bias_list.append(bias)
                chloroform_sdep_list.append(sdep)
            elif solvent == 'carbontet':
                carbontet_r2_list.append(r2)
                carbontet_rmsd_list.append(rmsd)
                carbontet_bias_list.append(bias)
                carbontet_sdep_list.append(sdep)
            else:
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

water_r2 = sum(water_r2_list) / len(water_r2_list)
water_rmsd = sum(water_rmsd_list) / len(water_rmsd_list)
water_bias = sum(water_bias_list) / len(water_bias_list)
water_sdep = sum(water_sdep_list) / len(water_sdep_list)
chloroform_r2 = sum(chloroform_r2_list) / len(chloroform_r2_list)
chloroform_rmsd = sum(chloroform_rmsd_list) / len(chloroform_rmsd_list)
chloroform_bias = sum(chloroform_bias_list) / len(chloroform_bias_list)
chloroform_sdep = sum(chloroform_sdep_list) / len(chloroform_sdep_list)
carbontet_r2 = sum(carbontet_r2_list) / len(carbontet_r2_list)
carbontet_rmsd = sum(carbontet_rmsd_list) / len(carbontet_rmsd_list)
carbontet_bias = sum(carbontet_bias_list) / len(carbontet_bias_list)
carbontet_sdep = sum(carbontet_sdep_list) / len(carbontet_sdep_list)

water_r2 = '{:.3f}'.format(water_r2)
water_rmsd = '{:.3f}'.format(water_rmsd)
water_bias = '{:.3f}'.format(water_bias)
water_sdep = '{:.3f}'.format(water_sdep)
chloroform_r2 = '{:.3f}'.format(chloroform_r2)
chloroform_rmsd = '{:.3f}'.format(chloroform_rmsd)
chloroform_bias = '{:.3f}'.format(chloroform_bias)
chloroform_sdep = '{:.3f}'.format(chloroform_sdep)
carbontet_r2 = '{:.3f}'.format(carbontet_r2)
carbontet_rmsd = '{:.3f}'.format(carbontet_rmsd)
carbontet_bias = '{:.3f}'.format(carbontet_bias)
carbontet_sdep = '{:.3f}'.format(carbontet_sdep)

avg_water_file = open(f'{root_dir}/avg_resample_test_stats_water.txt', 'w')
avg_water_file.write(f'r2_test: {water_r2}')
avg_water_file.write('\n')
avg_water_file.write(f'rmsd_test: {water_rmsd}')
avg_water_file.write('\n')
avg_water_file.write(f'bias_test: {water_bias}')
avg_water_file.write('\n')
avg_water_file.write(f'sdep_test: {water_sdep}')
avg_water_file.close()

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
results_file = open(f'{root_dir}/avg_pred_over_resamples_stats.txt', 'w')
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
    plt.plot([], [], ' ', label=f'Test set predictions')
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

with open(f'{root_dir}/resample_avg_stats.txt', 'r') as f:
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
   plt.plot([], [], ' ', label=f'Test set predictions')
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






 







