# General imports
#import cv2

#### FOR MAC OSX USERS
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pydot
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Project imports
from utils.images_loader import ImagesLoader
from utils.text_data_loader import TextDataLoader
from utils.data_transformer import HAL9001DataTransformer
from utils.utilities import plot_history
from utils.utilities import rmsle
from utils.utilities import load_x_y_from_loaders



# Names for acces to data in Dict (to be used with model inputs/output for better referencing)
IMAGE_INPUT_NAME = 'image'
TEXT_FEATURES_INPUT_NAME = 'text_features'
OUTPUT_NAME = 'likes'

# params
use_scaling = True
include_images = False


# data paths
train_text_path = r'./src_data/train.csv'
train_images_folder = r'./src_data/train_profile_images'
test_text_path = r'./src_data/test.csv'
test_images_folder = r'./src_data/test_profile_images'
log_folder = 'logs'


def main():
    # Create log folder if does not exist
    if not Path(log_folder).exists():
        os.mkdir(log_folder)


    # -- Load TRAIN set --
    # Images loader
    print('\nTRAIN:')
    images_loader = ImagesLoader(src_folder_path = train_images_folder)
    print('Number images: {:}'.format(images_loader.nbr_images))
    print('Images shape: {:}'.format(images_loader.image_shape))
    # Text data loader
    text_data_loader = TextDataLoader(src_csv_file_path = train_text_path)

    # -- Load TEST set --
    # Test Images loader
    print('\nTEST:')
    test_images_loader = ImagesLoader(src_folder_path = test_images_folder)
    print('Number images: {:}'.format(test_images_loader.nbr_images))
    print('Images shape: {:}'.format(test_images_loader.image_shape))
    # Test Text data loader
    test_text_data_loader = TextDataLoader(src_csv_file_path = test_text_path)
    
    # -- Data Transformer --
    data_transformer = HAL9001DataTransformer()


    # -- Split ids to train/valid --
    data_X, data_y = load_x_y_from_loaders(images_loader=images_loader,
                                           text_data_loader=text_data_loader,
                                           data_transformer=data_transformer,
                                           transform_only=False, # TODO Fit and Transform
                                           image_input_name=IMAGE_INPUT_NAME,
                                           text_features_input_name=TEXT_FEATURES_INPUT_NAME,
                                           output_name=OUTPUT_NAME,
                                           profiles_ids_list=None, # Load all profiles in data
                                           include_images=include_images)

    if include_images:
        train_X1, valid_X1, train_X2, valid_X2, train_y, valid_y = train_test_split(data_X[IMAGE_INPUT_NAME],
                                                                                 data_X[TEXT_FEATURES_INPUT_NAME],
                                                                                data_y[OUTPUT_NAME],
                                                                                test_size = .2,
                                                                                random_state=42,
                                                                                shuffle=True)

        # Put data back as dict (Need this for NN)
        train_X=dict({IMAGE_INPUT_NAME: train_X1,
                    TEXT_FEATURES_INPUT_NAME:train_X2})
        train_y=dict({OUTPUT_NAME: train_y})
        valid_X=dict({IMAGE_INPUT_NAME: valid_X1,
                    TEXT_FEATURES_INPUT_NAME:valid_X2})
        valid_y=dict({OUTPUT_NAME: valid_y})
    else:
        train_X, valid_X, train_y, valid_y = train_test_split(data_X[TEXT_FEATURES_INPUT_NAME],
                                                              data_y[OUTPUT_NAME],
                                                              test_size = .2,
                                                              random_state=42,
                                                              shuffle=True)
        # Put data back as dict (Need this for NN)
        train_X=dict({TEXT_FEATURES_INPUT_NAME:train_X})
        train_y=dict({OUTPUT_NAME: train_y})
        valid_X=dict({TEXT_FEATURES_INPUT_NAME:valid_X})
        valid_y=dict({OUTPUT_NAME: valid_y})


    # TODO using scaling
    if use_scaling:
        sc_x = StandardScaler()
        pt_cox_y = PowerTransformer(method='box-cox', standardize=False)
    
        # Fit and Transform on Train
        train_X[TEXT_FEATURES_INPUT_NAME] = sc_x.fit_transform(train_X[TEXT_FEATURES_INPUT_NAME])
        train_y[OUTPUT_NAME] = pt_cox_y.fit_transform((train_y[OUTPUT_NAME] + 1).reshape(-1, 1))
    
        # Transform only on Valid
        valid_X[TEXT_FEATURES_INPUT_NAME] = sc_x.transform(valid_X[TEXT_FEATURES_INPUT_NAME])
        valid_y[OUTPUT_NAME] = pt_cox_y.transform((valid_y[OUTPUT_NAME] + 1).reshape(-1, 1))

    

    # -- DEBUG linear regression on text features --
    from sklearn.linear_model import LinearRegression
    X= train_X[TEXT_FEATURES_INPUT_NAME]
    y= train_y[OUTPUT_NAME]
    reg = LinearRegression().fit(X, y)
    print('\nLinear reg score: {:}'.format(reg.score(X, y)))
    # -- DEBUG --


    # -- Prepare train/valid sets for non NN models --
    print('\n\nPredicting on validation set')
    train_values_X= train_X[TEXT_FEATURES_INPUT_NAME]
    train_values_y= train_y[OUTPUT_NAME].reshape(-1)

    valid_values_X= valid_X[TEXT_FEATURES_INPUT_NAME]
    valid_values_y= valid_y[OUTPUT_NAME].reshape(-1)

    print('Train shape {:} {:}'.format(train_values_X.shape, train_values_y.shape))
    print('Valid shape {:} {:}'.format(valid_values_X.shape, valid_values_y.shape))


    # -- GradientBoostingRegressor on train/valid --
    # With defaut values
    print('GradientBoostingRegressor on train/valid')
    gbr = GradientBoostingRegressor(random_state=42)

    gbr.fit(train_values_X, train_values_y)
    valid_pred_y = gbr.predict(valid_values_X)

    # Evaluation
    # Scale predicted values back (-1 because we added +1 before cox-box tansform)
    scaled_valid_y = (pt_cox_y.inverse_transform(valid_values_y.reshape(-1,1)) - 1).astype(int)
    scaled_valid_pred_y = (pt_cox_y.inverse_transform(valid_pred_y.reshape(-1,1)) - 1).astype(int)

    # Make sure predicted and scaled back values are not negatives!
    idexes = np.nonzero(scaled_valid_pred_y < 0)
    scaled_valid_pred_y[idexes] = 0

    # Compute rmse and rmsle
    rmse_val = np.sqrt(mean_squared_error(scaled_valid_pred_y, scaled_valid_y))
    print('Validation sqrt mse: {:}'.format(rmse_val))
    rmsle_val = rmsle(scaled_valid_pred_y, scaled_valid_y)
    print('Validation rmsle: {:}'.format(rmsle_val))








    # -- Predict on Test set ---
    print('\n\nPredicting on test set')
    test_profiles_ids_list = test_text_data_loader.get_orig_features()['Id'].values
    test_X = load_x_y_from_loaders(images_loader=test_images_loader,
                                   text_data_loader=test_text_data_loader,
                                   data_transformer=data_transformer,
                                   transform_only=False, # TODO Transform only (Not implemented yet, use fit_transform)
                                   image_input_name=IMAGE_INPUT_NAME,
                                   text_features_input_name=TEXT_FEATURES_INPUT_NAME,
                                   output_name=OUTPUT_NAME,
                                   profiles_ids_list=test_profiles_ids_list,
                                   include_images=include_images)


    # TODO using scaling
    if use_scaling:
        sc_x = StandardScaler()
        pt_cox_y = PowerTransformer(method='box-cox', standardize=False)
    
        # Fit and Transform on Data (all train set)
        data_X[TEXT_FEATURES_INPUT_NAME] = sc_x.fit_transform(data_X[TEXT_FEATURES_INPUT_NAME])
        data_y[OUTPUT_NAME] = pt_cox_y.fit_transform((data_y[OUTPUT_NAME] + 1).reshape(-1, 1))
    
        # Transform only on Test
        test_X[TEXT_FEATURES_INPUT_NAME] = sc_x.transform(test_X[TEXT_FEATURES_INPUT_NAME])


    # -- Prepare Alltrain data/Test set for non NN models --
    alldataset_X= data_X[TEXT_FEATURES_INPUT_NAME]
    alldataset_y= data_y[OUTPUT_NAME].reshape(-1)

    testset_X= test_X[TEXT_FEATURES_INPUT_NAME]

    print('All train data shape {:} {:}'.format(alldataset_X.shape, alldataset_y.shape))
    print('Test shape {:}'.format(testset_X.shape))

    # -- GradientBoostingRegressor on all Train data set --
    # With defaut values
    print('GradientBoostingRegressor on all Train data set')
    gbr = GradientBoostingRegressor(random_state=42)

    gbr.fit(alldataset_X, alldataset_y)

    # Predict on Test set
    test_pred_y = gbr.predict(testset_X)

    # Make sure if use_scaling, we need to apply inverse_transform on predicted y (likes)...
    # Scale predicted values to int (-1 because we added +1 before cox-box tansform)
    # TODO using scaling
    if use_scaling:
        scaled_test_pred_y = (pt_cox_y.inverse_transform(test_pred_y.reshape(-1,1)) - 1).astype(int)
    else:
        scaled_test_pred_y = test_pred_y

    # Make sure predicted and scaled back values are not negatives!
    idexes = np.nonzero(scaled_test_pred_y < 0)
    scaled_test_pred_y[idexes] = 0

    # Store test set predictions in DataFrame and save to file
    test_pd = pd.DataFrame()
    test_pd['Id'] = test_profiles_ids_list
    test_pd['Predicted'] = scaled_test_pred_y

    # ***********************   IMPORTANT:  ************************
    # Make sure to push your code to github to keep track of the code
    # used to make the predictions. This is very important to be able
    # to reproduce the predictions and track the models used. Make sure
    # to update the sumbmission csv file in submissions folder, add a
    # comment about the model, the features used the new transformations
    # done to the data and any other relevent information. Don't forget
    # to add the prediction file itself to the subfolder submissions\pred_files.
    # Also, name the prediction file based on the model, date, git version...
    test_tosubmit_folder = os.path.join(log_folder,'v6-GradientBoostingRegressor-Newfeature')
    # Create log folder if does not exist
    if not Path(test_tosubmit_folder).exists():
        os.mkdir(test_tosubmit_folder)
    test_name = 'GBReg-Default-RandState42-CoxBoxY-gitversion-xxxx-2020-11-30'
    prediction_file_save_path = os.path.join(test_tosubmit_folder, test_name+'.csv')
    print('\nSaving prediction to "{:}"'.format(prediction_file_save_path))
    test_pd.to_csv(prediction_file_save_path, sep=',', index=False)


if __name__ == '__main__':
    main()