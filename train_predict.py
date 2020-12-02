# General imports
#import cv2

#### FOR MAC OSX USERS
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pydot
import os
from pathlib import Path
from operator import itemgetter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline



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




# A custom Y transformer
class CustomTargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.power_transformer = PowerTransformer(method='box-cox', standardize=False)
        
    def fit(self, target):
        self.power_transformer.fit((target + 1).reshape(-1, 1))
        return self
    
    def transform(self, target):
        transformed_target = self.power_transformer.transform((target.astype(np.float64) + 1).reshape(-1, 1))
        return transformed_target.reshape(-1).astype(np.float64)

    def inverse_transform(self, target):
        # Scale target back
        scaled_target = (self.power_transformer.inverse_transform(target.reshape(-1,1)) - 1)
        
        # Make sure scaled back targets are not negatives!
        idexes = np.nonzero(scaled_target < 0)
        scaled_target[idexes] = 0

        return scaled_target.reshape(-1).astype(int)
    
    

# Create a custom pipline that get a Regressor as parmeter and return a pipline
# TODO include Dataloaders and HAL9001DataTansformer inside!
def create_pipeline(use_scaling=True,
                    regressor=GradientBoostingRegressor()):
    # X pipeline
   
    # using scaling
    if use_scaling:
        x_scaler=StandardScaler()
        pipe_X = make_pipeline((x_scaler),
                               (regressor))
    else:
        # X pipline with regressor only
        pipe_X = make_pipeline(regressor)
    
    # y Transformer
    if use_scaling:
        model = TransformedTargetRegressor(regressor=pipe_X,
                                           transformer=CustomTargetTransformer(),
                                           check_inverse=False) # TODO DEBUG For the moment
    else:
        model = pipe_X

    return model


# Main program
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
    """
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


    """

    # -- Prepare Train data X, y --
    print('\n\nEvaluating on Train data using k-fold-CV')
    data_X, data_y = load_x_y_from_loaders(images_loader=images_loader,
                                        text_data_loader=text_data_loader,
                                        data_transformer=data_transformer,
                                        transform_only=False, # TODO Fit and Transform
                                        image_input_name=IMAGE_INPUT_NAME,
                                        text_features_input_name=TEXT_FEATURES_INPUT_NAME,
                                        output_name=OUTPUT_NAME,
                                        profiles_ids_list=None, # Load all profiles in data
                                        include_images=include_images)
    # No Need for dict for the moment
    data_X = data_X[TEXT_FEATURES_INPUT_NAME]
    data_y = data_y[OUTPUT_NAME]

    print('Data shape {:} {:}'.format(data_X.shape, data_y.shape))

    # -- Prepare pipeline --
        
    # GBR With best searched hyper parms
    print('GradientBoostingRegressor')
    gbr = GradientBoostingRegressor(n_estimators=200,
                                    max_features=None,
                                    max_depth=3,
                                    min_samples_split=2,
                                    min_samples_leaf=10,
                                    random_state=42)

    
    # Pipeline (Has scaling, power_transform and regressor inside)
    pipe = create_pipeline(use_scaling=True,
                           regressor=gbr)

    # -- KFold CV using scorer based on rmsle --
    scorer = make_scorer(rmsle, greater_is_better=False)
    kfoldcv = KFold(n_splits=10, random_state=42, shuffle=True)
    rep_kfoldcv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)


    scores = cross_val_score(pipe,
                             data_X,
                             data_y,
                             scoring=scorer,
                             cv=kfoldcv,
                             n_jobs=-1)

    rep_scores = cross_val_score(pipe,
                             data_X,
                             data_y,
                             scoring=scorer,
                             cv=rep_kfoldcv,
                             n_jobs=-1)
                             
    # -- Print score --
    print('K-Fold CV RMSLE: %.10f (%.5f)' % (np.mean(scores), np.std(scores)))
    print('Repeated K-Fold CV RMSLE: %.10f (%.5f)' % (np.mean(rep_scores), np.std(rep_scores)))
    





    # -- Predict on Test set ---
    print('\n\nPredicting on Test data')

    # -- Preare Test data  X, y --
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
    # No Need for dict for the moment
    test_X = test_X[TEXT_FEATURES_INPUT_NAME]

    print('Test shape {:}'.format(test_X.shape))

    # -- Fit pipe (Transofrmation and model) on all Train data set --
    pipe.fit(data_X, data_y)

    # -- Predict on Test set --
    test_pred_y = pipe.predict(test_X)

    # Store test set predictions in DataFrame and save to file
    test_pd = pd.DataFrame()
    test_pd['Id'] = test_profiles_ids_list
    test_pd['Predicted'] = test_pred_y

    # ***********************   IMPORTANT:  ************************
    # Make sure to push your code to github to keep track of the code
    # used to make the predictions. This is very important to be able
    # to reproduce the predictions and track the models used. Make sure
    # to update the sumbmission csv file in submissions folder, add a
    # comment about the model, the features used the new transformations
    # done to the data and any other relevent information. Don't forget
    # to add the prediction file itself to the subfolder submissions\pred_files.
    # Also, name the prediction file based on the model, date, git version...
    test_tosubmit_folder = os.path.join(log_folder,'v8-GradientBoostingRegressor-MoreNewfeature-Hyperparams')
    # Create log folder if does not exist
    if not Path(test_tosubmit_folder).exists():
        os.mkdir(test_tosubmit_folder)
    test_name = 'GBReg-HyperParams-MoreNewFeature-RandState42-CoxBoxY-gitversion-xxxx-2020-12-01-PIPE'
    prediction_file_save_path = os.path.join(test_tosubmit_folder, test_name+'.csv')
    print('\nSaving prediction to "{:}"'.format(prediction_file_save_path))
    test_pd.to_csv(prediction_file_save_path, sep=',', index=False)


if __name__ == '__main__':
    main()