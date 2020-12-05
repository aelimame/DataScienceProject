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

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler, Normalizer
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, VotingRegressor
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
random_seed = 42

# Change this generate a prediction on test
predict_on_test = True

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
        scaled_target = (self.power_transformer.inverse_transform(target.reshape(-1,1)) - 1).astype(int)

        # Make sure scaled back targets are not negatives!
        #indexes = np.nonzero(scaled_target <= 0)
        #scaled_target[indexes] = 0
        scaled_target = np.clip(scaled_target, a_min=0, a_max=None)

        return scaled_target.reshape(-1)



# Create a custom pipline that get a Regressor as parmeter and return a pipline
# TODO include Dataloaders and HAL9001DataTansformer inside!
def create_pipeline(use_scaling=True,
                    regressor=GradientBoostingRegressor()):
    # X pipeline

    # using scaling
    if use_scaling:
        # TODO: Try other scalers/Transformers...
#        x_scaler = StandardScaler()
#        x_scaler = QuantileTransformer(n_quantiles=9, output_distribution='normal')
        x_scaler = RobustScaler()
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

    # Pipeline with best searched hyper parms
    print('Buiding and evaluating pipeline with models')
    # BGR like V7
    gbrv7 = GradientBoostingRegressor(n_estimators=100,
                                max_features='sqrt',
                                max_depth=5,
                                min_samples_split=2,
                                min_samples_leaf=7,
                                random_state=42)

    # -- Pipeline (Has scaling, power_transform and regressor inside) --
    pipe = create_pipeline(use_scaling=True,
                        regressor=gbrv7)#rand_forest)#voting_regressor)

    # -- KFold CV using scorer based on rmsle --
    scorer = make_scorer(rmsle, greater_is_better=False)
    kfoldcv = KFold(n_splits=10, random_state=random_seed, shuffle=True)
    rep_kfoldcv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=random_seed)

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
    if predict_on_test:
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
        print('Fitting on all data')
        pipe.fit(data_X, data_y)

        # -- Predict on Test set --
        print('Predicting on test')
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
        test_tosubmit_folder = os.path.join(log_folder,'v16-DataTransformLikeV7-GBRV7-HyperParams')
        # Create log folder if does not exist
        if not Path(test_tosubmit_folder).exists():
            os.mkdir(test_tosubmit_folder)
        test_name = 'v16-GBRV7-RobustScaler-HyperParams-DataTransformLikeV7-RandState42-CoxBoxY-gitversion-xxxx-2020-12-05'
        prediction_file_save_path = os.path.join(test_tosubmit_folder, test_name+'.csv')
        print('\nSaving prediction to "{:}"'.format(prediction_file_save_path))
        test_pd.to_csv(prediction_file_save_path, sep=',', index=False)


if __name__ == '__main__':
    main()