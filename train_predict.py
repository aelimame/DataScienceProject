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

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, VotingRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.base import clone

# Project imports
from utils.images_loader import ImagesLoader
from utils.text_data_loader import TextDataLoader
from utils.data_transformer import HAL9001DataTransformer, NumericalTransformer
from utils.utilities import rmsle
from utils.utilities import load_x_y_from_loaders
from utils.utilities import create_pipeline
from utils.utilities import remove_numerical_outliers, remove_numerical_outliers_iqr


# Names for acces to data in Dict (to be used with model inputs/output for better referencing)
IMAGE_INPUT_NAME = 'image'
TEXT_FEATURES_INPUT_NAME = 'text_features'
OUTPUT_NAME = 'likes'

# params
use_scaling_for_y = True
include_images = False
remove_outliers = True
random_seed = 42

# Change this generate a prediction on test
predict_on_test = True

# data paths
train_text_path = r'./src_data/train.csv'
train_images_folder = r'./src_data/train_profile_images'
test_text_path = r'./src_data/test.csv'
test_images_folder = r'./src_data/test_profile_images'
log_folder = 'logs'




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
#    data_transformer = HAL9001DataTransformer()
    #'regressor__hal9001datatransformer__enable_binary_features': True
    #'regressor__hal9001datatransformer__enable_numerical_features': True,
    #'regressor__hal9001datatransformer__enable_profile_col_features': True,
    #'regressor__hal9001datatransformer__enable_textual_features': True,
    #'regressor__hal9001datatransformer__enable_categorical_features': True,
    #'regressor__hal9001datatransformer__enable_datetime_features': True,
    #'regressor__hal9001datatransformer__num_languages_to_featureize': 11,
    #'regressor__hal9001datatransformer__num_tzones_to_featureize': 5,
    #'regressor__hal9001datatransformer__num_utc_to_featureize': 17,
    data_transformer = HAL9001DataTransformer(enable_binary_features = True,
                                              enable_numerical_features = True,
                                              enable_profile_col_features = True,
                                              enable_textual_features = True,
                                              enable_categorical_features = True,
                                              enable_datetime_features = True,
                                              num_languages_to_featureize = 9, # Default 9
                                              num_tzones_to_featureize = 10, # Default 10
                                              num_utc_to_featureize = 15) # Default 15

    # -- Feature Selector --
    # All current tests show that best performance is achieved using all 66 features
    feature_selector = None # SelectKBest(mutual_info_regression, k=66)

    # -- Prepare Train data X, y --
    print('\n\nEvaluating on Train data using k-fold-CV')
    data_X, data_y = load_x_y_from_loaders(images_loader=images_loader,
                                        text_data_loader=text_data_loader,
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
    # GBR
    # RandSearchCV params
    #'regressor__votingregressor__BagGbr__base_estimator__n_estimators': 200,
    #'regressor__votingregressor__BagGbr__base_estimator__max_features': 'sqrt',
    #'regressor__votingregressor__BagGbr__base_estimator__max_depth': 8,
    #'regressor__votingregressor__BagGbr__base_estimator__min_samples_split': 3,
    #'regressor__votingregressor__BagGbr__base_estimator__min_samples_leaf': 6,
    gbr_model = GradientBoostingRegressor(n_estimators=200,
                                    max_features='sqrt',
                                    max_depth=8,
                                    min_samples_split=3,
                                    min_samples_leaf=6,
                                    random_state=random_seed)
    bagging_gbr = BaggingRegressor(base_estimator=gbr_model, n_estimators=25, random_state=random_seed)

    # xgboost
    #'regressor__votingregressor__BagXgb__base_estimator__n_estimators': 200,
    #'regressor__votingregressor__BagXgb__base_estimator__colsample_bytree': 0.6,
    #'regressor__votingregressor__BagXgb__base_estimator__eta': 0.08,
    #'regressor__votingregressor__BagXgb__base_estimator__max_depth': 6,
    #'regressor__votingregressor__BagXgb__base_estimator__min_child_weight': 1,
    #'regressor__votingregressor__BagXgb__base_estimator__subsample': 0.8,
    xgb_model = xgb.XGBRegressor(objective="reg:squaredlogerror",
                                eval_metric='rmsle',
                                n_estimators=200,
                                colsample_bytree = 0.6,
                                eta = 0.08,
                                max_depth = 6,
                                min_child_weight = 1,
                                subsample = 0.8,
                                random_state=random_seed)
    bagging_xgb = BaggingRegressor(base_estimator=xgb_model, n_estimators=25, random_state=random_seed)

    # Lightgbm
    # RandSearchCV params
    #'regressor__votingregressor__BagGbm__base_estimator__n_estimators': 100,
    #'regressor__votingregressor__BagGbm__base_estimator__num_leaves': 40,
    #'regressor__votingregressor__BagGbm__base_estimator__learning_rate': 0.1,
    #'regressor__votingregressor__BagGbm__base_estimator__max_depth': -1,
    #'regressor__votingregressor__BagGbm__base_estimator__colsample_bytree': 1.0
    #'regressor__votingregressor__BagGbm__base_estimator__subsample': 0.8,
    gbm = lgb.LGBMRegressor(n_estimators=100,
                            num_leaves=40,
                            learning_rate=0.1,
                            max_depth = -1,
                            colsample_bytree = 1.0,
                            subsample = 0.8,
                            random_state=random_seed)
    bagging_gbm = BaggingRegressor(base_estimator=gbm, n_estimators=25, random_state=random_seed)

    #Voting regressor
    voting_regressor = VotingRegressor([('BagGbr', bagging_gbr),
                                        ('BagXgb', bagging_xgb),
                                        ('BagGbm', bagging_gbm)],
                                        n_jobs=-1)

    # -- Pipeline (Has scaling, power_transform and regressor inside) --
    pipe = create_pipeline(use_scaling_for_y=use_scaling_for_y,
                           data_transformer=data_transformer,
                           feature_selector=feature_selector,
                           regressor=voting_regressor)


    # --  Outliers removal k-fold CV evaluation --
    if remove_outliers:
        cv_outer = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        result_scores = []
        for train_ix, valid_ix in cv_outer.split(data_X):
            # split data
            train_X, valid_X = data_X.iloc[train_ix, :], data_X.iloc[valid_ix, :]
            train_y, valid_y = data_y[train_ix], data_y[valid_ix]

            # REMOVE OUTLIERS FROM train_X/train_y only.
            train_X, train_y = remove_numerical_outliers(train_X, train_y)

            # Evaluate/train pipe/models defined above using the "cleaned" train and valid sets
            pipe = create_pipeline(use_scaling_for_y=use_scaling_for_y,
                           data_transformer=data_transformer,
                           feature_selector=feature_selector,
                           regressor=voting_regressor)
            pipe_outliers = clone(pipe)
            pipe_outliers.fit(train_X, train_y)
            valid_pred_y = pipe_outliers.predict(valid_X)
            curr_score = rmsle(valid_y, valid_pred_y)

            # Save scores
            result_scores.append(curr_score)

        # -- Print score --
        print('\n\nK-Fold CV RMSE OUTLIERS-REMOVED: %.10f (%.5f)\n\n' % (np.mean(result_scores), np.std(result_scores)))   


    # -- KFold CV using scorer based on rmsle --
    scorer = make_scorer(rmsle, greater_is_better=False)
    kfoldcv = KFold(n_splits=5, random_state=random_seed, shuffle=True)
#    rep_kfoldcv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=random_seed)

    pipe = create_pipeline(use_scaling_for_y=use_scaling_for_y,
                           data_transformer=data_transformer,
                           feature_selector=feature_selector,
                           regressor=voting_regressor)
    pipe_cv = clone(pipe)
    scores = cross_val_score(pipe_cv,
                             data_X,
                             data_y,
                             scoring=scorer,
                             cv=kfoldcv,
                             n_jobs=-1)

#    pipe_rep_cv = clone(pipe)
#    rep_scores = cross_val_score(pipe_rep_cv,
#                             data_X,
#                             data_y,
#                             scoring=scorer,
#                             cv=rep_kfoldcv,
#                             n_jobs=-1)

    # -- Print score --
    print('K-Fold CV RMSLE: %.10f (%.5f)' % (np.mean(scores), np.std(scores)))
#    print('Repeated K-Fold CV RMSLE: %.10f (%.5f)' % (np.mean(rep_scores), np.std(rep_scores)))





    # -- Predict on Test set ---
    if predict_on_test:
        print('\n\nPredicting on Test data')

        # -- Preare Test data  X, y --
        test_profiles_ids_list = test_text_data_loader.get_orig_features()['Id'].values
        test_X = load_x_y_from_loaders(images_loader=test_images_loader,
                                    text_data_loader=test_text_data_loader,
                                    image_input_name=IMAGE_INPUT_NAME,
                                    text_features_input_name=TEXT_FEATURES_INPUT_NAME,
                                    output_name=OUTPUT_NAME,
                                    profiles_ids_list=test_profiles_ids_list,
                                    include_images=include_images)

        # No Need for dict for the moment
        test_X = test_X[TEXT_FEATURES_INPUT_NAME]

        print('Test shape {:}'.format(test_X.shape))

        # -- Remove Outliers --
        if remove_outliers:
            print('Removing Outliers')
            data_X, data_y = remove_numerical_outliers(data_X, data_y)
            print('Data (Outliers removed) shape {:} {:}'.format(data_X.shape, data_y.shape))

        # -- Fit pipe (Transofrmation and model) on all Train data set --
        print('Fitting on all data')
        pipe = create_pipeline(use_scaling_for_y=use_scaling_for_y,
                           data_transformer=data_transformer,
                           feature_selector=feature_selector,
                           regressor=voting_regressor)
        pipe_final = clone(pipe)
        pipe_final.fit(data_X, data_y)

        # -- Predict on Test set --
        print('Predicting on test')
        test_pred_y = pipe_final.predict(test_X)

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
        test_tosubmit_folder = os.path.join(log_folder,'V27-VotBag25GbrXgbLgbm-NumOutlierOneClassSVM')
        # Create log folder if does not exist
        if not Path(test_tosubmit_folder).exists():
            os.mkdir(test_tosubmit_folder)
        test_name = 'V27-VotBag25GbrXgbLgbm-NumOutlierOneClassSVM-RanSta42-CoxBoxY-gitvers-xxxx-2020-12-11'
        prediction_file_save_path = os.path.join(test_tosubmit_folder, test_name+'.csv')
        print('\nSaving prediction to "{:}"'.format(prediction_file_save_path))
        test_pd.to_csv(prediction_file_save_path, sep=',', index=False)


if __name__ == '__main__':
    main()