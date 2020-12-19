# General imports
import numpy as np
import random
import pandas as pd
import pydot
import os
from pathlib import Path
from operator import itemgetter
# Fix random seeds, Same one to be used everywhere
random_seed = 42
os.environ['PYTHONHASHSEED']=str(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

import xgboost as xgb

import lightgbm as lgb

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Project imports
from utils.images_loader import ImagesLoader
from utils.text_data_loader import TextDataLoader
from utils.data_transformer import HAL9001DataTransformer
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

num_bagging_estimators = 10

num_languages_to_featureize = 16

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
                                              num_languages_to_featureize = num_languages_to_featureize, # Default 9
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
    bagging_gbr = BaggingRegressor(base_estimator=gbr_model, n_estimators=num_bagging_estimators, random_state=random_seed, warm_start=False)

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
                                eta = 0.05, #0.08,
                                max_depth = 6,
                                min_child_weight = 1,
                                subsample = 0.8,
                                random_state=random_seed)
    bagging_xgb = BaggingRegressor(base_estimator=xgb_model, n_estimators=num_bagging_estimators, random_state=random_seed, warm_start=False)

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
                            learning_rate = 0.05, #0.1,
                            max_depth = -1,
                            colsample_bytree = 1.0,
                            subsample = 0.8,
                            random_state=random_seed)
    bagging_gbm = BaggingRegressor(base_estimator=gbm, n_estimators=num_bagging_estimators, random_state=random_seed, warm_start=False)

    # SVM Regressor
    svr_model = SVR(degree=3#,
                    #C = 3.5
                    )

    # Adaboost
#    adabr_model = AdaBoostRegressor()

    # Voting regressor
    regressor = VotingRegressor([('BagGbr', bagging_gbr),
                                ('BagXgb', bagging_xgb),
                                ('BagGbm', bagging_gbm),
                                ('svr_model', svr_model)],
                                n_jobs=-1)

    # -- Pipeline (Has scaling, power_transform and regressor inside) --
    pipe = create_pipeline(use_scaling_for_y=True,
                           data_transformer=data_transformer,
                           feature_selector=feature_selector,
                           regressor=svr_model)#voting_regressor) # TODO use all regressor or any single model you want to evaluate


    # --  Outliers removal k-fold CV evaluation --
    if remove_outliers:
        data_X, data_y = remove_numerical_outliers(data_X, data_y)


    # Print pipelines params
    print('\nPipelines params:')
    print(pipe.get_params())

    # -- Define params to search --
    # GradientBoostingRegressor params
    gbr_n_estimators = [100, 150, 200]
    gbr_max_features = ['auto', 'sqrt', None]
    gbr_max_depth = [3, 4, 5, 6, 8, None]
    gbr_min_samples_split = [2, 3, 4, 5, 6, 8]#, 10]
    gbr_min_samples_leaf = [1, 2, 3, 4, 5, 6, 8]#, 10]

    # XGBRegressor params
    xgbr_n_estimators = [100, 150, 200]
    xgbr_colsample_bytree = [0.6, 0.8, 1.0]
    xgbr__eta = [0.01, 0.05, 0.1, 0.15]
    xgbr_max_depth = [3, 4, 5, 6, 8]#, 10]
    xgbr_min_child_weight = [1, 3, 5, 7]
    xgbr_subsample = [0.6, 0.8, 1.0]

    # Rand Forest
    rfrst_n_estimators = [100, 150, 200]
    rfrst_max_features = ['auto', None]
    rfrst_max_depth = [10, 20, 50, 80, None]
    rfrst_min_samples_split = [2, 3, 4, 5]#, 6, 8, 10]
    rfrst_min_samples_leaf = [1, 2, 3, 4, 5]#, 6, 8]#, 10]
    rfrst_bootstrap = [True, False]

    # Ligth gbm
    gbm_n_estimators = [100, 150 ,200]# ,250] # larger
    gbm_learning_rate = [0.01, 0.05, 0.1, 0.15] # small
    gbm_max_depth =  [3, 4, 5, 6, 8, 10, 15, -1] #
    gbm_num_leaves = [5, 10, 20, 31, 40, 60]#, 80, 100] # large  (may cause over-fitting)
    gbm_colsample_bytree = [0.6, 0.8, 1.0]
    gbm_subsample = [0.6, 0.8, 1.0]
#    bag_gbm_n_estimators = [10, 25, 50] # TODO

    # HAL9001datatransformer
    hal9001dtransf_enable_binary_features = [False, True]
    hal9001dtransf_enable_numerical_features = [False, True]
    hal9001dtransf_enable_profile_col_features = [False, True]
    hal9001dtransf_enable_textual_features = [False, True]
    hal9001dtransf_enable_categorical_features = [False, True]
    hal9001dtransf_enable_datetime_features = [False, True]
    hal9001dtransf_num_languages_to_featureize = [1, 2, 3, 5, 7, 9, 12, 15, 17, 20]
    hal9001dtransf_num_tzones_to_featureize = [0, 1, 2, 3, 5, 7, 9, 12, 15, 17, 20]
    hal9001dtransf_num_utc_to_featureize = [0, 1, 2, 3, 5, 7, 9, 12, 15, 17, 20]


    # Create the random grid
    # SVM regressor
    param_grid = {'regressor__svr__degree': [3],#, 4, 5],
                  'regressor__svr__C': [3.41, 3.42, 3.43, 3.44, 3.45, 3.46, 3.47, 3.48, 3.49, 3.5, 3.51, 3.52, 3.53, 3.54, 3.55],
                  'regressor__svr__gamma': ['scale'],
                  'regressor__svr__kernel': ['rbf']
                 } 
    """
    # All params. Comment/Uncomment those you need to run a search on
    param_grid = {# GBR
                  'regressor__votingregressor__BagGbr__base_estimator__n_estimators': gbr_n_estimators,
                  'regressor__votingregressor__BagGbr__base_estimator__max_features': gbr_max_features,
                  'regressor__votingregressor__BagGbr__base_estimator__max_depth': gbr_max_depth,
                  'regressor__votingregressor__BagGbr__base_estimator__min_samples_split': gbr_min_samples_split,
                  'regressor__votingregressor__BagGbr__base_estimator__min_samples_leaf': gbr_min_samples_leaf,
                    
                  # Xgboost
                  'regressor__votingregressor__BagXgb__base_estimator__n_estimators': xgbr_n_estimators,
                  'regressor__votingregressor__BagXgb__base_estimator__colsample_bytree': xgbr_colsample_bytree,
                  #'regressor__votingregressor__BagXgb__base_estimator__eta': xgbr__eta,
                  'regressor__votingregressor__BagXgb__base_estimator__max_depth': xgbr_max_depth,
                  'regressor__votingregressor__BagXgb__base_estimator__min_child_weight': xgbr_min_child_weight,
                  'regressor__votingregressor__BagXgb__base_estimator__subsample': xgbr_subsample,
                   
                  # RandFrst TODO NOT USED currently
                  #'regressor__votingregressor__BagRanFrst__base_estimator__n_estimators': rfrst_n_estimators,
                  #'regressor__votingregressor__BagRanFrst__base_estimator__max_features': rfrst_max_features,
                  #'regressor__votingregressor__BagRanFrst__base_estimator__max_depth': rfrst_max_depth,
                  #'regressor__votingregressor__BagRanFrst__base_estimator__min_samples_split': rfrst_min_samples_split,
                  #'regressor__votingregressor__BagRanFrst__base_estimator__min_samples_leaf': rfrst_min_samples_leaf,
                  #'regressor__votingregressor__BagRanFrst__base_estimator__bootstrap': rfrst_bootstrap
                  
                  # Lgbm
                  'regressor__votingregressor__BagGbm__base_estimator__n_estimators': gbm_n_estimators,
                  #'regressor__votingregressor__BagGbm__base_estimator__learning_rate': gbm_learning_rate,
                  #'regressor__votingregressor__BagGbm__base_estimator__max_depth': gbm_max_depth,
                  'regressor__votingregressor__BagGbm__base_estimator__num_leaves': gbm_num_leaves,    
                  'regressor__votingregressor__BagGbm__base_estimator__colsample_bytree': gbm_colsample_bytree,    
                  'regressor__votingregressor__BagGbm__base_estimator__subsample': gbm_subsample,
                  #'regressor__votingregressor__BagGbm__n_estimators': bag_gbm_n_estimators
            
                  # hal9001datatransformer
                  #'regressor__hal9001datatransformer__enable_binary_features' : hal9001dtransf_enable_binary_features,
                  #'regressor__hal9001datatransformer__enable_numerical_features' : hal9001dtransf_enable_numerical_features,
                  #'regressor__hal9001datatransformer__enable_profile_col_features' : hal9001dtransf_enable_profile_col_features,
                  #'regressor__hal9001datatransformer__enable_textual_features' : hal9001dtransf_enable_textual_features,
                  #'regressor__hal9001datatransformer__enable_categorical_features' : hal9001dtransf_enable_categorical_features,
                  #'regressor__hal9001datatransformer__enable_datetime_features' : hal9001dtransf_enable_datetime_features,
                  #'regressor__hal9001datatransformer__num_languages_to_featureize': hal9001dtransf_num_languages_to_featureize,
                  #'regressor__hal9001datatransformer__num_tzones_to_featureize': hal9001dtransf_num_tzones_to_featureize,
                  #'regressor__hal9001datatransformer__num_utc_to_featureize': hal9001dtransf_num_utc_to_featureize
                }
    """
    print('\nHyper params to search:')
    print(param_grid)

    # -- Make sure RMSLE does not get negatives values
    # Define custom scorer based on rmsle
    def rmsle_fix_neg(y_true, y_pred):
        # Make sure values anre not negatives!
        y_true = np.clip(y_true, 0, None)
        y_pred = np.clip(y_pred, 0, None)
        return rmsle(y_true, y_pred)

    scorer = make_scorer(rmsle_fix_neg, greater_is_better=False)

    # -- Random Search --
    print('\nRunning Random Search...')
    # Random search of parameters, using k fold cross validation, 
    rand_search_itr = 100
    random_search = RandomizedSearchCV(estimator = pipe,
                                param_distributions = param_grid,
                                scoring = scorer,
                                n_iter = rand_search_itr,
                                cv = 5,
                                verbose = 1,
                                random_state = 42,
                                n_jobs = -1)
    # Fit the random search model
    random_search.fit(data_X, data_y)

    # Print results
    print('\nRandom Search results:')
    print('best_params: {:}'.format(random_search.best_params_))
    print('best_score: {:}'.format(random_search.best_score_))

    # -- Grid Search --
    # TODO not used, very long. Uncomment if you want to use anyways (reduce number of params to search)
    """
    grid_search = GridSearchCV(estimator = pipe,
                            param_grid = param_grid,
                            scoring = scorer,
                            cv = 5,
                            verbose = 3,
                            n_jobs = -1)

    # Fit the random search model
    grid_search.fit(data_X, data_y)

    # Print results
    print('\Grid Search results:')
    print('best_params: {:}'.format(grid_search.best_params_))
    print('best_score: {:}'.format(grid_search.best_score_))
    """



if __name__ == '__main__':
    main()