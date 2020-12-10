import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler, Normalizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from utils.data_transformer import HAL9001DataTransformer

from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

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


def remove_numerical_outliers(train_X, train_y):
    numerical_features = ['Num of Followers',
                        'Num of People Following',
                        'Num of Status Updates',
                        'Num of Direct Messages',
                        'Avg Daily Profile Visit Duration in seconds',
                        'Avg Daily Profile Clicks',
                        'Num of Profile Likes']
    
    #Impute values rather than drop
    numerical_imputer = SimpleImputer(strategy = 'median')
    train_X_numerical = numerical_imputer.fit_transform( train_X[numerical_features] )
    
    #Scale values before removing outliers
    numerical_scaler = RobustScaler()
    train_X_numerical = numerical_scaler.fit_transform(train_X_numerical)

    #outliers_remover = IsolationForest(contamination='auto', random_state = random_seed, n_jobs=-1) # 1.709 (0.051)
    #outliers_remover = LocalOutlierFactor(contamination='auto', n_neighbors = 20, leaf_size = 30, n_jobs=-1) #1.706 (0.045)
    outliers_remover = OneClassSVM(nu=0.01 ) # nu=0.01 -> 1.700 (0.046)
    
    yhat = outliers_remover.fit_predict(train_X_numerical)

    # select all rows that are not outliers
    mask = yhat != -1
    train_X = pd.DataFrame(data=train_X.values[mask, :], columns=train_X.columns)
    train_y = train_y[mask]

    return ( train_X, train_y )

# Create a custom pipline that get a Regressor as parmeter and return a pipline
# HAL9001DataTansformer is now inside!
def create_pipeline(use_scaling_for_y=True,
                    data_transformer=HAL9001DataTransformer(),
                    feature_selector=None,
                    regressor=GradientBoostingRegressor()):

    # X pipeline
    if feature_selector:
        pipe_X = make_pipeline((data_transformer),
                               (feature_selector),
                               (regressor))
    else:
        pipe_X = make_pipeline((data_transformer),
                               (regressor))

    # y Transformer
    if use_scaling_for_y:
        model = TransformedTargetRegressor(regressor=pipe_X,
                                           transformer=CustomTargetTransformer(),
                                           check_inverse=False) # TODO DEBUG For the moment
    else:
        model = pipe_X

    return model



# TODO VERIFY RMSLE metric
def rmsle(y_true, y_pred, sample_weight=None):
    return np.sqrt(mean_squared_log_error(y_true, y_pred, sample_weight))

def rmsle_debug(y_true, y_pred, sample_weight=None):
    y_true_log = np.log(np.clip(y_true, 0, None) + 1.)
    y_pred_log = np.log(np.clip(y_pred, 0, None) + 1.)
    return np.sqrt(np.mean(np.square(y_true_log - y_pred_log)))


# Method to help load X and y from data loaders
def load_x_y_from_loaders(images_loader,
                          text_data_loader,
                          image_input_name,
                          text_features_input_name,
                          output_name,
                          profiles_ids_list=None,
                          include_images=False):

    # initialize likes to None
    likes = None

    # If profiles_ids_list not provided fetch all profiles_ids in text data loader
    if profiles_ids_list is None:
        profiles_ids_list = text_data_loader.get_orig_features()['Id'].values

    orig_features = text_data_loader.get_orig_features()
    orig_features = orig_features[orig_features['Id'].isin(profiles_ids_list)]
    #orig_features = orig_features.drop(columns =['Id'])

    # TODO outliers removal, should be done outside of the DataTransormer since sklearn
    # does not allow dropping X values (Y and X won't match!)
    # Do it here?
#    if 'Num of Profile Likes' in orig_features:
#       orig_features = orig_features[orig_features['Num of Profile Likes'] < 200000]
    
    profiles_ids_list = orig_features['Id'].values # Update profiles_ids_list

    # Extract Y
    if 'Num of Profile Likes' in orig_features:
        likes = orig_features['Num of Profile Likes']
        likes = np.array(likes)

    # Construct dictionaries
    if include_images:
        images = np.array([images_loader.get_image_data_for_profile_id(profile_id) for profile_id in profiles_ids_list])
        X = {image_input_name: images, # Images Features
             text_features_input_name: orig_features} # Orig Text Features
    else:
        X = {text_features_input_name: orig_features} # Orig Text Features
    
    if likes is not None:
        y = {output_name: likes} # Y
        return X, y
    
    return X


# Method to plot learning curves of Keras model
def plot_history(history):
#    err = history.history['mean_squared_logarithmic_error']
#    val_err = history.history['val_mean_squared_logarithmic_error']
#    err2 = history.history['mean_squared_error']
#    val_err2 = history.history['val_mean_squared_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(loss) + 1)

    plt.figure(figsize=(5, 5))
#    plt.subplot(1, 2, 1)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('loss')
    #plt.ylim(-0.5, 3)
    plt.legend()

    """
    plt.subplot(1, 2, 2)
    plt.plot(x, err, 'b', label='Training mean_squared_logarithmic_error')
    plt.plot(x, val_err, 'r', label='Validation mean_squared_logarithmic_error')
    plt.title('mean_squared_logarithmic_error')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(x, err2, 'b', label='Training mean_squared_error')
    plt.plot(x, val_err2, 'r', label='Validation mean_squared_error')
    plt.title('mean_squared_error')
    plt.legend()
    """

    plt.show()
    