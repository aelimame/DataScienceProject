# TODO comment and refactor

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
import numpy as np


# RMSLE metric
def rmsle(y_true, y_pred, sample_weight=None):
    # Make sure predicted and scaled back values are not negatives!
    idexes = np.nonzero(y_pred < 0)
    y_pred[idexes] = 0
    return np.sqrt(mean_squared_log_error(y_true, y_pred, sample_weight))


# Method to help load X and y from data loaders
def load_x_y_from_loaders(images_loader,
                          text_data_loader,
                          image_input_name,
                          text_features_input_name,
                          output_name,
                          profiles_ids_list=None,
                          include_images=False):
    likes = None

    # If profiles_ids_list not provided fetch all profiles_ids in text data loader
    if profiles_ids_list is None:
        profiles_ids_list = text_data_loader.get_transformed_features()['Id'].values

    # Retunr X and y
    images = np.array([images_loader.get_image_data_for_profile_id(profile_id) for profile_id in profiles_ids_list])
    features = text_data_loader.get_transformed_features()
    features = features[features['Id'].isin(profiles_ids_list)]
    if 'Num of Profile Likes' in features:
        likes = features['Num of Profile Likes']
        likes = np.array(likes)
        features = features.drop(columns =['Id', 'Num of Profile Likes'])
    else:
        features = features.drop(columns =['Id'])
    features = np.array(features)
    
    if include_images:
        X = {image_input_name: images, # Images
             text_features_input_name: features} # Transformed Text Features
    else:
        X = {text_features_input_name: features} # Transformed Text Features
    
    if likes is not None:
        y = {output_name: likes} # Likes
        return X, y
    
    return X


# Method to plot learning curves of Keras model
def plot_history(history):
    err = history.history['mean_squared_error']
    val_err = history.history['val_mean_squared_error']
    err2 = history.history['mean_squared_logarithmic_error']
    val_err2 = history.history['val_mean_squared_logarithmic_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(err) + 1)

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('loss')
    #plt.ylim(-0.5, 3)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(x, err, 'b', label='Training mean_squared_error')
    plt.plot(x, val_err, 'r', label='Validation mean_squared_error')
    plt.title('mean_squared_error')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(x, err2, 'b', label='Training mean_squared_logarithmic_error')
    plt.plot(x, val_err2, 'r', label='Validation mean_squared_logarithmic_error')
    plt.title('mean_squared_logarithmic_error')
    plt.legend()

    plt.show()
    