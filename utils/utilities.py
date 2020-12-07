import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
import numpy as np

# TODO VERIFY RMSLE metric
def rmsle(y_true, y_pred, sample_weight=None):
    return np.sqrt(mean_squared_log_error(y_true, y_pred, sample_weight))

def rmsle_debug(y_true, y_pred, sample_weight=None):
    y_true_log = np.log(np.clip(y_true, 0, None) + 1.)
    y_pred_log = np.log(np.clip(y_pred, 0, None) + 1.)
    return np.sqrt(np.mean(np.square(y_true_log - y_pred_log)))


# Method to help load X and y from data loaders
# Params: 
#  data_transformer=None, # Will be used (if provided) to fit / transform data
#  transform_only=False, # Transform only if true, fit and transform otherwise
def load_x_y_from_loaders(images_loader,
                          text_data_loader,
                          image_input_name,
                          text_features_input_name,
                          output_name,
                          data_transformer=None,
                          transform_only=False,
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
    if 'Num of Profile Likes' in orig_features:
       orig_features = orig_features[orig_features['Num of Profile Likes'] < 200000]

    # Fit/Transform data
    if data_transformer is not None:
        if transform_only:
            transf_features = data_transformer.transform(orig_features)
        else:
            data_transformer.fit(orig_features)
            transf_features = data_transformer.transform(orig_features)

    # Extract Y
    if 'Num of Profile Likes' in orig_features:
        likes = orig_features['Num of Profile Likes']
        likes = np.array(likes)

    # Construct dictionaries
    if include_images:
        images = np.array([images_loader.get_image_data_for_profile_id(profile_id) for profile_id in profiles_ids_list])
        X = {image_input_name: images, # Images Features
             text_features_input_name: transf_features} # Transformed Text Features
    else:
        X = {text_features_input_name: transf_features} # Transformed Text Features
    
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
    