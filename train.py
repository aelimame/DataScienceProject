# General imports
#import cv2
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import MeanSquaredError, MeanSquaredLogarithmicError
from tensorflow.keras.optimizers import Adam
import pydot
import tensorflow as tf
import os
from pathlib import Path

# Project imports
from utils.images_loader import ImagesLoader
from utils.text_data_loader import TextDataLoader
from model import TextOnlyModel, ImageAndTextModel
from model import IMAGE_INPUT_NAME, TEXT_FEATURES_INPUT_NAME, OUTPUT_NAME


# GPU specific
os.environ['CUDA_VISIBLE_DEVICES']='1'


# Hyper-params TODO move to json file? for better modularity and tracking
learning_rate = 0.001
training_batch_size = 128 #32
n_epochs = 200
include_image = False
# DEBUG/TESTS
use_scaling = True

# data paths
train_text_path = r'.\src_data\train.csv'
train_images_folder = r'.\src_data\train_profile_images'
log_folder = 'logs'


def main():
    # See devices and if we have a GPU
    print(tf.config.experimental.list_physical_devices())

    # Create log folder if does not exist
    if not Path(log_folder).exists():
        os.mkdir(log_folder)
    

    # Images loader
    images_loader = ImagesLoader(src_folder_path = train_images_folder)
    print('Number images: {:}'.format(images_loader.nbr_images))
    print('Images shape: {:}'.format(images_loader.image_shape))

    # Text data loader
    text_data_loader = TextDataLoader(src_csv_file_path = train_text_path)

    # TEST Get and profile by its ID
#    profile_id = '00NJOGS399G79OP3'
#    image_data = images_loader.get_image_data_for_profile_id(profile_id)
#    text_data = text_data_loader.get_orig_data_for_profile_id(profile_id)
    # Print data for this profile
#    print(text_data)
    # Show profile image
#    plt.imshow(image_data)
#    plt.show()

    # -- Split ids to train/valid --
    #all_profiles_ids_list = text_data_loader.get_orig_data()['Id'].values
    all_profiles_ids_list = text_data_loader.get_transformed_features()['Id'].values
    train_profiles_ids, valid_profiles_ids = train_test_split(all_profiles_ids_list, test_size = .2, random_state=42, shuffle=True)

    # -- Preare data for model --
    # Will use a dict to associate the corresponding data to the right input in the model (Image vs Features)
    
    # TODO DEBUG using scaler for tests
    if use_scaling:
        sc_x = StandardScaler()
        sc_y = StandardScaler()

    # - Train: use train_profiles_ids
    train_images = np.array([images_loader.get_image_data_for_profile_id(profile_id) for profile_id in train_profiles_ids])

# TODO gives errors:    train_features = np.array([text_data_loader.get_transformed_features_for_profile_id(profile_id) for profile_id in train_profiles_ids])
    train_features = []
    for idx, profile_id in enumerate(train_profiles_ids, start=0):
        train_features += [text_data_loader.get_transformed_features_for_profile_id(profile_id).drop(columns =['Id', 'Num of Profile Likes']).values[0]]
        #train_features += [text_data_loader.get_transformed_features_for_profile_id(profile_id).drop(columns =['Id']).values[0]]
    train_features = np.array(train_features)
    if use_scaling:
        train_features = sc_x.fit_transform(train_features)

# TODO gives errors:    train_likes = np.array([text_data_loader.get_transformed_features_for_profile_id(profile_id)['Num of Profile Likes'] for profile_id in train_profiles_ids])
    train_likes = []
    for idx, profile_id in enumerate(train_profiles_ids, start=0):
        train_likes += [text_data_loader.get_transformed_features_for_profile_id(profile_id)['Num of Profile Likes'].values[0]]
    train_likes = np.array(train_likes)
    if use_scaling:
        train_likes = sc_y.fit_transform(train_likes.reshape(-1, 1))
   
    train_X = {IMAGE_INPUT_NAME: train_images, # Images
               TEXT_FEATURES_INPUT_NAME: train_features} # Encoded Text Features
    train_y = {OUTPUT_NAME: train_likes} # Likes

    # - Valid: use valid_profiles_ids
    valid_images = np.array([images_loader.get_image_data_for_profile_id(profile_id) for profile_id in valid_profiles_ids])
# TODO gives errors:   valid_features = np.array([text_data_loader.get_transformed_features_for_profile_id(profile_id) for profile_id in valid_profiles_ids])
    valid_features = []
    for idx, profile_id in enumerate(valid_profiles_ids, start=0):
        valid_features += [text_data_loader.get_transformed_features_for_profile_id(profile_id).drop(columns =['Id', 'Num of Profile Likes']).values[0]]
        #valid_features += [text_data_loader.get_transformed_features_for_profile_id(profile_id).drop(columns =['Id']).values[0]]
    valid_features = np.array(valid_features)
    if use_scaling:
        valid_features = sc_x.transform(valid_features)

# TODO gives errors :  valid_likes = np.array([text_data_loader.get_transformed_features_for_profile_id(profile_id)['Num of Profile Likes'] for profile_id in valid_profiles_ids])
    valid_likes = []
    for idx, profile_id in enumerate(valid_profiles_ids, start=0):
        valid_likes += [text_data_loader.get_transformed_features_for_profile_id(profile_id)['Num of Profile Likes'].values[0]]
    valid_likes = np.array(valid_likes)
    if use_scaling:
        valid_likes = sc_y.transform(valid_likes.reshape(-1, 1))

    valid_X = {IMAGE_INPUT_NAME: valid_images, # Images
               TEXT_FEATURES_INPUT_NAME: valid_features} # Encoded Text Features
    valid_y = {OUTPUT_NAME: valid_likes} # Likes


    # -- Prepare model --
    image_height = images_loader.image_shape[0]
    image_width = images_loader.image_shape[1]
    image_nbr_channels = images_loader.image_shape[2]
    nbr_text_features = train_features.shape[1] # TODO text_data_loader.get_nbr_features()

    # Text and Image model
    if include_image:
        model = ImageAndTextModel(image_height = image_height,
                                image_width = image_width,
                                image_nbr_channels = image_nbr_channels,
                                nbr_text_features = nbr_text_features)

    # Text only model
    else :
        X= train_X[TEXT_FEATURES_INPUT_NAME]
        y= train_y[OUTPUT_NAME]
        model = TextOnlyModel(nbr_text_features = nbr_text_features)

    # Model summary
    model.summary() # TODO model summary to file
    plot_model(model, os.path.join(log_folder, 'model.png'), show_shapes=True)

    # -- Compile model --
    model.compile(loss=MeanSquaredError(), #MeanSquaredLogarithmicError(),
                  optimizer=Adam(lr=learning_rate),
                  metrics=['mean_squared_error', 'mean_squared_logarithmic_error'])

    # -- Fit model --
    hist = model.fit(x=train_X,
                     y=train_y,
                     batch_size = training_batch_size,
                     validation_data=(valid_X, valid_y),
                     epochs=n_epochs,
                     verbose=1)

    # -- DEBUG linear regression on text features --
    from sklearn.linear_model import LinearRegression
    X= train_X[TEXT_FEATURES_INPUT_NAME]
    y= train_y[OUTPUT_NAME]
    reg = LinearRegression().fit(X, y)
    print(reg.score(X, y))
    # -- DEBUG --

    # -- Plot train/valid learning curves error/loss
    # TODO Move method outside (in utils). Can also use Keras callback for checkpoints and tensorboard logging...
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
        plt.title('Training and validation loss')
        #plt.ylim(-0.5, 3)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(x, err, 'b', label='Training mean_squared_error')
        plt.plot(x, val_err, 'r', label='Validation mean_squared_error')
        plt.title('Training and validation mean_squared_error')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(x, err2, 'b', label='Training mean_squared_logarithmic_error')
        plt.plot(x, val_err2, 'r', label='Validation mean_squared_logarithmic_error')
        plt.title('Training and validation mean_squared_logarithmic_error')
        plt.legend()

        plt.show()

    plot_history(hist)

    # -- TODO Predict on test set here?---
    # Make sure if use_scaling, we need to apply inverse_transform on predicted y (likes)...
    #...

if __name__ == '__main__':
    main()