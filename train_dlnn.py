# General imports
#import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import MeanSquaredError, MeanSquaredLogarithmicError
from tensorflow.keras.optimizers import Adam

import pydot
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Project imports
from utils.images_loader import ImagesLoader
from utils.text_data_loader import TextDataLoader
from utils.utilities import plot_history
from utils.utilities import rmsle
from keras_models import TextOnlyModel, ImageAndTextModel

# Names for acces to data in Dict (to be used with model inputs/output for better referencing)
IMAGE_INPUT_NAME = 'image'
TEXT_FEATURES_INPUT_NAME = 'text_features'
OUTPUT_NAME = 'likes'

# GPU specific
os.environ['CUDA_VISIBLE_DEVICES']='1'


# params
use_scaling = True
include_images = True

# Model Hyper-params TODO move to json file? for better modularity and tracking
learning_rate = 0.001
training_batch_size = 128 #32
n_epochs = 10


# data paths
train_text_path = r'./src_data/train.csv'
train_images_folder = r'./src_data/train_profile_images'
test_text_path = r'./src_data/test.csv'
test_images_folder = r'./src_data/test_profile_images'
log_folder = 'logs'


def main():
    # See devices and if we have a GPU
#    print(tf.config.experimental.list_physical_devices())

    # Create log folder if does not exist
    if not Path(log_folder).exists():
        os.mkdir(log_folder)


    # Images loader
    images_loader = ImagesLoader(src_folder_path = train_images_folder)
    print('Number images: {:}'.format(images_loader.nbr_images))
    print('Images shape: {:}'.format(images_loader.image_shape))

    # Text data loader
    text_data_loader = TextDataLoader(src_csv_file_path = train_text_path)


    # -- Split ids to train/valid --
    all_profiles_ids_list = text_data_loader.get_transformed_features()['Id'].values

    train_profiles_ids, valid_profiles_ids = train_test_split(all_profiles_ids_list, test_size = .2, random_state=42, shuffle=True)

    # -- Preare data for model  Train/Valid--
    # Will use a dict to associate the corresponding data to the right input in the model (Image vs Features)

    # TODO using scaling
    if use_scaling:
        sc_x = StandardScaler()
        pt_cox_y = PowerTransformer(method='box-cox', standardize=True)

    # - Train: use train_profiles_ids
    train_X, train_y = load_x_y_from_loaders(images_loader=images_loader,
                                            text_data_loader=text_data_loader,
                                            image_input_name=IMAGE_INPUT_NAME,
                                            text_features_input_name=TEXT_FEATURES_INPUT_NAME,
                                            output_name=OUTPUT_NAME,
                                            profiles_ids_list=train_profiles_ids,
                                            include_images=include_images)
    if use_scaling:
        # Fit and Transform on Train
        train_X[TEXT_FEATURES_INPUT_NAME] = sc_x.fit_transform(train_X[TEXT_FEATURES_INPUT_NAME])
        train_y[OUTPUT_NAME] = pt_cox_y.fit_transform((train_y[OUTPUT_NAME] + 1).reshape(-1, 1))

    # - Valid: use valid_profiles_ids
    valid_X, valid_y = load_x_y_from_loaders(images_loader=images_loader,
                                             text_data_loader=text_data_loader,
                                             image_input_name=IMAGE_INPUT_NAME,
                                             text_features_input_name=TEXT_FEATURES_INPUT_NAME,
                                             output_name=OUTPUT_NAME,
                                             profiles_ids_list=valid_profiles_ids,
                                             include_images=include_images)


    if use_scaling:
        # Transform only on Valid
        valid_X[TEXT_FEATURES_INPUT_NAME] = sc_x.transform(valid_X[TEXT_FEATURES_INPUT_NAME])
        valid_y[OUTPUT_NAME] = pt_cox_y.transform((valid_y[OUTPUT_NAME] + 1).reshape(-1, 1))



    # -- Prepare DL NN model --
    image_height = images_loader.image_shape[0]
    image_width = images_loader.image_shape[1]
    image_nbr_channels = images_loader.image_shape[2]
    nbr_text_features = train_X[TEXT_FEATURES_INPUT_NAME].shape[1] # TODO text_data_loader.get_nbr_features()

    # Text and Image model
    if include_images:
        model = ImageAndTextModel(image_height = image_height,
                                image_width = image_width,
                                image_nbr_channels = image_nbr_channels,
                                nbr_text_features = nbr_text_features,
                                image_input_name=IMAGE_INPUT_NAME,
                                text_features_input_name=TEXT_FEATURES_INPUT_NAME,
                                output_name=OUTPUT_NAME)

    # Text only model
    else :
        model = TextOnlyModel(nbr_text_features = nbr_text_features,
                              text_features_input_name=TEXT_FEATURES_INPUT_NAME,
                              output_name=OUTPUT_NAME)

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

    # -- Plot train/valid learning curves error/loss
    plot_history(hist)


    # -- TODO Predict on test set here?---
    # Make sure if use_scaling, we need to apply inverse_transform on predicted y (likes)...
    # See Notebook-Experiences.ipynb 
    #...

if __name__ == '__main__':
    main()