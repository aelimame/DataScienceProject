# General imports
#import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import MeanSquaredError, MeanSquaredLogarithmicError
from tensorflow.keras.optimizers import Adam
# Disable eager execution
tf.compat.v1.disable_eager_execution()

import pydot
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler

# Project imports
from utils.images_loader import ImagesLoader
from utils.text_data_loader import TextDataLoader
from utils.data_transformer_new import HAL9001DataTransformer
from utils.utilities import plot_history
from utils.utilities import rmsle, rmsle_debug
from utils.utilities import load_x_y_from_loaders
from keras_models import TextOnlyModel, ImageAndTextModel

# Names for acces to data in Dict (to be used with model inputs/output for better referencing)
IMAGE_INPUT_NAME = 'image'
TEXT_FEATURES_INPUT_NAME = 'text_features'
OUTPUT_NAME = 'likes'

# params
#use_scaling_for_X = True
use_scaling_for_y = False # TODO Don't use with loss RMSLE!?
include_images = True
random_seed = 42

# Change this generate a prediction on test
predict_on_test = True

# data paths
train_text_path = r'./src_data/train.csv'
train_images_folder = r'./src_data/train_profile_images'
test_text_path = r'./src_data/test.csv'
test_images_folder = r'./src_data/test_profile_images'
log_folder = 'logs'

# Model Hyper-params TODO move to json file? for better modularity and tracking
learning_rate = 0.001
training_batch_size = 32 #32
n_epochs = 15

# GPU specific
#os.environ['CUDA_VISIBLE_DEVICES']='1'


def main():
    # See devices and if we have a GPU
#    print(tf.config.experimental.list_physical_devices())

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

    #-- Preare data for model  Train/Valid--
    # Use a dict to associate the corresponding data to the right input in the model (Image vs Features)
    # -- Data Transformer --
    data_transformer = HAL9001DataTransformer()

    # -- Split ids to train/valid --
    print('\n\nEvaluating on 1 simple Train/Valid split')
    data_X, data_y = load_x_y_from_loaders(images_loader=images_loader,
                                        text_data_loader=text_data_loader,
                                        image_input_name=IMAGE_INPUT_NAME,
                                        text_features_input_name=TEXT_FEATURES_INPUT_NAME,
                                        output_name=OUTPUT_NAME,
                                        profiles_ids_list=None, # Load all profiles in data
                                        include_images=include_images)
    if include_images:
        print('Data shape X: {:} {:}, y: {:}'.format(data_X[IMAGE_INPUT_NAME].shape, data_X[TEXT_FEATURES_INPUT_NAME].shape, data_y[OUTPUT_NAME].shape))
    else:
        print('Data shape X: {:}, y: {:}'.format(data_X[TEXT_FEATURES_INPUT_NAME].shape, data_y[OUTPUT_NAME].shape))


    if include_images:
        train_X1, valid_X1, \
        train_X2, valid_X2, \
        train_y, valid_y = train_test_split(data_X[IMAGE_INPUT_NAME],
                                            data_X[TEXT_FEATURES_INPUT_NAME],
                                            data_y[OUTPUT_NAME],
                                            test_size = .2,
                                            random_state=random_seed,
                                            shuffle=True)

        # Put data back as dict (Need this for Keras NN)
        train_X=dict({IMAGE_INPUT_NAME: train_X1,
                    TEXT_FEATURES_INPUT_NAME:train_X2})
        train_y=dict({OUTPUT_NAME: train_y})
        valid_X=dict({IMAGE_INPUT_NAME: valid_X1,
                    TEXT_FEATURES_INPUT_NAME:valid_X2})
        valid_y=dict({OUTPUT_NAME: valid_y})
    else:
        train_X, valid_X, \
        train_y, valid_y = train_test_split(data_X[TEXT_FEATURES_INPUT_NAME],
                                                   data_y[OUTPUT_NAME],
                                                   test_size = .2,
                                                   random_state=random_seed,
                                                   shuffle=True)
        # Put data back as dict (Need this for Keras NN)
        train_X=dict({TEXT_FEATURES_INPUT_NAME:train_X})
        train_y=dict({OUTPUT_NAME: train_y})
        valid_X=dict({TEXT_FEATURES_INPUT_NAME:valid_X})
        valid_y=dict({OUTPUT_NAME: valid_y})


    # Fit/Transform on Train and transform Valid
    data_transformer.fit(train_X[TEXT_FEATURES_INPUT_NAME])
    train_X[TEXT_FEATURES_INPUT_NAME] = data_transformer.transform(train_X[TEXT_FEATURES_INPUT_NAME])

    valid_X[TEXT_FEATURES_INPUT_NAME] = data_transformer.transform(valid_X[TEXT_FEATURES_INPUT_NAME])

    # Using scaling for target (Dont use with loss RMSLE!)
    if use_scaling_for_y:
        pt_cox_y = PowerTransformer(method='box-cox', standardize=False)

        # Fit and Transform on Train
        train_y[OUTPUT_NAME] = pt_cox_y.fit_transform((train_y[OUTPUT_NAME] + 1).reshape(-1, 1))

        # Transform only on Valid
        valid_y[OUTPUT_NAME] = pt_cox_y.transform((valid_y[OUTPUT_NAME] + 1).reshape(-1, 1))


    # -- Prepare DL NN model --
    image_height = images_loader.image_shape[0]
    image_width = images_loader.image_shape[1]
    image_nbr_channels = images_loader.image_shape[2]
    nbr_text_features = train_X[TEXT_FEATURES_INPUT_NAME].shape[1]

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

    # -- Define RMSLE for keras
    def keras_rmsle(y_true, y_pred):
        y_true_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
        y_pred_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
        return K.sqrt(K.mean(K.square(y_true_log - y_pred_log)))
    """
    # Not Good?!
    def keras_rmsle(y_true, y_pred):
        y_true_clip = K.clip(y_true, K.epsilon(), None)
        y_pred_clip = K.clip(y_pred, K.epsilon(), None)
        return K.sqrt(mean_squared_logarithmic_error(y_true_clip, y_pred_clip))
        #return K.sqrt(mean_squared_logarithmic_error(y_true, y_pred))
    """

    # -- Compile model --
    #model.compile(loss=MeanSquaredError(), #MeanSquaredLogarithmicError()
    #model.compile(loss=MeanSquaredLogarithmicError(),
    model.compile(loss=keras_rmsle,
                  optimizer=Adam(lr=learning_rate))
                  #metrics=['mean_squared_logarithmic_error']#, 'mean_squared_error']


    # -- Fit model --
    hist = model.fit(x=train_X,
                     y=train_y,
                     batch_size = training_batch_size,
                     validation_data=(valid_X, valid_y),
                     epochs=n_epochs,
                     verbose=1)

    # -- Plot train/valid learning curves error/loss
    plot_history(hist)

    # Evaluation No transform on Y
    valid_pred_y = model.predict(valid_X)
    valid_pred_y = np.clip(valid_pred_y, 0, a_max=None).astype(int)

    # Compute rmsle
    rmsle_val = rmsle(valid_pred_y, valid_y[OUTPUT_NAME].astype(int))
    rmsle_val_debug = rmsle_debug(valid_pred_y, valid_y[OUTPUT_NAME].astype(int))
    print('\n RMSLE on 1 validation split : %.10f ' % rmsle_val)
    print('\n RMSLE on 1 validation split : %.10f ' % rmsle_val_debug)
    print('Eval Keras: %.10f ' % (model.evaluate(valid_X, valid_y)))

    # -- TODO Save keras model? --
    # ..





    # -- TODO Predict on Test set here? Better to save model ckpt and generated from them saved model ---
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

        # test_X is a dict with Image and TextFeatures numpy arrays
        if include_images:
            print('Test shape X: {:} {:}'.format(test_X[IMAGE_INPUT_NAME].shape, test_X[TEXT_FEATURES_INPUT_NAME].shape))
        else:
            print('Test shape X: {:}'.format(test_X[TEXT_FEATURES_INPUT_NAME].shape))

        # TODO re fit model again on data_X data_y?
        # ...

        # Tansform Test TEXT_FEATURES
        test_X[TEXT_FEATURES_INPUT_NAME] = data_transformer.transform(test_X[TEXT_FEATURES_INPUT_NAME])

        # -- Predict on Test set --
        test_pred_y = model.predict(test_X)

        # Using scaling for target (Dont use with loss RMSLE!)
        if use_scaling_for_y:
            # Need to do the inverse y transform done above!
            scaled_test_pred_y = (pt_cox_y.inverse_transform(test_pred_y.reshape(-1,1)) - 1).astype(int)

            # Make sure scaled back targets are not negatives!
            scaled_target = np.clip(scaled_test_pred_y, a_min=0, a_max=None)
            test_pred_y = scaled_target

        # Store test set predictions in DataFrame and save to file
        test_pd = pd.DataFrame()
        test_pd['Id'] = test_profiles_ids_list
        test_pd['Predicted'] = test_pred_y

        test_tosubmit_folder = os.path.join(log_folder,'V-NNN-CombinedNeuralNetwork')
        # Create log folder if does not exist
        if not Path(test_tosubmit_folder).exists():
            os.mkdir(test_tosubmit_folder)
        test_name = 'VNNN-CombinedNeuralNetwork-2020-12-087'

        prediction_file_save_path = os.path.join(test_tosubmit_folder, test_name+'.csv')
        print('\nSaving prediction to "{:}"'.format(prediction_file_save_path))
        test_pd.to_csv(prediction_file_save_path, sep=',', index=False)


if __name__ == '__main__':
    main()