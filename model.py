import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate

from tensorflow.keras.models import Model

# Names for the model inputs/output for better referencing (using dicts)
IMAGE_INPUT_NAME = 'image'
TEXT_FEATURES_INPUT_NAME = 'text_features'
OUTPUT_NAME = 'likes'


def TextOnlyModel(nbr_text_features):
    # Input
    text_features_input = Input(shape=(nbr_text_features,), name=TEXT_FEATURES_INPUT_NAME)

    kern_init = 'glorot_uniform'

    # Dense (TODO embeding?) for text features
#    x = LayerNormalization()(text_features_input)
    x = Dense(nbr_text_features, activation='relu', kernel_initializer=kern_init)(text_features_input)
#    x = Dropout(0.25)(x)
    x = Dense(32, activation='relu', kernel_initializer=kern_init)(x)
#    x = Dropout(0.25)(x)
    x = Dense(16, activation='relu', kernel_initializer=kern_init)(x)
#    x = Dropout(0.25)(x)

    # Final output dense regression layer
    output_likes = Dense(1, name=OUTPUT_NAME)(x)

    return Model(inputs=text_features_input, outputs=output_likes)



    
def ImageAndTextModel(image_height, image_width, image_nbr_channels, nbr_text_features):
    
    # Inputs
    image_input = Input(shape=(image_height, image_width, image_nbr_channels), name=IMAGE_INPUT_NAME)
    text_features_input = Input(shape=(nbr_text_features,), name=TEXT_FEATURES_INPUT_NAME)


    # CNN block for image
    kern_init = 'glorot_uniform'
    image_cnn = image_input
#    image_cnn = LayerNormalization()(image_cnn)
    image_cnn = Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer=kern_init)(image_cnn)
    image_cnn = MaxPooling2D(pool_size=(2,2))(image_cnn)
    image_cnn = Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer=kern_init)(image_cnn)
    image_cnn = MaxPooling2D(pool_size=(2,2))(image_cnn)
    image_cnn = Dropout(0.5)(image_cnn)
    image_out = Flatten()(image_cnn)


    # Dense (TODO embeding?) for text features
    text_features = text_features_input
    text_features = Dense(nbr_text_features, activation='relu', kernel_initializer=kern_init)(text_features)
#    text_features = Dropout(0.25)(text_features)
    text_features = Dense(32, activation='relu', kernel_initializer=kern_init)(text_features)
#    text_features = Dropout(0.25)(text_features)
    text_features = Dense(16, activation='relu', kernel_initializer=kern_init)(text_features)
#    text_features = Dropout(0.25)(text_features)
    text_features_out = text_features


    # Merge all available features into a single large vector via concatenation
    x = Concatenate(name='merge')([image_out, text_features_out])

    # Dense layers
#    x = LayerNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation='relu', kernel_initializer=kern_init)(x)
    x = Dropout(0.5)(x)
#    x = Dense(16, activation='relu', kernel_initializer=kern_init)(x)
#    x = Dropout(0.5)(x)

    # Final output dense  layer
    output_likes = Dense(1, name=OUTPUT_NAME)(x)

    return Model(inputs=[image_input, text_features_input], outputs=output_likes)