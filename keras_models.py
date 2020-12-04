import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation


from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model



def TextOnlyModel(nbr_text_features, text_features_input_name, output_name):
    # Input
    text_features_input = Input(shape=(nbr_text_features,), name=text_features_input_name)

    kern_init = 'glorot_uniform'

    # Dense (TODO embeding?) for text features
    text_features = text_features_input
    text_features = Dense(nbr_text_features, activation='relu', kernel_initializer=kern_init)(text_features)
    text_features = Dropout(0.25)(text_features)
    text_features = Dense(64, activation='relu', kernel_initializer=kern_init)(text_features)
    text_features = Dropout(0.25)(text_features)
    text_features = Dense(256, activation='relu', kernel_initializer=kern_init)(text_features)
    text_features = Dropout(0.25)(text_features)
    text_features = Dense(512, activation='relu', kernel_initializer=kern_init)(text_features)
    text_features = Dropout(0.25)(text_features)
    text_features_out = text_features

    # Final output dense regression layer
    output_likes = Dense(1, name=output_name)(text_features_out)

    return Model(inputs=text_features_input, outputs=output_likes)




def ImageAndTextModel(image_height, image_width, image_nbr_channels, nbr_text_features,
                      image_input_name, text_features_input_name, output_name):

    # Inputs
    image_input = Input(shape=(image_height, image_width, image_nbr_channels), name=image_input_name)
    text_features_input = Input(shape=(nbr_text_features,), name=text_features_input_name)


    # CNN block for image
    kern_init = 'glorot_uniform'
    image_cnn = image_input
    AlexNet = Sequential() # https://analyticsindiamag.com/hands-on-guide-to-implementing-alexnet-with-keras-for-multi-class-image-classification/

    #1st Convolutional Layer
    AlexNet.add(Conv2D(filters=96, input_shape=(32,32,3), kernel_size=(11,11), strides=(4,4), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #2nd Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #3rd Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    #4th Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    #5th Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #Passing it to a Fully Connected layer
    AlexNet.add(Flatten())
    # 1st Fully Connected Layer
    AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    AlexNet.add(Dropout(0.4))

    #2nd Fully Connected Layer
    AlexNet.add(Dense(4096))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #3rd Fully Connected Layer
    AlexNet.add(Dense(1000))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #Output Layer
    AlexNet.add(Dense(1))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('linear'))
    image_out = AlexNet(image_input)

    # Dense (TODO embeding?) for text features
    text_features = text_features_input
    text_features = Dense(nbr_text_features, activation='relu', kernel_initializer=kern_init)(text_features)
    text_features = Dropout(0.25)(text_features)
    text_features = Dense(64, activation='relu', kernel_initializer=kern_init)(text_features)
    text_features = Dropout(0.25)(text_features)
    text_features = Dense(128, activation='relu', kernel_initializer=kern_init)(text_features)
    text_features = Dropout(0.25)(text_features)
    text_features_out = text_features


    # Merge all available features into a single large vector via concatenation
    #x = image_out # TODO test image only
    x = Concatenate(name='merge')([image_out, text_features_out]) # TODO uncomment to use image + text features


    # Dense layers
#    x = LayerNormalization()(x)
#    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_initializer=kern_init)(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_initializer=kern_init)(x)
    x = Dropout(0.5)(x)

    # Final output dense  layer
    output_likes = Dense(1, name=output_name)(x)

    return Model(inputs=[image_input, text_features_input], outputs=output_likes)