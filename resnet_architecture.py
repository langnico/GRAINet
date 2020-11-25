import numpy as np
from keras.layers import add, Conv2D, Activation, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, GlobalAveragePooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

# set up printing values from numpy arrays to not use exponential representation
np.set_printoptions(suppress=True)


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):
    """
    Implementation of the convolutional block. There are conv layers at the shortcut.

    Arguments:
    input_tensor - input tensor of shape ??? 4D tensor with shape: (samples, rows, cols, channels)
    kernel_size - integer, specifying the shape of the middle CONV's window (kernel size) for the main path
    filters - list of integers, defining the number of filters of 3 CONV layers of the main path
    stage - integer, used to name the layers, depending on their position in the network (current stage)
    block - characters, used to name the layers, depending on their position in the network (current block label)
    stride - integer, specifying the stride size to be used

    Returns:
    x - output of the identity block, tensor of shape ?????? 4D tensor with shape: (samples, new_rows, new_cols, filters)
    """
    # retriving filters
    F1, F2, F3 = filters

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x_shortcut = input_tensor

    # First component of main path
    x = Conv2D(filters=F1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # Second component of main path
    x = Conv2D(filters=F2, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third component of main path
    x = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    # shortcut convolution layer
    x_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=strides, padding='valid', name=conv_name_base + '1')(
        x_shortcut)
    x_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(x_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    x = add([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def identity_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):
    """
    Implementation of the identity block. There is a direct shortcut (identity mapping).

    Arguments:
    input_tensor - input tensor of shape ??? 4D tensor with shape: (samples, rows, cols, channels)
    kernel_size - integer, specifying the shape of the middle CONV's window (kernel size) for the main path
    filters - list of integers, defining the number of filters of 3 CONV layers of the main path
    stage - integer, used to name the layers, depending on their position in the network (current stage)
    block - characters, used to name the layers, depending on their position in the network (current block label)
    stride - integer, specifying the stride size to be used

    Returns:
    x - output of the identity block, tensor of shape ?????? 4D tensor with shape: (samples, new_rows, new_cols, filters)
    """
    # retriving filters
    F1, F2, F3 = filters

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x_shortcut = input_tensor

    # First component of main path
    x = Conv2D(filters=F1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # Second component of main path
    x = Conv2D(filters=F2, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third component of main path
    x = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    x = add([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def FCN_grainsize(img_input, bins=22, output_scalar=False):
    """
    Fully convolutional network (FCN). Reduce the input resolution by factor of 8.
    FCN allows to use different input resolutions.

    Arguments:
    img_input - Input as a tensor with input_shape

    Returns:
    model - a Model() instance in Keras
    """

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), name='conv1', padding='same')(img_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(2, 2))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', strides=(1, 1))

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(2, 2))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', strides=(1, 1))

    x = conv_block(x, 3, [128, 128, 512], stage=4, block='a', strides=(2, 2))
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='b', strides=(1, 1))

    if not output_scalar:
        # output (invariant to input size)
        x = Conv2D(filters=bins, kernel_size=(1, 1), strides=(1, 1), name='conv3', padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax', name='histogram_prediction')(x)
    else:
        x = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), name='conv3', padding='same')(x)
        x = GlobalAveragePooling2D()(x)

    return Model(inputs=img_input, outputs=x)


# Data generator to load the training data as batches
def generator(features, labels, batch_size, img_rows, img_cols, channels, bins):
    # Create empty arrays to contain batch of features and labels
    batch_features = np.zeros((batch_size, img_rows, img_cols, channels))
    batch_labels = np.zeros((batch_size, bins))

    while True:
        for i in range(batch_size):
             # choose random index in features
             index = np.random.choice(a=features.shape[0], size=1)

             #Optional: Add some data augmentation here

             batch_features[i] = features[index, :, :, :]
             batch_labels[i] = labels[index]

        yield batch_features, batch_labels


def initialize_weights(model, layer_name=None):
    """
    Re-initialize the weights before starting a new experiment/training
    :param model: keras model
    :param layer_name: if None initialize all model layers, else only specified layer
    :return:
    """
    session = K.get_session()
    if layer_name is None:
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                #print('initialize weights of layer: {}'.format(layer.name))
                layer.kernel.initializer.run(session=session)
    else:
        layer = model.get_layer(name=layer_name)
        if hasattr(layer, 'kernel_initializer'):
            print('initialize weights of layer: {}'.format(layer.name))
            layer.kernel.initializer.run(session=session)

