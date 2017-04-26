from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.core import Permute, SpatialDropout2D
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D


def initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, conv_stride=(2, 2)):
    conv = Convolution2D(nb_filter, (nb_row, nb_col), padding='same', strides=conv_stride)(inp)
    max_pool = MaxPooling2D()(inp)
    merged = concatenate([conv, max_pool], axis=1)
    return merged


def bottleneck(inp, output, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output // internal_scale
    encoder = inp

    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Convolution2D(internal, (input_stride, input_stride), padding='same', strides=(input_stride, input_stride), use_bias=False)(encoder)
    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder) # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # conv
    if not asymmetric and not dilated:
        encoder = Convolution2D(internal, (3, 3), padding='same')(encoder)
    elif asymmetric:
        encoder = Convolution2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
        encoder = Convolution2D(internal, (asymmetric, 1), padding='same')(encoder)
    elif dilated:
        encoder = Convolution2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise(Exception('You shouldn\'t be here'))

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)
    
    # 1x1
    encoder = Convolution2D(output, (1, 1), padding='same', use_bias=False)(encoder)

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # other branch
    if downsample:
        print(encoder.get_shape(), inp.get_shape(), other.get_shape(),output)
        other = MaxPooling2D()(other)
        
        other = Permute((3, 2, 1))(other)
        pad_featmaps = output - inp.get_shape().as_list()[1]
        tb_pad = (0, 0)
        lr_pad = (0, pad_featmaps)
        print(other.get_shape(), "pad", lr_pad)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((3, 2, 1    ))(other)

    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)
    return encoder


def build(inp, dropout_rate=0.01):
    enet = initial_block(inp)
    enet = bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
    for i in range(4):
        enet = bottleneck(enet, 64, dropout_rate=dropout_rate) # bottleneck 1.i
    
    enet = bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
    # bottleneck 2.x and 3.x
    print("bottleneck 2.x and 3.x   ")
    for i in range(2):
        enet = bottleneck(enet, 128)  # bottleneck 2.1
        enet = bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3
        enet = bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
        enet = bottleneck(enet, 128)  # bottleneck 2.5
        enet = bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7
        enet = bottleneck(enet, 128, dilated=16)  # bottleneck 2.8
    return enet

