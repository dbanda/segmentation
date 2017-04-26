from keras.engine.topology import Input
from keras.models import Model
from keras.layers.core import Activation, Reshape, Dense
from keras.utils import plot_model
from models import encoder, decoder


def transfer_weights(model, weights=None):
    '''
    Always trains from scratch; never transfers weights
    '''
    print('ENet has found no compatible pretrained weights! Skipping weight transfer...')
    return model


def autoencoder(nc, input_shape, output_shape,
                loss='categorical_crossentropy',
                optimizer='adadelta'):
    # data_shape = input_shape[0] * input_shape[1] if input_shape and None not in input_shape else None
    data_shape = input_shape[0] * input_shape[1] if input_shape and None not in input_shape else -1  # TODO: -1 or None?
    inp = Input(shape=(3,input_shape[0], input_shape[1]))
    enet = encoder.build(inp)
    enet = decoder.build(enet, nc=nc, in_shape=input_shape)

    # enet = Reshape((data_shape, nc), input_shape=(input_shape[0], input_shape[1], nc))(enet)
    # from keras import backend as K
    # enet = K.reshape(enet, (data_shape, nc))
    # print(K.int_shape(enet))
    enet = Reshape((data_shape, nc))(enet)  # TODO: need to remove data_shape for multi-scale training
    enet = Activation('softmax')(enet)
    # enet = Reshape((-1,output_shape[0],output_shape[0],151))
    # print(enet, )

    model = Model(inputs=inp, outputs=enet)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mean_squared_error'])
    name = 'enet'
    print(model.summary())
    return model, name

if __name__ == "__main__":
    autoencoder, name = autoencoder(nc=151, input_shape=(512, 512))
    # plot_model(autoencoder, to_file='{}.png'.format(name), show_shapes=True)
