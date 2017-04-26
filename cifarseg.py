'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from model import get_frontend, add_softmax
from PIL import Image
from models import enet
import numpy as np
from keras.preprocessing.image import (
    load_img, img_to_array,
    flip_axis)

n_samples = 100
batch_size = 10
nb_classes = 151
nb_epoch = 3
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32,32

# The CIFAR10 images are RGB.
img_channels = 3

# output dimensions
mask_rows, mask_cols = 32,32

# The data, shuffled and split between train and test sets:
X_train = np.zeros((n_samples,3,img_rows, img_cols))
y_train = np.zeros((n_samples,mask_rows*mask_cols,nb_classes))
X_test = np.zeros((n_samples,3,img_rows, img_cols))
y_test = np.zeros((n_samples,mask_rows*mask_cols,nb_classes))

for i in range(1,n_samples):
    if i%100 ==0:
        print("loaded ", i)
    ade_dir = '/Users/dalitsobanda/github/segmentation/ADEChallengeData2016/'

    train_img_fname = '{}/images/training/ADE_train_{:08d}.jpg'.format(ade_dir, i)
    img = load_img(train_img_fname, grayscale=False, target_size= (img_rows, img_cols))
    # img.show(title=train_img_fname)
    x = img_to_array(img, data_format='channels_first')
    X_train[i] = x

    train_mask_fname = '{}/annotations/training/ADE_train_{:08d}.png'.format(ade_dir, i)
    img = load_img(train_mask_fname, grayscale=True, target_size= (mask_rows, mask_cols))
    # img.show(title=train_img_fname)
    y = img_to_array(img, data_format='channels_first')
    y_train[i] = np_utils.to_categorical(y,nb_classes)

    val_img_fname = '{}/images/validation/ADE_val_{:08d}.jpg'.format(ade_dir, i)
    img = load_img(val_img_fname, grayscale=False, target_size= (img_rows, img_cols))
    x = img_to_array(img, data_format='channels_first')
    X_test[i] = x
    
    val_mask_fname = '{}/annotations/validation/ADE_val_{:08d}.png'.format(ade_dir, i)
    img = load_img(val_mask_fname, grayscale=True, target_size= (mask_rows, mask_cols))
    y = img_to_array(img, data_format='channels_first')
    y_test[i] = np_utils.to_categorical(y,nb_classes)

# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
# Y_train = np_utils.to_categorical(y_train, nb_classes).reshape((2000,151,16,16))
# Y_test = np_utils.to_categorical(y_test, nb_classes).reshape((2000,151,16,16))

#dilatednet
# model = add_softmax(
#     get_frontend(img_rows, img_cols))

#enet
model,name = enet.autoencoder(nc=nb_classes,input_shape=(img_rows, img_cols), 
    output_shape=(mask_rows, mask_cols) )
print(model.summary())


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(X_train, augment=False, seed=seed)
mask_datagen.fit(y_train.reshape(-1,1,mask_cols, mask_rows ),augment=False, seed=seed)
ade_dir = '/Users/dalitsobanda/github/segmentation/ADEChallengeData2016/'

image_generator = image_datagen.flow_from_directory(
    ade_dir+'images',
    target_size=(img_cols,img_rows),
    color_mode='rgb',
    classes=['training'],
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    ade_dir+'annotations',
    target_size = (mask_rows, mask_cols),
    color_mode='grayscale',
    classes=['training'],
    class_mode=None,
    seed=seed)

def mask_gen(mask_generator):
    while True:
        n = next(mask_generator)
        out = np.zeros((n.shape[0], mask_cols*mask_rows,nb_classes))
        for i in range(len(n)):
            out[i] = np_utils.to_categorical(n[i],nb_classes)
        yield out

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_gen(mask_generator))
n = next(image_generator)
arr = np.uint8(n[0]).transpose((1,2,0))
print(arr.shape)
im = Image.fromarray(arr)
im.show()

n = next(mask_generator)
arr = np.uint8(n[0]).reshape(mask_cols, mask_rows)
print(arr.shape)
im = Image.fromarray(arr)
im.show()



if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    print("starting to train")
    # model.fit_generator(datagen.flow(X_train, y_train,
    #                     batch_size=batch_size),
    #                     steps_per_epoch=X_train.shape[0],
    #                     epochs=nb_epoch,
    #                     validation_data=(X_test, y_test))
    model.fit_generator(train_generator,
                        steps_per_epoch=X_train.shape[0],
                        epochs=nb_epoch,
                        validation_data=(X_test, y_test))
    choice = np.random.choice(X_train.shape[0], 3)
    test = X_train[choice]
    pred = model.predict(test)
    print(pred.shape)
    # for i in range(pred.shape[0]):
    #     arr = np.uint8(test[i]).transpose((2,1,0))
    #     print(arr.shape)
    #     im = Image.fromarray(arr)
    #     im.show()

    #     arr = pred[i].argmax(1).reshape((mask_rows, mask_cols))
    #     arr = np.uint8(arr)
    #     im = Image.fromarray(arr)
    #     im.show()

