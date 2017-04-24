from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.utils import to_categorical
from PIL import Image
from keras import optimizers
import numpy as np
import ade_layers

model = Sequential()
model.add(Convolution2D(64, kernel_size=(3,3), activation='relu', input_shape=(3,384,384), padding='same'))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(3,512,711)))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(padding='same', pool_size=(2,2), strides=(2, 2)))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(128, kernel_size=(3,3), activation='relu', padding='same'))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(padding='same', pool_size=(2,2), strides=(2, 2)))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(256, kernel_size=(3,3), activation='relu', padding='same'))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(256, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(padding='same', pool_size=(2,2), strides=(2, 2)))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, kernel_size=(3,3) , activation='relu', padding='same'))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, kernel_size=(3,3) , activation='relu', padding='same'))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, kernel_size=(3,3) , activation='relu', padding='same'))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, kernel_size=(3,3), activation='relu', padding='same'))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, kernel_size=(3,3) , activation='relu', padding='same'))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, kernel_size=(3,3) , activation='relu', padding='same'))
# model.add(ZeroPadding2D(padding=(12, 12)))
model.add(Convolution2D(4096, kernel_size=(7,7) , activation='relu', padding='same')) #dilation=4

model.add(Dropout(0.5))
model.add(Convolution2D(4096, kernel_size=(1,1), activation='relu', padding='same')) 
model.add(Dropout(0.5))
model.add(Convolution2D(151, kernel_size=(1,1), activation='relu', padding='same')) 
# c =Convolution2D(151, kernel_size=(1,1), activation='relu', padding='same')
# model.add(Deconvolution2D(1,kernel_size=(1,1),strides=2  ,padding='same')) #group 151

model.add(convolutional.Conv2DTranspose(151 , kernel_size=(16,16), strides=(8,8) , activation='relu', padding='same',dilation_rate=1)) #group 151
model.add(Flatten())
# model.add(Dense(384*384 * 151))
model.add(Reshape((-1, 151)))
model.add(Activation('softmax'))


# # 
# Conv2DTranspose(filters,
#                                           kernel_size=(3, 3),
#                                           strides=(2, 2),
#                                           padding='valid',
#                                           activation='relu')
# model.add(ZeroPadding2D(padding=(4, 4)))
# model.add(Dense(151, activation='relu'))
# model.add(Dense(151*14*14, activation='relu'))
# model.add(convolutional.Conv2DTranspose(151 , kernel_size=(2,2), strides=(2,2) , padding='same')) #group 151
# model.add(Deconvolution2D(151, 4, 4, bias=False, subsample=(5, 6))) #group 151
# keras.layers.convolutional.Conv2DTranspose(151, kernel_size=(4,4), strides=(1, 1), padding='same', 
#     data_format=None, activation=None, use_bias=True, 
#     kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# model.add(Flatten(input_shape=model.output_shape[1:]))
# model.add(Flatten())
# model.add(Dense(151))
# model.add(Activation('softmax'))
# model.add(Reshape((-1,-1,512,711)))

json_string = model.to_json()
print(json_string)

a = ade_layers.AdeSegDataLayer()
a.setup([],["data","label"])    
X_train = []
Y_train = []

def imTo3oneHot(im):
    onehot = to_categorical(im, 151)
    print("onehot",onehot.shape)


    return onehot
def showOneHot(onehot):
    arr = onehot.argmax(1)
    arr = arr.reshape((384,384))
    print("imgsize",arr.size)
    im = Image.fromarray(np.uint8(arr))
    im.show()

for i in range(2,3):
    X_train.append(a.load_image(i))
    Y_train.append(imTo3oneHot(a.load_label(i)))
    
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(X_train.shape,Y_train.shape)
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("now training")
# dummy_input = np.ones((1, 3, 384, 384))
# For TensorFlow dummy_input = np.ones((32, 12, 12, 3))
# preds = model.predict(dummy_input)
# print(preds.shape)
# model.fit(X_train, Y_train, 
#           batch_size=1, nb_epoch=10, verbose=1)


# x = X_train[0]
# y = Y_train[0]
preds = model.predict(X_train)
print(preds.shape)
showOneHot(preds[0])