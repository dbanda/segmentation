from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras import optimizers
import numpy as np
import ade_layers

model = Sequential()
model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(3,384,384), padding='valid'))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(3,512,711)))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', padding='causal'))
model.add(MaxPooling2D(padding='same', pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(128, kernel_size=3, activation='relu', padding='causal'))
model.add(MaxPooling2D(padding='same', pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(256, kernel_size=3,, activation='relu'))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(256, kernel_size=3,, activation='relu'))
model.add(MaxPooling2D(border_mode='valid', pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, kernel_size=3, activation='relu'), padding='causal')
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, kernel_size=3, activation='relu'), padding='causal')
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, kernel_size=3, activation='relu'), padding='causal')
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, kernel_size=3, activation='relu'), padding='causal')
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, kernel_size=3, activation='relu'), padding='causal')
# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, kernel_size=3, activation='relu'), padding='causal')
# model.add(ZeroPadding2D(padding=(12, 12)))
model.add(Convolution2D(4096, kernel_size=7, activation='relu'), padding='causal') #dilation=4

model.add(Dropout(0.5))
model.add(Convolution2D(4096, kernel_size=1, activation='relu')) 
model.add(Dropout(0.5))
model.add(Convolution2D(151, kernel_size=1, activation='relu')) 
model.add(ZeroPadding2D(padding=(4, 4)))
model.add(Deconvolution2D(151, kernel_size = 16,strides=8,  bias=False ,padding='valid')) #group 151
# model.add(Deconvolution2D(151, 4, 4, bias=False, subsample=(5, 6))) #group 151

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

for i in range(2,3):
    X_train.append(a.load_image(i))
    Y_train.append(a.load_label(i))

X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(X_train.shape,Y_train.shape)
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("now training")
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)