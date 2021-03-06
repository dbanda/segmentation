# dilatedNet

# Please use Keras version 2.0.3 and tf 1.0.0 otherwise bad things may
# happen

# Download training data http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip 

# Basically, predictions are a pixelwise output in 150 classes. The classes
# are described in ./ADEChallengeData/objectinfo150.txt. Each pixel is 
# classified as a one hot vector into one of the 150 classes.



from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.utils import to_categorical
from PIL import Image
from keras import optimizers
import numpy as np
import ade_layers
import random

# implemtation of dilatedNet in keras
model = Sequential()

# model.add(ZeroPadding2D((1, 1), input_shape=(input_width, input_height, 3)))
model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu', name='conv1_1', input_shape=(3,384, 384), padding='same' ))
model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu', name='conv1_2', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu', name='conv2_1',padding='same'))
model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu', name='conv2_2',padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(256, kernel_size=(3, 3), activation='relu', name='conv3_1',padding='same'))
model.add(Convolution2D(256, kernel_size=(3, 3), activation='relu', name='conv3_2',padding='same'))
model.add(Convolution2D(256, kernel_size=(3, 3), activation='relu', name='conv3_3',padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(512, kernel_size=(3, 3), activation='relu', name='conv4_1',padding='same'))
model.add(Convolution2D(512, kernel_size=(3, 3), activation='relu', name='conv4_2',padding='same'))
model.add(Convolution2D(512, kernel_size=(3, 3), activation='relu', name='conv4_3',padding='same'))

model.add(Convolution2D(512, kernel_size=(3,3) , dilation_rate=2, activation='relu', name="conv5_1",padding='same'))
model.add(Convolution2D(512, kernel_size=(3,3),dilation_rate=2, activation='relu', name="conv5_2",padding='same'))
model.add(Convolution2D(512, kernel_size=(3,3),dilation_rate=2 , activation='relu', name="conv5_3",padding='same'))

model.add(Convolution2D(4096, kernel_size=(7,7) ,dilation_rate=4, activation='relu', name="fc6", padding='same')) 
# model.add(Dropout(0.5))
model.add(Convolution2D(4096, kernel_size=(1,1), activation='relu', name="fc7",padding='same')) 
# model.add(Dropout(0.5))
model.add(Convolution2D(151, kernel_size=(1,1), activation='linear', name='fc-final',padding='same')) 
# c =Convolution2D(151, kernel_size=(1,1), activation='relu', padding='same')
# model.add(Deconvolution2D(1,kernel_size=(1,1),strides=2  ,padding='same')) #group 151

model.add(convolutional.Conv2DTranspose(151 , kernel_size=(16,16), strides=(8,8) , activation='linear', padding='same')) #group 151
# model.add(Flatten())
# model.add(Dense(384*384 * 151))
model.add(Reshape((384*384, 151)))
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
    # showOneHot(onehot)
    # print("onehot",onehot.shape)
    return onehot
def showOneHot(onehot):
    arr = onehot.argmax(1)
    arr = arr.reshape((384,384))
    for i in range(384):
        for j in range(384):
            if arr[i][j] > 151:
                raise ValueError

    print("imgsize",arr.size)
    im = Image.fromarray(np.uint8(arr))
    im.show()



def my_generator():
    idx = 0
    NUM_TRAIN=150
    batch_size = 20
    X_train = []
    Y_train = []
    while True:
        (x,x_name),(y,y_name) = a.load_image(), a.load_label()
        if x !=None and y != None:
            # print (x_name, y_name)
            X_train.append(x)
            Y_train.append(y.reshape((384*384,1)))
            idx += 1
            
            a.forward()
            if len(X_train) == batch_size:
                (b_x,b_y) = np.array(X_train),np.array(Y_train)
                X_train=[]
                Y_train=[]
                yield (b_x,b_y)
        else:
            a.forward()
    
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(X_train.shape,Y_train.shape)
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# dummy_input = np.ones((1, 3, 384, 384))
# For TensorFlow dummy_input = np.ones((32, 12, 12, 3))
# preds = model.predict(dummy_input)
# print(preds.shape)
print("now training")
# for xi in X_train:
#     print(xi.shape)
# print()
# for yi in Y_train:
#     print(yi.shape)
# model.fit(X_train, Y_train, 
#           batch_size=NUM_TRAIN, epochs=30, verbose=1)
model.fit_generator(my_generator(), epochs=1, verbose=1, steps_per_epoch=30)

# pick random test image to display
# i = random.randint(2,NUM_TRAIN)
preds = model.predict(np.array(a.load_image(show=True)))
print(preds.shape)
showOneHot(preds[0]) #show prediction
a.load_label(i,True) #show label