import model
import numpy as np
from keras.preprocessing.image import *
m = model.get_frontend(1024,1024)
m = model.add_softmax(m)
print(m.summary())
weights_data = np.load('/Users/dalitsobanda/github/caffe-tensorflow/dilatednet.npy', encoding='latin1').item()
print("loaded layers ", weights_data.keys())
for layer in m.layers:
    if layer.name in weights_data.keys():
        layer_weights = weights_data[layer.name]
        # print(len(layer_weights), layer.name)
        # print(layer_weights.keys())
        # print (layer_weights['weights'].shape)
        if 'biases' in layer_weights:
            layer.set_weights((layer_weights['weights'],
                               layer_weights['biases']))
        else:
            print("bad layer",layer.name)
            p = np.pad(layer_weights['weights'],
                ((0,0),
                (0,0),
                (0,150),
                (0,0)), mode='reflect')
            layer.set_weights((p,))
    else:
        print("layer not found", layer.name)

img = load_img('./ADEChallengeData2016/images/validation/ADE_val_00000012.jpg',
    target_size=(1024,1024))
img.show()
arr=img_to_array(img)
pred = m.predict(np.array([arr]))
print(pred.shape)
print(np.histogram(pred[0].argmax(1), np.arange(151)))
import matplotlib.pyplot as plt
plt.hist(pred[0].argmax(1), bins=np.arange(151))
plt.xticks(np.arange(151))
plt.show()
im = array_to_img(pred[0].argmax(1).reshape((128,128,1)) )
im.show()

img = load_img('./ADEChallengeData2016/annotations/validation/ADE_val_00000012.png',
    target_size=(128,128),grayscale=True)
img.show()

arr=img_to_array(img)
plt.hist(arr.ravel(), bins=np.arange(151))
plt.xticks(np.arange(151))
plt.show()
