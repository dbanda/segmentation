#import caffe

import numpy as np
from PIL import Image

import random

class AdeSegDataLayer:
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - ade_dir: path to ADE dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)


        """
        # config
        # params = eval(self.param_str)
        params =  {'seed': 1337, 'split': '', 'mean': (109.5388, 118.6897, 124.6901)}
        self.ade_dir = './ADEChallengeData2016' # add the path to the dataset
        self.split = params['split']
        self.split_dir = '~/Downloads/ADEChallengeData2016/images/training' # add path to the split files
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.indices = np.zeros(202010)
        # split_f  = self.split_dir+'/{}.txt'.format(self.split)
        # self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 1

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self):#, bottom, top):
        # assign output
        # top[0].data[...] = self.data
        # top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx=0,show=False):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        idx = self.idx
        fname = '{}/images/training/ADE_train_{:08d}.jpg'.format(self.ade_dir, idx)
        im = Image.open(fname)
        width, height = im.size
        if width < 384 or height <384:
            # if this image is too small to be cropped skip it
            return None,fname

        ## crop to 384*384

        im = im.crop((0,0,384,384))
        if show:
            im.show()

        in_ = np.array(im, dtype=np.float32)
        if (in_.ndim == 2):
            in_ = np.repeat(in_[:,:,None], 3, axis = 2)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_,fname


    def load_label(self, idx=0,show=False):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        idx = self.idx
        fname = '{}/annotations/training/ADE_train_{:08d}.png'.format(self.ade_dir, idx)
        im = Image.open(fname)
        width, height = im.size
        if width < 384 or height <384:
            # if this image is too small to be cropped skip it
            return None,fname
        im = im.crop((0,0,384,384))
        if show:
            im.show()
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label,fname


if __name__ == '__main__':
    a = AdeSegDataLayer()
    a.setup([],["data","label"])
    print(a.load_image(1).shape)
    print(a.load_label(1).shape)

'~/Downloads/ADEChallengeData2016/images/training/ADE_train_00000001.jpg'
'~/Downloads/ADEChallengeData2016/images/training/ADE_train_00000001.jpg'
