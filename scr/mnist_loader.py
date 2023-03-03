import struct as st
import numpy as np
from PIL import Image
import random

def label_as_vector(label):
    v = np.zeros((10, 1))
    v[label] = 1.0
    return v

def get_data(test_percent=0.1):
    filename = {'images' : '../MNIST/train-images-idx3-ubyte', 'labels' : '../MNIST/train-labels-idx1-ubyte'}
    imagesfile = open(filename['images'],'rb')
    labelsfile = open(filename['labels'],'rb')

    imagesfile.seek(0)
    magic_images = st.unpack('>4B', imagesfile.read(4))
    labelsfile.seek(0)
    magic_lables = st.unpack('>4B', labelsfile.read(4))

    nImg = st.unpack('>I',imagesfile.read(4))[0] #num of images
    nR = st.unpack('>I',imagesfile.read(4))[0] #num of rows
    nC = st.unpack('>I',imagesfile.read(4))[0] #num of column
    nL = st.unpack('>I',labelsfile.read(4))[0] #num of labels

    nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
    images_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,imagesfile.read(nBytesTotal))).reshape((nImg,nR * nC, 1))
    labels_array = np.asarray(st.unpack('>'+'B'*nL,labelsfile.read(nBytesTotal))).reshape(nL)
    labels_array = [label_as_vector(label) for label in labels_array]

    data = [(img/255.0,lb) for img,lb in zip(images_array, labels_array)]
    cut = int(nImg * (1-test_percent))
    training_data = data[:cut]
    test_data = data[cut:]

    return training_data, test_data