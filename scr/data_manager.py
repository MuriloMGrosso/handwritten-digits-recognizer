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

def rotate_image(image, max):
    angle = random.uniform(-max, max)
    return image.rotate(angle)

def scale_image(image, scale, stretch):
    width = random.uniform(scale[0], scale[1])
    height = width * random.uniform(stretch[0], stretch[1])
    new_size = (int(width * 28), int(height * 28))

    left = new_size[0]/2 - 14
    up = new_size[1]/2 - 14
    right = new_size[0]/2 + 14
    down = new_size[1]/2 + 14

    return image.resize(new_size).crop((left,up,right,down)), width

def offset_image(image, max):
    offset_x = int(random.uniform(-max,max))
    offset_y = int(random.uniform(-max,max))
    return image.crop((offset_x,offset_y,28 + offset_x, 28 + offset_y))

def randomize_data(data):
    images_array = [x for x,y in data]
    new_data = []

    for i in range(len(data)):
        image = Image.fromarray(np.uint8(images_array[i].reshape(28,28)*255))
        image, scale = scale_image(image,(0.4,1.2),(0.8,1.2))
        image = rotate_image(image,30)
        image = offset_image(image, (1.2-scale)*10)

        noise_weight = random.uniform(0.01,0.1)
        image_array = np.asarray(image)*(1.0 - noise_weight) + np.random.randint(int(255*noise_weight),size=(28,28))

        new_data.append((image_array.reshape(28 * 28, 1)/255.0, data[i][1]))

    return new_data