import struct
from array import array as pyarray
from numpy import *

def loadMnist(dataset="training", digits=arange(10)):
    
    # Loads MNIST files into 3D numpy arrays

    if dataset == "training":
        fname_img = './kmeans/train-images.idx3-ubyte'
        fname_lbl = './kmeans/train-labels.idx1-ubyte'
    elif dataset == "testing":
        fname_img = '../kmeans/t10k-images-idx3-ubyte'
        fname_lbl = '../kmeans/t10k-labels-idx1-ubyte'
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    labels_raw = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    images_raw = pyarray("B", fimg.read())
    fimg.close()

    indices = range(size)

    N = len(indices)

    images = zeros((N, rows*cols), dtype=uint8)

    labels = zeros((N), dtype=int8)
    for i, index in enumerate(indices):
        images[i] = array(images_raw[ indices[i]*rows*cols : (indices[i]+1)*rows*cols ])
        labels[i] = labels_raw[indices[i]]

    images = images.astype(float)/255.0

    return images, labels
