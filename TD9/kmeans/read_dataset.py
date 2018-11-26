# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 11:06:26 2014

@author: jb
"""
from mnist2 import load_mnist
from numpy import append,linalg, real, trace, mean, argmax, sum, eye, array, int8, uint8, zeros, float64, float32, concatenate, transpose, dot,shape

def read_dataset(taille_test,size_tot):
    
    images_test, labels_test = load_mnist('testing', digits=[0,1,2,3,4,5,6,7,8,9])
    images_test=images_test.astype(float64)
    images_test=images_test[1:(taille_test+1),:,:]
    labels_test=labels_test[1:(taille_test+1)]
    return images_test, labels_test