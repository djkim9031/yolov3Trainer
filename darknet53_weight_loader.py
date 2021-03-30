#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:35:42 2021

@author: djkim9031
"""
#https://github.com/zzh8829/yolov3-tf2/blob/71208e1a6a77485c01d1bfe7247a244b1c39fa1a/yolov3_tf2/utils.py#L25

import numpy as np
import tensorflow as tf


def load_darknet_weights(model, weight_file):
    wf = open(weight_file,'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    darknet = model.layers[1]
    
    for i,layer in enumerate(darknet.layers):
        if layer.name.endswith('conv2d'):
            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]
            
            #darknet53.conv.74 shape (out_dims, in_dims, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            #tf darknet shape (height, width, in_dims, out_dims)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
            darknet.layers[i].set_weights([conv_weights])
            

            if i+1 < len(darknet.layers) and darknet.layers[i+1].name.endswith('bn'):

                # darknet53.conv.74 shape  (beta, gamma, mean, variance)
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                #tf darknet shape (gamma, beta, mean, variance)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

                darknet.layers[i+1].set_weights(bn_weights)
        else:
            continue
        
    assert len(wf.read()) == 0, 'Failed to read all data'
    wf.close()

            