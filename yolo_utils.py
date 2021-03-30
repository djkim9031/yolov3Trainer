#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 11:54:07 2021

@author: djkim9031
"""


import tensorflow as tf
EPSILON = 1e-7

def xywh_to_x1x2y1y2(box):
    xy = box[...,0:2]
    wh = box[...,2:4]
    
    x1y1 = xy - wh/2
    x2y2 = xy + wh/2
    
    y_box = tf.concat([x1y1,x2y2],axis=-1)
    
    return y_box

def xywh_to_y1x1y2x2(box):
    x = box[...,0]
    y = box[...,1]
    w = box[...,2]
    h = box[...,3]
    
    yx = tf.concat([y,x],axis=-1)
    hw = tf.concat([h,w],axis=-1)
    
    y1x1 = yx - hw/2
    y2x2 = yx + hw/2
    
    y_box = tf.concat([y1x1,y2x2],axis=-1)
    
    return y_box

def broadcast_iou(box_a,box_b):
    """
    

    Parameters
    ----------
    box_a : a tensor full of boxes, e.g., (Batch,grid*grid*3(=N),4), box is in x1y1x2y2
    box_b : another tensor full of boxes (Batch,100(=M),4)

    """
    
    #(B,N,1,4)
    box_a = tf.expand_dims(box_a,-2)
    
    #(B,1,M,4)
    box_b = tf.expand_dims(box_b,-3)
    
    #(B,N,M,4)
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_a),tf.shape(box_b))
    
    #(B,N,M,4)
    box_a = tf.broadcast_to(box_a,new_shape)
    box_b = tf.broadcast_to(box_b,new_shape)
    
    #Each (B,N,M,1)
    al, at, ar, ab = tf.split(box_a,4,-1)
    bl, bt, br, bb = tf.split(box_b,4,-1)
    
    #Each (B,N,M,1)
    left = tf.math.maximum(al,bl)
    right = tf.math.minimum(ar,br)
    top = tf.math.maximum(at,bt)
    bottom = tf.math.minimum(ab,bb)
    
    #Each (B,N,M,1)
    iw = tf.clip_by_value(right-left,0,1)
    ih = tf.clip_by_value(bottom-top,0,1)
    i = iw * ih
    
    #Each (B,N,M,1)
    area_a = (ar-al)*(ab-at)
    area_b = (br-bl)*(bb-bt)
    union = area_a + area_b - i
    
    #(B,N,M)
    iou = tf.squeeze(i/(union+EPSILON),axis=-1)
    
    return iou

def binary_cross_entropy(logits, labels):
    logits = tf.clip_by_value(logits,EPSILON,1-EPSILON)
    return -(labels*tf.math.log(logits) + (1-labels)*tf.math.log(1-logits))
    
    
    
    
    
    
    