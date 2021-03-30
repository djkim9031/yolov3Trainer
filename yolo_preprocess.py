#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 14:17:03 2021

@author: djkim9031
"""


import tensorflow as tf
import numpy as np
import sys
sys.path.append('/Users/djkim9031/Desktop/YOLOv3_Trainer/')
from yolov3 import anchors_wh

class Preprocessor(object):
    def __init__(self, is_train, num_classes, output_shape=(416,416)):
        self.is_train = is_train
        self.num_classes = num_classes
        self.output_shape = output_shape
        
    def __call__(self,example):
        features = self.parse_tfexample(example)
        
        encoded = features['image/encoded']
        image = tf.io.decode_jpeg(encoded)
        image = tf.cast(image, tf.float32)
        
        classes, bboxes = self.parse_y_features(features)
        image, bboxes = self.random_flip_image_and_label(image, bboxes)
        image, bboxes = self.random_crop_image_and_label(image, bboxes)
        image = tf.image.resize(image, self.output_shape)
        image = tf.cast(image, tf.float32)/127.5 - 1
        
        label = (
                self.preprocess_label_for_one_scale(classes, bboxes, 52, np.array([6,7,8])),
                self.preprocess_label_for_one_scale(classes, bboxes, 26, np.array([3,4,5])),
                self.preprocess_label_for_one_scale(classes, bboxes, 13, np.array([0,1,2])),
            )
        return image, label
        
    def random_flip_image_and_label(self, image, bboxes):
        '''
        flip left and right for 50% of images

        '''
        r = tf.random.uniform([1])
        if r < 0.5:
            image = tf.image.flip_left_right(image)
            xmin, ymin, xmax, ymax = tf.split(bboxes, [1,1,1,1],-1)
            xmin, xmax = 1-xmax, 1-xmin
            bboxes = tf.squeeze(tf.stack([xmin,ymin,xmax,ymax],axis=1),axis=-1)
            
        return image, bboxes
    
    def get_random_crop_delta(self, bboxes):
        '''
        Get a random crop which includes all bounding boxes.
        Since all bboxes here belong to one image, we can calculate the minimum of
        all xmin and ymin, and the maximum of all xmin and ymax to get the area that
        can include all boxes. The crop will be randomly picked between this area boundary
        and the boundary of the whole image
        '''
        min_xmin = tf.math.reduce_min(bboxes[...,0])
        min_ymin = tf.math.reduce_min(bboxes[...,1])
        max_xmax = tf.math.reduce_max(bboxes[...,2])
        max_ymax = tf.math.reduce_max(bboxes[...,3])
        
        xmin_delta = tf.random.uniform([1],0,min_xmin)
        ymin_delta = tf.random.uniform([1],0,min_ymin)
        xmax_delta = tf.random.uniform([1],0,1-max_xmax)
        ymax_delta = tf.random.uniform([1],0,1-max_ymax)
        
        return xmin_delta, ymin_delta, xmax_delta, ymax_delta
    
    def random_crop_image_and_label(self, image, bboxes):
        '''
        Crop images randomly at 50% chance but preserve all bounding boxes. The crop is guaranteed to
        include all bounding boxes
        '''
        r = tf.random.uniform([1])
        if r < 0.5:
            xmin_delta, ymin_delta, xmax_delta, ymax_delta = self.get_random_crop_delta(bboxes)
            xmin, ymin, xmax, ymax = tf.split(bboxes, [1,1,1,1],-1)
            
            xmin = (xmin - xmin_delta) / (1 - xmin_delta - xmax_delta)
            ymin = (ymin - ymin_delta) / (1 - ymin_delta - ymax_delta)
            xmax = (xmax - xmin_delta) / (1 - xmin_delta - xmax_delta)
            ymax = (ymax - ymin_delta) / (1 - ymin_delta - ymax_delta)
            
            bboxes = tf.squeeze(tf.stack([xmin, ymin, xmax, ymax],axis=1),axis=-1)
            h = tf.cast(tf.shape(image)[0], dtype = tf.float32)
            w = tf.cast(tf.shape(image)[1], dtype = tf.float32)
            
            #ymin_delta[0] to get the scalar value
            offset_height = tf.cast(ymin_delta[0]*h, dtype = tf.int32)
            offset_width = tf.cast(xmin_delta[0]*w, dtype = tf.int32)
            target_height = tf.cast(
                tf.math.ceil((1 - ymax_delta - ymin_delta)[0] * h),
                dtype=tf.int32)
            target_width = tf.cast(
                tf.math.ceil((1 - xmax_delta - xmin_delta)[0] * w),
                dtype=tf.int32)
            
            image = image[offset_height:offset_height+target_height, offset_width:offset_width+target_width, :]
            
        return image, bboxes
    
    def parse_y_features(self, features):
        classes = tf.sparse.to_dense(features['image/object/class/label'])
        classes = tf.one_hot(classes,self.num_classes)
        #classes shape = (# of detection or bbox, num_classes)
        
        #bboxes shape (None, 4)
        bboxes = tf.stack([
            tf.sparse.to_dense(features['image/object/bbox/xmin']),
            tf.sparse.to_dense(features['image/object/bbox/ymin']),
            tf.sparse.to_dense(features['image/object/bbox/xmax']),
            tf.sparse.to_dense(features['image/object/bbox/ymax']),
            ], axis=1)
        return classes, bboxes
    
    def preprocess_label_for_one_scale(self, classes, bboxes, grid_size, valid_anchors=None):
        '''
        Preprocess the class and bounding boxes annotations into model-desired format for each scale
        (grid, grid, anchor, (centroid_x, centroid_y, width, height, objectness, one-hot classes))

        Returns
        -------
        y: the desired label format to calculate loss

        '''
        y = tf.zeros((grid_size, grid_size, 3, 5 + self.num_classes))
        anchor_indices = self.find_best_anchor(bboxes)
        
        tf.Assert(classes.shape[0] == bboxes.shape[0], [classes])
        tf.Assert(anchor_indices.shape[0] == bboxes.shape[0], [anchor_indices])
        
        num_boxes = tf.shape(classes)[0]
        indices = tf.TensorArray(tf.int32, 1, dynamic_size=True)
        updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
        
        valid_count = 0
        for i in tf.range(num_boxes):
            curr_class = tf.cast(classes[i],tf.float32) #shape = # of classes
            curr_box = bboxes[i] # shape = (4,)
            curr_anchor = anchor_indices[i] #scalar number
            #OR operation
            anchor_found = tf.reduce_any(curr_anchor == valid_anchors)
            if anchor_found:
                #MOD operation (0 or 1 or 2)
                adjusted_anchor_index = tf.math.floormod(curr_anchor,3)
                
                #Converting (xmin, ymin, xmax, ymax) box format to
                #(centroid x, centroid y, width, height)
                curr_box_xy = (curr_box[...,0:2] + curr_box[...,2:4])/2
                curr_box_wh = curr_box[...,2:4] - curr_box[...,0:2]

                #Calculate which grid cell we should use
                #e.g., if curr_box_xy = [0.25,0.25] and grid_size = 26, the idx of grid cell is floor(0.25*26)=6
                grid_cell_xy = tf.cast(curr_box_xy//tf.cast((1/grid_size),dtype=tf.float32),tf.int32)
                
                #For this box, we need to update y at location (grid, grid, adjusted_anchor_idx)
                #grid[y][x][anchor] = (bx, by, bw, bh, obj, class)
                index = tf.stack(
                    [grid_cell_xy[1],grid_cell_xy[0],adjusted_anchor_index]
                    )
                update = tf.concat(
                    values = [curr_box_xy,curr_box_wh,tf.constant([1.0]),curr_class]
                    ,axis=0)
                
                indices = indices.write(valid_count,index) #idx, value
                updates = updates.write(valid_count,update)
   
                valid_count = 1 + valid_count
                
        y = tf.tensor_scatter_nd_update(y, indices.stack(), updates.stack())
        #returned shape = (grid, grid, anchors, 5(bx,by,bw,bh,obj) + # of class)
        return y
    
    def find_best_anchor(self, y_box):
        '''
        find the best anchor for ground truth boxes in y_box.
        Returns a tensor in shape of (num_boxes) that indicates the indices of best anchor for each box

        Parameters
        ----------
        y_box : ground truth boxes in shape of (num_boxes, 4)

        Returns
        -------
        anchor_idx: anchor indices in shape of (num_boxes)

        '''
        box_wh = y_box[...,2:4] - y_box[...,0:2] # max - min
        
        #since box_wh is (# boxes, 2) and anchor_wh is (9,2), we need to tile box_wh first
        #to match number of anchor in order to apply tf.minimum later
        #i.e., box_wh - > (# boxes,9,2)
        box_wh = tf.tile(
            tf.expand_dims(box_wh,-2),(1,tf.shape(anchors_wh)[0],1)
            )
        
        #The intersection here is not calculated based on real coordinates
        #but assuming an anchor and a boxe share the same centroid helps us decide 
        #which is the best fit anchor for this particular box
        #so we just take the product of minimum width and height as intersecion
        intersection = tf.minimum(box_wh[...,0],anchors_wh[...,0]) * tf.minimum(box_wh[...,1],anchors_wh[...,1])
        box_area = box_wh[...,0] * box_wh[...,1] #shape = (# boxes, 9)
        anchor_area = anchors_wh[...,0] * anchors_wh[...,1] #shape = (9,)
        iou = intersection / (box_area + anchor_area - intersection) #shape = (# boxes, 9)
        
        #shape = (# of boxes,)
        anchor_idx = tf.cast(tf.argmax(iou,axis=-1),tf.int32)
        
        return anchor_idx
        
    
    def parse_tfexample(self, example_proto):
        image_feature_description = {
            'image/height': tf.io.FixedLenFeature([],tf.int64),
            'image/width' : tf.io.FixedLenFeature([],tf.int64),
            'image/depth' : tf.io.FixedLenFeature([],tf.int64),
            'image/object/class/label' : tf.io.VarLenFeature(tf.int64),
            'image/object/bbox/xmin' : tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin' : tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax' : tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax' : tf.io.VarLenFeature(tf.float32),
            'image/encoded' : tf.io.FixedLenFeature([],tf.string),
            'image/filename' : tf.io.FixedLenFeature([],tf.string),
            }
        return tf.io.parse_single_example(example_proto, image_feature_description)
    
    