#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:22:32 2021

@author: djkim9031
"""


import tensorflow as tf
import sys
sys.path.append('/Users/djkim9031/Desktop/YOLOv3_Trainer/')

from yolo_utils import broadcast_iou, xywh_to_x1x2y1y2

class Postprocessor(object):
    def __init__(self, iou_thresh, score_thresh, max_detection=100):
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.max_detection = max_detection
        
    def __call__(self, raw_yolo_outputs):
        #shapes = (batch, grid, grid, anchor, 4), (batch, grid, grid, anchor, 1), (batch, grid, grid, anchor, # of classes) 
        boxes, objectness, class_probs = [], [], []
        
        
        for o in raw_yolo_outputs:
            # o = (bboxes, objectness, class_prob) for each of three outputs(small, medium, large)
            # o[0] = bboxes, o[1] = objectness, o[2] = class_prob
            batch_size = tf.shape(o[0])[0]
            num_classes = tf.shape(o[2])[-1]
            boxes.append(tf.reshape(o[0],(batch_size,-1,4))) #batch,grid*grid*anchor,4
            objectness.append(tf.reshape(o[1],(batch_size,-1,1))) #batch,grid*grid*anchor,1
            class_probs.append(tf.reshape(o[2],(batch_size,-1,num_classes))) #batch,grid*grid*anchor,# of classes
            
        boxes = xywh_to_x1x2y1y2(tf.concat(boxes,axis=1)) #shape=(batch, grid*grid*anchors*3,4)
        objectness = tf.concat(objectness,axis=1) #shape=(batch, grid*grid*anchors*3,1)
        class_probs = tf.concat(class_probs,axis=1) #shape=(batch, grid*grid*anchors*3,# of classes)
        
        
        scores = objectness
        scores = tf.reshape(scores, (tf.shape(scores)[0],-1,tf.shape(scores)[-1])) #Ensuring the shape of (batch, grid*grid*anchors*3,# of classes) might be unnecessary
        final_boxes, final_scores, final_classes, valid_detections = self.batch_non_maximum_suppression(boxes, scores, class_probs, self.iou_thresh, self.score_thresh,self.max_detection)
        return final_boxes, final_scores, final_classes, valid_detections
        
    @staticmethod
    def batch_non_maximum_suppression(boxes, scores, classes, iou_threshold, score_threshold, max_detection):
        def single_batch_nms(candidate_boxes):
            #filter out predictions with score less than score_threshold
            #candidate_boxes[...,0:4] x1x2y1y2, candidate_boxes[...,4] score, candidate_boxes[...,4:] classes
            #shape = (# of grids out of total(=grid*grid*anchor*3) that contains obj_score>threshold, 4+1+#of classes)
            candidate_boxes = tf.boolean_mask(candidate_boxes,candidate_boxes[...,4]>=score_threshold)
            outputs = tf.zeros((max_detection+1,tf.shape(candidate_boxes)[-1])) #shape = (max_det+1 , 4+1+#of classes)
            indices = []
            updates = []
            count = 0
          
            
            #Keep running this until there's no more candidate box or until count<max_detection
            while candidate_boxes.shape[0] > 0 and count<max_detection:
                #pick the box with the highest score
                best_idx = tf.math.argmax(candidate_boxes[...,4],axis=0)
                best_box = candidate_boxes[best_idx] #shape = (4+1+#of classes)
                indices.append([count])
                updates.append(best_box)
                count+=1
                
                #remove this box from candidate boxes
                candidate_boxes = tf.concat([
                    candidate_boxes[0:best_idx],
                    candidate_boxes[best_idx+1:tf.shape(candidate_boxes)[0]]
                    ],axis=0)
                
                #calculate IOU between this box and all remaining candidate boxes
                iou = broadcast_iou(best_box[0:4], candidate_boxes[...,0:4]) #shape = (1, # of candidate boxes remaining) = iou for each remaining boxes

                #remove all candidate boxes with IOU bigger than iou_threshold
                candidate_boxes = tf.boolean_mask(candidate_boxes,iou[0]<=iou_threshold) #shape = (# of remaininng boxes that satisfy the condition, 4+1+#of classes)

            #afterwards, tf.shape(candidate_boxes) = (# of remaining boxes, 4+1+# of classes), OR (0, 4+1+#of classes)

            if count > 0:
                #also append num_detection to the result
                count_index = [[max_detection]]
                count_updates = [
                    tf.fill([tf.shape(candidate_boxes)[-1]],count)
                    ] #list of tensors with shape (4+1+# of classes, each filled with count number)
                indices = tf.concat([indices,count_index],axis=0) #shape (count + 1 (max_num),1)
                updates = tf.concat([updates,count_updates],axis=0)#shape (count +1 (1 for tensors containing count for all 4+1+#of classes) , 4+1+#of classes)
                outputs = tf.tensor_scatter_nd_update(outputs, indices, updates) #outputs shape = (max_det+1, 4+1+#of classes = N). Each count is corresponding to index of this tensor, and at each index containing N values.
                #If count = 3, outputs[0], outputs[1], outputs[2] each contains 7 non-zero values. The last outputs[100] = 7 count values. The rest is 0.
                
            return outputs
        
        combined_boxes = tf.concat([boxes,scores,classes],axis=2) #shape =(batch, grid*grid*anchors*3, 4+1+#classes)
        #Note that tf.map_fn takes a single batch (first tensor element)
        result = tf.map_fn(single_batch_nms,combined_boxes) #shape = (batch, max_detection+1, 4+1+#classes)

        #take out num_detection(count value) from the result
        valid_counts = tf.expand_dims(tf.map_fn(lambda x: x[max_detection][0],result),axis=-1)
        final_result = tf.map_fn(lambda x: x[0:max_detection],result) #Also unnecessary, essentially final_result = result
        nms_boxes,nms_scores,nms_classes = tf.split(final_result,[4,1,-1],axis=-1)
        return nms_boxes,nms_scores,nms_classes,tf.cast(valid_counts,tf.int32)

                
                