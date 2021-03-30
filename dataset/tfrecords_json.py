#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 17:47:47 2021

@author: djkim9031
"""


import numpy as np
import tensorflow as tf
import os
import json


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray



num_train_shards = 1
num_val_shards = 1
num_test_shards = 1
ray.init()
tf.get_logger().setLevel('ERROR')

def chunkify(l,n):
    size = len(l) // n
    start = 0
    results = []
    for i in range(n-1):
        results.append(l[start:start+size])
        start += size
    results.append(l[start:])
    return results

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))): #EagerTensor
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) #Takes the bytelist (string) value

def generate_tfexample(anno):
    with open(anno['filepath'],'rb') as image_file:
        content = image_file.read()
    width = anno['width']
    height = anno['height']
    depth = anno['depth']
    if depth != 3:
        print('WARNING: Image {} has depth of {}'.format(anno['filename'],depth))
    class_ids = []
    class_texts = []
    bbox_xmins = []
    bbox_ymins = []
    bbox_xmaxes = []
    bbox_ymaxes = []
    for bbox in anno['bboxes']:
        class_ids.append(bbox['class_id'])
        class_texts.append(bbox['class_text'].encode())
        xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = float(xmin)/width, float(ymin)/height, float(xmax)/width, float(ymax)/height
        
        #Sanity Check
        if bbox_xmin<=bbox_xmax:
            temp_x_min = bbox_xmin
            temp_x_max = bbox_xmax
        else:
            temp_x_min = bbox_xmax
            temp_x_max = bbox_xmin
        if bbox_ymin<=bbox_ymax:
            temp_y_min = bbox_ymin
            temp_y_max = bbox_ymax 
        else:
            temp_y_min = bbox_ymax
            temp_y_max = bbox_ymin
            
            

        assert bbox_xmin <= 1 and bbox_xmin >= 0
        assert bbox_ymin <= 1 and bbox_ymin >= 0
        assert bbox_xmax <= 1 and bbox_xmax >= 0
        assert bbox_ymax <= 1 and bbox_ymax >= 0
        
        bbox_xmins.append(temp_x_min)
        bbox_ymins.append(temp_y_min)
        bbox_xmaxes.append(temp_x_max)
        bbox_ymaxes.append(temp_y_max)
    
    feature = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width' : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/depth' : tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
        'image/object/bbox/xmin' : tf.train.Feature(float_list=tf.train.FloatList(value=bbox_xmins)),
        'image/object/bbox/ymin' : tf.train.Feature(float_list=tf.train.FloatList(value=bbox_ymins)),
        'image/object/bbox/xmax' : tf.train.Feature(float_list=tf.train.FloatList(value=bbox_xmaxes)),
        'image/object/bbox/ymax' : tf.train.Feature(float_list=tf.train.FloatList(value=bbox_ymaxes)),
        'image/object/class/label' : tf.train.Feature(int64_list = tf.train.Int64List(value=class_ids)),
        'image/object/class/text' : tf.train.Feature(bytes_list = tf.train.BytesList(value=class_texts)),
        'image/encoded' : _bytes_feature(content),
        'image/filename' : _bytes_feature(anno['filename'].encode()),
        }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

@ray.remote
def build_single_tfrecord(chunk, path):
    print('Start to build TF records for ' + path)
    
    with tf.io.TFRecordWriter(path) as writer:
        for anno in chunk:
            tf_example = generate_tfexample(anno)
            writer.write(tf_example.SerializeToString())
    
    print('Finished building TF records for ' + path)

def build_tf_records(annotations, total_shards, split):
    chunks = chunkify(annotations,total_shards)
    futures = [
        #train_0001_of_0064.tfrecords
        build_single_tfrecord.remote(chunk, './tfrecords/Hand/{}_{}_of_{}.tfrecords'.format(split,str(i+1).zfill(4),str(total_shards).zfill(4))) for i,chunk in enumerate(chunks)
        ]
    ray.get(futures)
    
                    
    
def parse_one_json(name):
    json_file = './Hand/Annotations/'+name+'.json'
    jpg = './Hand/'+name+'.jpg'
    
    with open(json_file) as f:
        d = json.load(f)

    width = d['imageWidth']
    height = d['imageHeight']
    depth = 3
    
    bboxes = []

    class_text = 'Hand'
    bboxes.append({
        'class_text' : class_text,
        'class_id' : 0,
        'xmin': d['shapes'][0]['points'][0][0],
        'ymin': d['shapes'][0]['points'][0][1],
        'xmax': d['shapes'][0]['points'][1][0],
        'ymax': d['shapes'][0]['points'][1][1],
        })
    
    
    return {
        'filepath' : jpg,
        'filename': name,
        'width': width,
        'height': height,
        'depth': depth,
        'bboxes': bboxes,
        }
    


def main():
    print("Start Parsing Annotations...")
    if not os.path.exists('./tfrecords/Hand'):
        os.makedirs('./tfrecords/Hand')
    
    # (name_xx.jpg) ->Train xx==1~200, Val xx ==201~220
    train = []
    val = []
    for name in os.listdir('./Hand'):
        if name.endswith('.jpg'):
            if name.split('_')[1].startswith('2') and len(name.split('_')[1])==7 and not name.split('_')[1].startswith('200'): 
                val.append(name.split('.')[0])
            else:
                train.append(name.split('.')[0])
    
    names = train + val
    
    
    train_annotations = []
    val_annotations = []
    
    for name in names:
        if name in train:
            train_annotations.append(parse_one_json(name))
        elif name in val:
            val_annotations.append(parse_one_json(name))

    #eturn train_annotations, val_annotations
    print('Start Building TF Records...')
    
    build_tf_records(train_annotations,num_train_shards,'train')
    build_tf_records(val_annotations,num_val_shards,'val')

    
    print('Successfully wrote {} annotations to TF Records.'.format(len(train_annotations) + len(val_annotations)))
    
if __name__ == '__main__':
   main()

