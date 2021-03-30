#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 11:51:31 2021

@author: djkim9031
"""


import argparse
import math
import datetime
import os
import time

import tensorflow as tf
import numpy as np

import sys
sys.path.append('/Users/djkim9031/Desktop/YOLOv3_Trainer/')

from yolov3 import YOLOv3, YOLOLoss, anchors_wh
from yolo_preprocess import Preprocessor
from darknet53_weight_loader import load_darknet_weights

BATCH_SIZE = 16
TOTAL_CLASSES = 20
TOTAL_EPOCHS = 300
OUTPUT_SHAPE = (416, 416)
TF_RECORDS = './dataset/tfrecords'
HUE = 0.1
SATURATION = 1.5
EXPOSURE = 1.5


tf.random.set_seed(42)

class Trainer(object):
    def __init__(self, model, initial_epoch, epochs, global_batch_size, strategy, last_val_loss, initial_learning_rate=1e-3):
        self.model = model
        self.initial_epoch = initial_epoch
        self.epochs = epochs
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.loss_objects = [
            YOLOLoss(num_classes = TOTAL_CLASSES, valid_anchors_wh = anchors_wh[6:9]), #small scale 52x52
            YOLOLoss(num_classes = TOTAL_CLASSES, valid_anchors_wh = anchors_wh[3:6]), #medium scale 26x26
            YOLOLoss(num_classes = TOTAL_CLASSES, valid_anchors_wh = anchors_wh[0:3]), #large scale 13x13
            ]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        
        #for learning rate schedule
        self.current_learnining_rate = initial_learning_rate
        self.last_val_loss = last_val_loss
        self.lowest_val_loss = last_val_loss
        self.patience_count = 0
        self.max_patience = 10
        
    def lr_decay(self):
        '''
        This effectively simulates ReduceOnPlateau learning rate schedule.
        Learning rate will be reduced by a factor of 10 if there's no improvement
        over [max_patience] epochs
        '''
        if self.patience_count > self.max_patience:
            self.current_learnining_rate /= 10.0
            self.patience_count = 0
        elif self.last_val_loss == self.lowest_val_loss:
            self.patience_count = 0
        self.patience_count += 1
        self.optimizer.learning_rate = self.current_learnining_rate

    
    def augmentation(self, image):
        dhue = np.random.uniform(-HUE,HUE)
        dsat = np.random.uniform(1,SATURATION)
        dexp = np.random.uniform(-EXPOSURE,EXPOSURE)

        g = tf.where(tf.math.abs(image[...,1]*dsat)>1,image[...,1],image[...,1]*dsat)
        b = tf.where(tf.math.abs(image[...,2]*dexp)>1,image[...,2],image[...,2]*dexp)
        r = tf.where(tf.math.abs(image[...,0]+dhue)>1,image[...,0],image[...,0]+dhue)
        augmented = tf.stack((r,g,b),axis=2)
        return augmented
        
    def train_step(self, inputs):
        images, labels = inputs
        images = tf.map_fn(self.augmentation,images)
        
        with tf.GradientTape() as tape:
            outputs = self.model(images,training=True)
            total_losses = []
            xy_losses = []
            wh_losses = []
            class_losses = []
            obj_losses = []
            
            #iterate over all three scales
            for loss_object, y_pred, y_true in zip(self.loss_objects, outputs, labels):
                total_loss, loss_breakdown = loss_object(y_true, y_pred)
                xy_loss,wh_loss,class_loss,obj_loss = loss_breakdown
                total_losses.append(total_loss * (1./self.global_batch_size))
                xy_losses.append(xy_loss * (1./self.global_batch_size))
                wh_losses.append(wh_loss * (1./self.global_batch_size))
                class_losses.append(class_loss * (1./self.global_batch_size))
                obj_losses.append(obj_loss * (1./self.global_batch_size))
            
            total_loss = tf.reduce_sum(total_losses)
            total_xy_loss = tf.reduce_sum(xy_losses)
            total_wh_loss = tf.reduce_sum(wh_losses)
            total_class_loss = tf.reduce_sum(class_losses)
            total_obj_loss = tf.reduce_sum(obj_losses)
        
        grads = tape.gradient(target = total_loss, sources = self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        
        return total_loss, (total_xy_loss, total_wh_loss, total_class_loss, total_obj_loss)
    
    def val_step(self, inputs):
        images, labels = inputs
        
        outputs = self.model(images, training = False)
        losses = []
        
        #iterate over all three scales
        for loss_object, y_pred, y_true in zip(self.loss_objects, outputs, labels):
            loss, _ = loss_object(y_true, y_pred)
            losses.append(loss * (1./self.global_batch_size))
        total_loss = tf.reduce_sum(losses)
        
        return total_loss
    
    def get_current_time(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    def run(self, train_dist_dataset, val_dist_dataset):
        total_steps = tf.constant(0, dtype=tf.int64)
        
        @tf.function
        def distributed_train_epoch(dataset, train_summary_writer, total_steps):
            total_loss = 0.0
            num_train_batches = tf.constant(0, dtype = tf.int64)
            for one_batch in dataset:
                per_replica_losses, per_replica_losses_breakdown = self.strategy.run(self.train_step, args=(one_batch,))
                per_replica_xy_losses, per_replica_wh_losses, per_replica_class_losses, per_replica_obj_losses = per_replica_losses_breakdown
                
                batch_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM,
                    per_replica_losses,
                    axis=None
                    )
                batch_xy_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM,
                    per_replica_xy_losses,
                    axis=None
                    )
                batch_wh_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM,
                    per_replica_wh_losses,
                    axis=None
                    )
                batch_obj_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM,
                    per_replica_obj_losses,
                    axis=None
                    )
                batch_class_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM,
                    per_replica_class_losses,
                    axis=None
                    )
                total_loss += batch_loss
                num_train_batches += 1
                tf.print('Trained Batch: ',num_train_batches, 'Batch Loss: ',batch_loss,
                         'Batch xy Loss: ',batch_xy_loss, 'Batch wh Loss: ',batch_wh_loss,
                         'Batch obj Loss: ',batch_obj_loss, 'Batch class Loss: ',batch_class_loss,
                         'Epoch Total Loss: ',total_loss)
                with train_summary_writer.as_default():
                    tf.summary.scalar(
                        'batch train loss',
                        batch_loss,
                        step=total_steps + num_train_batches)
                    tf.summary.scalar(
                        'batch xy loss',
                        batch_xy_loss,
                        step=total_steps + num_train_batches)
                    tf.summary.scalar(
                        'batch wh loss',
                        batch_wh_loss,
                        step=total_steps + num_train_batches)
                    tf.summary.scalar(
                        'batch obj loss',
                        batch_obj_loss,
                        step=total_steps + num_train_batches)
                    tf.summary.scalar(
                        'batch class loss',
                        batch_class_loss,
                        step=total_steps + num_train_batches)
            return total_loss, num_train_batches
        
        @tf.function
        def distributed_val_epoch(dataset):
            total_loss = 0.0
            num_val_batches = tf.constant(0, dtype = tf.int64)
            for one_batch in dataset:
                per_replica_losses = self.strategy.run(self.val_step, args=(one_batch,))
                batch_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM,
                    per_replica_losses,
                    axis=None
                    )
                total_loss += batch_loss
                num_val_batches += 1
            return total_loss, num_val_batches
        
        current_time = self.get_current_time()
        train_log_dir = 'logs/gradient_tape/train/'+current_time
        val_log_dir = 'logs/gradient_tape/val/'+current_time
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        
        tf.print('{} Start Training...'.format(current_time))
        for epoch in range(self.initial_epoch, self.epochs+1):
            t0 = time.time()
            self.lr_decay()
            
            tf.print(
                '{} Started Epoch {} with learning rate {}. Current LR patience count is {} epochs. Last lowest val loss is {}.'.format(self.get_current_time(),epoch,self.current_learnining_rate,self.patience_count,self.lowest_val_loss)
                )
            
            train_total_loss, num_train_batches = distributed_train_epoch(train_dist_dataset,train_summary_writer,total_steps)
            t1 = time.time()
            train_loss = train_total_loss / tf.cast(num_train_batches,dtype=tf.float32)
            tf.print(
                '{} Epoch {} train loss {}, total train batches {}, {} examples per second'.format(self.get_current_time(),epoch,train_loss,num_train_batches,tf.cast(num_train_batches,dtype=tf.float32)*self.global_batch_size/(t1-t0))
                )
            with train_summary_writer.as_default():
                tf.summary.scalar('epoch train loss', train_loss, step=epoch)
            total_steps += num_train_batches
            
            val_total_loss, num_val_batches = distributed_val_epoch(val_dist_dataset)
            t2 = time.time()
            val_loss = val_total_loss / tf.cast(num_val_batches,dtype=tf.float32)
            tf.print(
                '{} Epoch {} val loss {}, total val batches {}, {} examples per second'.format(self.get_current_time(), epoch, val_loss, num_val_batches, tf.cast(num_val_batches,dtype=tf.float32)*self.global_batch_size /(t2-t1))
                )
            with val_summary_writer.as_default():
                tf.summary.scalar('epoch val loss',val_loss, step = epoch)
            
            #save the current model when reaching a new lowest validation loss
            if val_loss<self.lowest_val_loss:
                self.save_model(epoch, val_loss)
                self.lowest_val_loss = val_loss
            self.last_val_loss = val_loss
        
        #Saving the final model after completion of training
        self.save_model(self.epochs, self.last_val_loss)
        print('{} Finished Training.'.format(self.get_current_time()))
        
    def save_model(self,epoch,loss):
        model_name = './models/model-v3.0.1-epoch-{}-loss-{:.4f}.tf'.format(epoch,loss)
        self.model.save_weights(model_name)
        print("Model {} saved".format(model_name))
        
def create_dataset(tfrecords, batch_size, is_train):
    preprocess = Preprocessor(is_train,TOTAL_CLASSES,OUTPUT_SHAPE)
    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    
    if is_train:
        dataset = dataset.shuffle(512)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=str,help='checkpoint file path')
    args = parser.parse_args()

    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = strategy.num_replicas_in_sync * BATCH_SIZE
    train_dataset = create_dataset('{}/train*'.format(TF_RECORDS),global_batch_size,is_train=True)
    val_dataset = create_dataset('{}/val*'.format(TF_RECORDS),global_batch_size,is_train=False)
    

    if not os.path.exists(os.path.join('./models')):
        os.makedirs(os.path.join('./models/'))
            
    with strategy.scope():
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
        model = YOLOv3(shape=(416,416,3),num_classes=TOTAL_CLASSES)
        model.summary()
        

        initial_epoch = 1
        if args.checkpoint:
            model.load_weights(args.checkpoint)
            initial_epoch = int(args.checkpoint.split('-')[-3])+1
            val_loss = float(args.checkpoint.split('-')[-1][:-3])
            print('Resume Training from checkpoint {} and epoch {}'.format(args.checkpoint, initial_epoch))
        else:
          print("Loading Imagenet pretrained darknet53.conv.74 weights for Darknet...")
          load_darknet_weights(model, "./darknet53.conv.74")
          print("Pretrained weights loaded for Darknet53")
          val_loss = math.inf
            
        trainer = Trainer(
            model = model,
            initial_epoch = initial_epoch,
            epochs = TOTAL_EPOCHS,
            global_batch_size = global_batch_size,
            strategy = strategy,
            last_val_loss = val_loss,
            )
        trainer.run(train_dist_dataset,val_dist_dataset)
       
        
if __name__ == '__main__':
    main()
                
                
                
        
        
        