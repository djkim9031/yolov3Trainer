# yolov3Trainer


Personal full implementation of YOLOv3 for object detection.

# For training:
1. First download train/validation set images and annotate them
(For this project, I used VOC 2007 oxford dataset)
2. Create TFDataset by running the file(tfrecords_json.py) under dataset folder.
(In the case of VOC2007, since the annotation should be parsed from xml, and image directories are stored in .txt, use tfrecords.py)
3. Then train the model using train.py!
4. The train total loss gets under 10 around 90+ epochs with the VOC 2007 dataset, and provides the promising result (However, a better optimization scheme should come into place in the future)
- Training from scratch: Darknet starts with the "darknet53.conv.74" weights
- Training from saved weights: provide the checkpoint with the weight file:
e.g.,) !python train.py --checkpoint='./models/model-v5.0.1-epoch-130-loss-6.5867.tf'


# For inference:
Refer to Inference_Test.ipynb file!
