# Multi-Object-Classification-using-CNN-YOLO

Object Detection and Classification while comparing the Efficiency and Effectivenesss of the multilayered convolutional network against the You Only Look Once (YOLO) Algorithm.
Task 1: Gathering Database
Objectives:
- Collect and organize an image database for object recognition. - Segment the database into train/dev/test sets.
Description of the Task:
- Choose a source for the image database (VOC2007)
- Downloaded and organized the database with labels/classes. - Divided the dataset into Training,Test subsets
Task 2: Train a CNN for Object Recognition
Objectives:
- Designing and creating a multi layered convolutional neural network
for object detection and classification
Description of the Task:
The architecture of the designed network is a two layered network of tensor size ( 38*28*28). The model was trained and tested against 3 classes of the VOC2007 Database. The Code was executed by creating a python notebook in stable virtual environment on visual studio.
Experimental Outcome:
The CNN predictions are outlined in the following Confution Matrix
 
 This Tells us that the model is over fitted slightly to detect Bikes, Giving better accuracy to detect wheeled objects over person.

Task 3 : Evaluate Data set using YOLO
Objectives:
To Understand the architecture and implementiation of the You Only Look Once Model
Description of the Task:
Imported and evaluated the YOLO version 8 Model from ultralytics. The pretrained model vas validate against the previous Dataset and its accuracy was Deterined. Due to issue with implementing Ultralytics on Mac M2 Chip . The code was executed using Google COlab
Experimental Outcome :
The pre-trained YOLO model gives almost perfect accuracy while evaluating against the preprocessed VOC2007 dataset.
References :
1) https://github.com/ultralytics/ultralytics
2) https://www.kaggle.com/datasets/zaraks/pascal-voc-2007
3) https://www.kaggle.com/code/saweraa/object-detection-and-
classification

