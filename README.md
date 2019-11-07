# YOLOv3-Retrain-OpenSource-Project in Google Colab 

This prject is designed as a part of our Final Year project. This project is inspired from [this link](https://pylessons.com/YOLOv3-custom-training/) Training custom YOLO v3 object detector tutorial.

### YOLO v3 Introduction 

YOLOv3 is the third object detection algorithm in YOLO (You Only Look Once) family. It improved the accuracy with many tricks and is more capable of detecting small objects. Let's take a closer look at the improvements. 

### The algorithm

First, during training, YOLOv3 network is fed with input images to predict 3D tensors (which is the last feature map) corresponding to 3 scales, as shown in the middle one in the above diagram. The three scales are designed for detecting objects with various sizes. Here we take the scale 13x13 as an example. For this scale, the input image is divided into 13x13 grid cells, each grid cell corresponds to a 1x1x255 voxel inside a 3D tensor. Here, 255 comes from (3x(4+1+80)). Values in a 3D tensor such as bounding box coordinate, objectness score and class confidence are shown on the right of the diagram.

Second, if the center of the object's ground truth bounding box falls in a certain grid cell(i.e. the red one on the bird image),  this grid cell is responsible for predicting the object's bounding box. The corresponding objectness score is "1" for this grid cell and "0" for others. For each grid cell, it is assigned with 3 prior boxes of different sizes. What it learns during training is to choose the right box and calculate precise offset/coordinate. But how does the grid cell know which box to choose? There is a rule that it only chooses the box that overlaps ground truth bounding box most. 

Lastly, how to choose the initial size of those 3 prior boxes? The author uses K-mean clustering to classify the total bboxes from COCO dataset to 9 clusters before training. This results in 9 sizes chosen from 9 cluster, 3 for 3 scales. This prior information is helpful for the network to learn to compute box offset/coordinate precisely because intuitively, bad choice of box size make it harder and longer for the network to learn. 

![ ](/image.png)

### The Network Architecture

Here is a diagram of YOLOv3's network architecture. It is a feature-learning based network that adopts 75 convolutional layers as its most powerful tool. No fully-connected layer is used. This structure makes it possible to deal with images with any sizes. Also, no pooling layers are used. Instead, a convolutional layer with stride 2 is used to downsample the feature map, passing size-invariant feature forwardly.  In addition, a ResNet-alike structure and FPN-alike structure is also a key to its accuracy improvement. 


![ ](/image1.png)


### 3 Scales: handling objects of different sizes

There are objects of different sizes on the images. Some are big and some are small. It is desirable for the network to detect all of them. Therefore, the network needs to be capable of "seeing" objects that are of different sizes. As the network goes deeper, its feature map gets smaller. That is to say, the deeper it goes, the harder it is to detect smaller objects. Intuitively, it is better to detect the objects at different feature maps before small objects end up disappearing. As in SSD, object detection is carried out on different features maps in order catch various scales. (see our previous post about SSD) 

However, the features are not absolutely relevant  at different depth. What does this mean? Well, with network depth increasing, the features change from low-level features (edges, colors, rough 2D positions. etc) to high-level features (semantic-meaningful information: dog, cat, car. etc) with depth increasing. So making predictions on feature maps at different depth does sound it is able to do detection for multi-scale objects, but actually it is not as accurate as expected. 

In YOLOv3, this is improved by adopting an FPN-like structure. 

