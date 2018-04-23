# Object detection with YOLO Algorithm

The YOLO (You only look once) algorithm is a very good algorithm for object detection because it gives us high accuracy while being run at real time. The reason for it being so computationally fast is that it divides the image into smaller parts (usually 19x19 small boxes) and searches for an object in that part. 

![grid 2x](https://user-images.githubusercontent.com/30028589/39106096-7952bd8c-4687-11e8-9cef-dda2ce9f4774.png)


## Filtering Boxes
The network will output atleast one bounding box for each small square. We then need to filter out boxes which have a very low probability of detecting an object.

![low prob](https://user-images.githubusercontent.com/30028589/39106201-1ef1bba8-4688-11e8-8f98-9a9703ee0362.png)


## Non-Max Suppression and Intersection over Union
Sometimes a network can output more than one bounding box for the same object. We can check which bounding box is the best by performing Non-max Suppression. Non Max Suppression uses special function called Intersection over Union or IoU.

IoU is, literally what it says it is, the ratio of intersection of area to the union of area of two boxes.

<img width="667" alt="iou" src="https://user-images.githubusercontent.com/30028589/39106740-bfa976be-468b-11e8-8bd4-b80f84154a37.png">

The bounding box is retained if the IoU > 0.6. All other bounding Boxes are removed.

After performing IoU, all boxes, except the box with the highest probability of detecting the object, are removed. The bounding box which is remaining is the bounding box for the object of a specific class detected in that image. 

## Project Details

For this code, I have divided the image into 19 x 19 grid cells. Each cell predicts 5 bounding boxes for any class detected.
The dataset consists of 80 classes, Thus the network can predict 80 classes. Each bounding box will need to have following details:

1. Probability of the class detected
2. X and Y co-ordinates of the center of the bounding box.
3. dimensions of the bounding box.
4. 80-Dimensional Vector in which the index of the class which is detected will have value 1 and all others will have the value 0. 

So the output Y of our network will be a 4-D Tensor of shape (19x19x5x85).

This output vector will then go through Box filtering and Non-Max Suppression to output the final detected objects.

Here is the output of the Network on a few images.

![low prob1](https://user-images.githubusercontent.com/30028589/39107922-63c28504-4693-11e8-87e7-5abc5e1b0414.png)

![low prob1](https://user-images.githubusercontent.com/30028589/39108310-98c68d84-4695-11e8-90e4-e392aa7901e3.png)
