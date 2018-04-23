# Object detection with YOLO Algorithm

The YOLO (You only look once) algorithm is a very good algorithm for object detection because it gives us high accuracy while being run at real time. The reason for it being so computationally fast is that it divides the image into smaller parts (usually 19x19 small boxes) and searches for an object in that part. 

The network will output atleast one bounding box for each small square. We then need to filter out boxes with a very low probability of detecting an object. 





Sometimes a network can output more than one bounding box for the same object 
