# Visual-Fusion
* LiDAR Fusion with Vision
* Data taken from [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) Dataset
* Download Yolov4 model weights from [here](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)

## Low-Level Fusion
### Yolo Detections
<p align="center">
<img src="https://github.com/mjoshi07/Visual-Sensor-Fusion/blob/main/Data/output/videos/out4_yolo_small.gif"/>
</p>

### Lidar Points Projection
<p align="center">
<img src="https://github.com/mjoshi07/Visual-Sensor-Fusion/blob/main/Data/output/videos/out4_lidar_small.gif"/>
</p>

### Visualizing LiDAR points in 3D
<p align="center">
<img src="https://github.com/mjoshi07/Visual-Sensor-Fusion/blob/main/Data/output/videos/visualizing_lildar.gif"/>
</p>

### LiDAR points Fused with YOLO detections 
<p align="center">
<img src="https://github.com/mjoshi07/Visual-Sensor-Fusion/blob/main/Data/output/videos/out4_fused_small.gif"/>
</p>

* LiDAR points are projected on the image using calibration matrix and transformation matrix
* The points that lie within the detected 2D Bounding Box by YOLO are stored and rest are ignored
* Some points that do not belong to an object might also be considered since we used only (x, y) coordinates of the projected LiDAR points.
* One way to resolve this is to shrink the bounding box size so that the points that absolutely belong to the desired objects are only considered.
* Another way is to use the Sigma Rule, i.e include the points that are within 1 sigma or 2 sigma away from gaussian mean

## Mid-Level Fusion
### Yolo Detections
<p align="center">
<img src="https://github.com/mjoshi07/Visual-Sensor-Fusion/blob/main/Data/output/images/yolo_detections.png" width=640/>
</p>

### LiDAR Points projected on Image
<p align="center">
<img src="https://github.com/mjoshi07/Visual-Sensor-Fusion/blob/main/Data/output/images/all_lidar_pts.png" width=640/>
</p>

### 3D Bounding Boxes From LiDAR
<p align="center">
<img src="https://github.com/mjoshi07/Visual-Sensor-Fusion/blob/main/Data/output/images/lidar_3d.png" width=640/>
</p>

### 3D BBox converted to 2D BBox
<p align="center">
<img src="https://github.com/mjoshi07/Visual-Sensor-Fusion/blob/main/Data/output/images/lidar_2d.png" width=640/>
</p>

### LiDAR 2D BBox Fused with YOLO 2D BBox using Intersection Over Union
<p align="center">
<img src="https://github.com/mjoshi07/Visual-Sensor-Fusion/blob/main/Data/output/images/final_fused_with_lidar_pts.png" width=640/>
</p>

* 2D Bboxes from LiDAR are associated with YOLO 2D Bboxes using [Hungarian](https://en.wikipedia.org/wiki/Hungarian_algorithm) Algorithm
* Green Bounding Boxes are detected by YOlO whereas Blue Bounding Boxes are calculated using LiDAR points
* YOLO missed 1 vehicle, whereas 2 vehicles are missed by LiDAR, one of which is half out of frame, at the bottom right side

### TODO
- [ ] Add Run Instructions
- [ ] Add Dependencies
- [ ] Add References
