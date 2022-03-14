import numpy as np


class FusedObject(object):
    def __init__(self, bbox2d, bbox3d, category, t, confidence):
        self.bbox2d = bbox2d
        self.bbox3d = bbox3d
        self.category = category
        self.confidence = confidence
        self.t = t
        with open("..//Data//model//yolov4/coco.names", 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        self.class_ = classes[category]


class Object2D(object):
    def __init__(self, box):
        self.category = int(box[0])
        self.confidence = box[2]
        self.xmin = int(box[1][0])
        self.ymin = int(box[1][1])
        self.xmax = int(box[1][0] + box[1][2])
        self.ymax = int(box[1][1] + box[1][3])
        self.bbox = np.array([self.xmin, self.ymin, self.xmax, self.ymax])


class Object3D(object):
    """ 3d object label """
    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.bbox2d = np.zeros(shape=(2,2))
        self.bbox3d = np.zeros(shape=(4,2))
