import glob
import cv2
import numpy as np
import Fusion as fu


def load_data(data_dir):
    image_files = sorted(glob.glob(data_dir+"//images//*.png"))
    point_files = sorted(glob.glob(data_dir+"//points//*.pcd"))
    label_files = sorted(glob.glob(data_dir+"//labels//*.txt"))
    calib_files = sorted(glob.glob(data_dir+"//calibs//*.txt"))

    return image_files, point_files, label_files, calib_files


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [fu.Object3D(line) for line in lines if line.split(" ")[0]!="DontCare"]
    return objects


def fill_2D_obstacles(detections):
    return [fu.Object2D(box) for box in detections]


def get_rotation_mat(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    # rotation matrix about Y-axis
    # since in camera frame vertical is Y-axis and yaw angle in about the vertical axis
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def convert_3d_to_homo(pts_3d):
    """ Input: nx3 points in Cartesian
        Output: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def convert_3DBox_to_2DBox(pts_2d):
    x0 = np.min(pts_2d[:, 0])
    x1 = np.max(pts_2d[:, 0])
    y0 = np.min(pts_2d[:, 1])
    y1 = np.max(pts_2d[:, 1])
    x0 = max(0, x0)
    y0 = max(0, y0)
    return np.array([x0, y0, x1, y1])


def get_iou(box1, box2):
    """
    Computer Intersection Over Union
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = (box1_area + box2_area) - inter_area

    # compute the IoU
    iou = inter_area/float(union_area)
    return iou


def draw_projected_box2d(image, qs, color=(255,0,0), thickness=2):
    return cv2.rectangle(image, (int(qs[0]), int(qs[1])), (int(qs[2]), int(qs[3])), color, thickness)


def draw_projected_box3d(image, qs, color=(255, 0, 0), thickness=2):
    """
    Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
         1 --------- 0
        /|          /|
        2 -------- 3 |
        | |        | |
        | 5 -------| 4
        |/         |/
        6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

    return image














