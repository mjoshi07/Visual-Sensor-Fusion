import numpy as np
import cv2
import struct
import matplotlib.pyplot as plt
import open3d as o3d
import Utils as ut


def project_to_image(l2c_object, pts_3d):
    """ Project 3d points to image plane.
    """
    # Convert to Homogeneous Coordinates
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # Multiply with the P Matrix
    pts_2d = np.dot(pts_3d_extend, np.transpose(l2c_object.P))  # nx3
    # Convert Back to Cartesian
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    return pts_2d[:, 0:2]


def compute_box_3d(l2c_object, obj):
    """ Projects the 3d bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = ut.get_rotation_mat(obj.ry)
    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    # multiply the rotation matrix with the points
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))

    # perform translation in all axis
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]

    # only draw 3d bounding box for objs in front of the camera
    # corners_3d[2, :] = z (distance) value from the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(l2c_object, np.transpose(corners_3d))

    return corners_2d


def get_image_with_bboxes(l2c_object, img, objects):
    img2 = img.copy()
    img3 = img.copy()
    for obj in objects:
        boxes = compute_box_3d(l2c_object, obj)
        if boxes is not None:
            obj.bbox3d = boxes
            obj.bbox2d = ut.convert_3DBox_to_2DBox(boxes)
            img2 = ut.draw_projected_box2d(img2, obj.bbox2d) # Draw the 2D Bounding Box
            img3 = ut.draw_projected_box3d(img3, obj.bbox3d) # Draw the 3D Bounding Box
    return img2, img3


def convert_3D_to_2D(l2c_object, pts_3d_velo, print_values=False):
    '''
    Input: 3D points in Velodyne Frame [nx3]
    Output: 2D Pixels in Image Frame [nx2]
    '''
    R0_homo = np.vstack((l2c_object.R0, [0,0,0]))
    if print_values:
        print("R0_homo ", R0_homo)

    R0_homo_2 = np.hstack((R0_homo, [[0],[0],[0],[1]]))
    if print_values:
        print("R0_homo_2 ", R0_homo_2)

    P_R0 = np.dot(l2c_object.P, R0_homo_2)
    if print_values:
        print("P_R0 ", P_R0)

    P_R0_Rt = np.dot(P_R0, np.vstack((l2c_object.V2C, [0, 0, 0, 1])))
    if print_values:
        print("P_R0_Rt ", P_R0_Rt)

    pts_3d_homo = np.hstack((pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1))))
    if print_values:
        print("pts_3d_homo ", pts_3d_homo)

    P_R0_Rt_X = np.dot(P_R0_Rt, pts_3d_homo.T)
    if print_values:
        print("P_R0_Rt_X ", P_R0_Rt_X)

    pts_2d_homo = P_R0_Rt_X.T
    if print_values:
        print("pts_2d_homo ", pts_2d_homo)

    pts_2d_homo[:, 0] /= pts_2d_homo[:, 2]
    pts_2d_homo[:, 1] /= pts_2d_homo[:, 2]

    pts_2d = pts_2d_homo[:, :2]

    if print_values:
        print("pts_2d ", pts_2d)

    return pts_2d


def remove_lidar_points_beyond_img(l2c_object, pts_3d, xmin, ymin, xmax, ymax, clip_distance=2.0):
    """ Filter lidar points, keep only those which lie inside image """
    pts_2d = convert_3D_to_2D(l2c_object, pts_3d)
    inside_pts_indices = ((pts_2d[:, 0] >= xmin) & (pts_2d[:, 0] < xmax) & (pts_2d[:, 1] >= ymin) & (pts_2d[:, 1] < ymax))

    # pc_velo are the points in LiDAR frame
    # therefore x-axis is in forward direction
    # we want to keep objects that are at least clip_distance away from sensor
    # X points are at 0th index column
    inside_pts_indices = inside_pts_indices & (pts_3d[:, 0] > clip_distance)
    pts_3d_inside_img = pts_3d[inside_pts_indices, :]

    return pts_3d_inside_img, pts_2d, inside_pts_indices


def get_lidar_on_image(l2c_object, pc_velo, size):
    """ Project LiDAR points to image """
    imgfov_pc_velo, all_pts_2d, fov_inds = remove_lidar_points_beyond_img(l2c_object,
        pc_velo, 0, 0, size[0], size[1], 1.5
    )

    return imgfov_pc_velo, all_pts_2d[fov_inds, :]


def display_lidar_on_image(l2c_object, pc_velo, image):
    img = image.copy()
    imgfov_pc_velo, pts_2d = get_lidar_on_image(l2c_object, pc_velo, (img.shape[1], img.shape[0]))

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(pts_2d.shape[0]):
        depth = imgfov_pc_velo[i, 0]

        # if depth is 2, then 510 / 2 will be  255, white color
        # 510 has been calculated according to the clip distance, usually clip distance is 2
        # therefore we get depth color in the range (0, 255)
        color = cmap[int(510.0 / depth), :]
        pt = (int(np.round(pts_2d[i, 0])), int(np.round(pts_2d[i, 1])))
        cv2.circle(img, pt, 2, color=tuple(color), thickness=-1)

    return img


def convert_bin_to_pcd(point_files):
    list_pcd = []
    size_float = 4

    for i in range(len(point_files)):
        file_to_open = point_files[i]
        file_to_save = str(point_files[i])[:-3] + "pcd"
        with open(file_to_open, "rb") as f:
            byte = f.read(size_float * 4)
            while byte:
                x, y, z, intensity = struct.unpack("ffff", byte)
                list_pcd.append([x, y, z])
                byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd)
    o3d.io.write_point_cloud(file_to_save, pcd)
