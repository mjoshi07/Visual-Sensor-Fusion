import cv2
import os
import random
import glob
import numpy as np
import open3d as o3d
import Utils as ut
import Lidar2Camera as l2c
import LidarUtils as lu
import YoloDetector as yd
import YoloUtils as yu
import FusionUtils as fu


def low_level_fusion(data_dir, show_random_pcl=False, display_video=True,  save_video=False):
    imgs, pts, lbls, calbs = ut.load_data(data_dir)

    if save_video:
        out_dir = os.path.join(data_dir, "output//videos")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    if show_random_pcl:
        idx = random.randint(0, len(pts) - 1)
        pcd = o3d.io.read_point_cloud(pts[idx])
        o3d.visualization.draw_geometries([pcd])

    lidar2cam = l2c.LiDAR2Camera(calbs[0])
    print("P :"+str(lidar2cam.P))
    print("-")
    print("RO "+str(lidar2cam.R0))
    print("-")
    print("Velo 2 Cam " +str(lidar2cam.V2C))

    video_images = sorted(glob.glob(data_dir+"//test//video4//images/*.png"))
    video_points = sorted(glob.glob(data_dir+"//test//video4//points/*.pcd"))

    result_video = []

    weights = data_dir + "//model//yolov4//yolov4.weights"
    config = data_dir + "//model//yolov4//yolov4.cfg"
    names = data_dir + "//model//yolov4//coco.names"

    detector = yd.Detector(0.4)
    detector.load_model(weights, config, names)

    image = cv2.imread(video_images[0])

    if display_video:
        cv2.namedWindow("fused_result", cv2.WINDOW_KEEPRATIO)

    for idx, img_path in enumerate(video_images):
        image = cv2.imread(img_path)
        detections, image = detector.detect(image, True, True)
        point_cloud = np.asarray(o3d.io.read_point_cloud(video_points[idx]).points)
        # image = lu.display_lidar_on_image(lidar2cam, point_cloud, image)
        pts_3D, pts_2D = lu.get_lidar_on_image(lidar2cam, point_cloud, (image.shape[1], image.shape[0]))
        image, _ = fu.lidar_camera_fusion(pts_3D, pts_2D, detections, image)
        if display_video:
            cv2.imshow("fused_result", image)
            cv2.waitKey(10)
        if save_video:
            result_video.append(image)

    if save_video:
        out = cv2.VideoWriter(os.path.join(out_dir, "fused_result.avi"), cv2.VideoWriter_fourcc(*'DIVX'), 30, (image.shape[1], image.shape[0]))

        for i in range(len(result_video)):
            # out.write(cv2.cvtColor(result_video[i], cv2.COLOR_BGR2RGB))
            out.write(result_video[i])
        out.release()


def mid_level_fusion(data_dir, index=0, display_image=True, save_image=False):
    imgs, pts, labels, calibs = ut.load_data(data_dir)

    if save_image:
        out_dir = os.path.join(data_dir, "output//images")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    weights = data_dir + "//model//yolov4//yolov4.weights"
    config = data_dir + "//model//yolov4//yolov4.cfg"
    names = data_dir + "//model//yolov4//coco.names"

    detector = yd.Detector(0.4)
    detector.load_model(weights, config, names)

    if display_image:
        cv2.namedWindow("fused_result", cv2.WINDOW_KEEPRATIO)

    """PIPELINE STARTS FROM HERE"""

    # load the image
    image = cv2.imread(imgs[index])

    # create LiDAR2Camera object
    lidar2cam = l2c.LiDAR2Camera(calibs[index])

    # 1 - Run 2D object detection on image
    detections, yolo_detections = detector.detect(image, draw_bboxes=False, display_labels=False)

    # load lidar points and project them inside 2d detection
    point_cloud = np.asarray(o3d.io.read_point_cloud(pts[index]).points)
    pts_3D, pts_2D = lu.get_lidar_on_image(lidar2cam, point_cloud, (image.shape[1], image.shape[0]))
    lidar_pts_img, _ = fu.lidar_camera_fusion(pts_3D, pts_2D, detections, image)

    # Build a 2D Object
    list_of_2d_objects = ut.fill_2D_obstacles(detections)

    # Build a 3D Object (from labels)
    list_of_3d_objects = ut.read_label(labels[index])

    # Get the LiDAR Boxes in the Image in 2D and 3D
    lidar_2d, lidar_3d = lu.get_image_with_bboxes(lidar2cam, lidar_pts_img, list_of_3d_objects)

    # Associate the LiDAR boxes and the Camera Boxes
    lidar_boxes = [obs.bbox2d for obs in list_of_3d_objects]  # Simply get the boxes
    camera_boxes = [np.array([box[0], box[1], box[0] + box[2], box[1]+box[3]]) for box in detections[:, 1]]
    # camera_boxes = [obs.bbox for obs in list_of_2d_objects]
    matches, unmatched_lidar_boxes, unmatched_camera_boxes = fu.associate(lidar_boxes, camera_boxes)

    # Build a Fused Object
    final_image, _ = fu.build_fused_object(list_of_2d_objects, list_of_3d_objects, matches, lidar_2d)

    # draw yolo detections on top to fused results
    final_image = yu.draw_yolo_detections(final_image, detections)

    if display_image:
        cv2.imshow("lidar_3d", lidar_3d)
        cv2.imshow("lidar_2d", lidar_2d)
        cv2.imshow("yolo_detections", yolo_detections)
        cv2.imshow("lidar_pts_img", lidar_pts_img)
        cv2.imshow("final_image", final_image)
        cv2.waitKey(0)
    if save_image:
        cv2.imwrite(os.path.join(out_dir,"fused_result.png"), final_image)

    return final_image


if __name__ == "__main__":
    data_dir = "..//Data//"
    # low_level_fusion(data_dir, show_random_pcl=False, display_video=True, save_video=False)
    mid_level_fusion(data_dir, index=0, display_image=True, save_image=False)