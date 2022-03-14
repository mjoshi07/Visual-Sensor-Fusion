import utils as ut
import numpy as np
import cv2
import open3d as o3d
import random
import glob
import classes as cl


def low_level_fusion(data_dir, show_random_pcl=False):
    imgs, pts, lbls, calbs = ut.load_data(data_dir)

    if show_random_pcl:
        idx = random.randint(0, len(pts) - 1)
        pcd = o3d.io.read_point_cloud(pts[idx])
        o3d.visualization.draw_geometries([pcd])

    lidar2cam = cl.LiDAR2Camera(calbs[0])
    print("P :"+str(lidar2cam.P))
    print("-")
    print("RO "+str(lidar2cam.R0))
    print("-")
    print("Velo 2 Cam " +str(lidar2cam.V2C))

    video_images = sorted(glob.glob(data_dir+"//test//video4//images/*.png"))
    video_points = sorted(glob.glob(data_dir+"//test//video4//points/*.pcd"))

    # Build a LiDAR2Camera object
    lidar2cam = cl.LiDAR2Camera(calbs[0])

    result_video = []

    weights = data_dir + "//model//yolov4//yolov4.weights"
    config = data_dir + "//model//yolov4//yolov4.cfg"
    names = data_dir + "//model//yolov4//coco.names"

    detector = cl.Detector(0.4)
    detector.load_model(weights, config, names)

    image = cv2.imread(video_images[0])

    for idx, img_path in enumerate(video_images):
        image = cv2.imread(img_path)
        detections, image = detector.detect(image, True, True)
        point_cloud = np.asarray(o3d.io.read_point_cloud(video_points[idx]).points)
        # image = ut.display_lidar_on_image(lidar2cam, point_cloud, image)
        pts_3D, pts_2D = ut.get_lidar_on_image(lidar2cam, point_cloud, (image.shape[1], image.shape[0]))
        image, _ = ut.lidar_camera_fusion(pts_3D, pts_2D, detections, image)
        result_video.append(image)

    out = cv2.VideoWriter('..//Data//output//videos//out4_fused.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (image.shape[1], image.shape[0]))

    for i in range(len(result_video)):
        # out.write(cv2.cvtColor(result_video[i], cv2.COLOR_BGR2RGB))
        out.write(result_video[i])
    out.release()


def mid_level_fusion(data_dir, index=0):
    imgs, pts, labels, calibs = ut.load_data(data_dir)

    weights = data_dir + "//model//yolov4//yolov4.weights"
    config = data_dir + "//model//yolov4//yolov4.cfg"
    names = data_dir + "//model//yolov4//coco.names"

    detector = cl.Detector(0.4)
    detector.load_model(weights, config, names)

    """PIPELINE STARTS FROM HERE"""

    # load the image
    image = cv2.imread(imgs[index])

    # create LiDAR2Camera object
    lidar2cam = cl.LiDAR2Camera(calibs[index])

    # 1 - Run 2D object detection on image
    detections, yolo_detections = detector.detect(image, draw_bboxes=False, display_labels=False)

    # load lidar points and project them inside 2d detection
    point_cloud = np.asarray(o3d.io.read_point_cloud(pts[index]).points)
    pts_3D, pts_2D = ut.get_lidar_on_image(lidar2cam, point_cloud, (image.shape[1], image.shape[0]))
    lidar_pts_img, _ = ut.lidar_camera_fusion(pts_3D, pts_2D, detections, image)

    # Build a 2D Object
    list_of_2d_objects = ut.fill_2D_obstacles(detections)

    # Build a 3D Object (from labels)
    list_of_3d_objects = ut.read_label(labels[index])

    # Get the LiDAR Boxes in the Image in 2D and 3D
    lidar_2d, lidar_3d = ut.get_image_with_bboxes(lidar2cam, lidar_pts_img, list_of_3d_objects)

    # Associate the LiDAR boxes and the Camera Boxes
    lidar_boxes = [obs.bbox2d for obs in list_of_3d_objects]  # Simply get the boxes
    camera_boxes = [np.array([box[0], box[1], box[0] + box[2], box[1]+box[3]]) for box in detections[:, 1]]
    # camera_boxes = [obs.bbox for obs in list_of_2d_objects]
    matches, unmatched_lidar_boxes, unmatched_camera_boxes = ut.associate(lidar_boxes, camera_boxes)

    # Build a Fused Object
    final_image, _ = ut.build_fused_object(list_of_2d_objects, list_of_3d_objects, matches, lidar_2d)

    # draw yolo detections on top to fused results
    final_image = ut.draw_yolo_detections(final_image, detections)

    cv2.imshow("lidar_3d", lidar_3d)
    cv2.imshow("lidar_2d", lidar_2d)
    cv2.imshow("yolo_detections", yolo_detections)
    cv2.imshow("lidar_pts_img", lidar_pts_img)
    cv2.imshow("final_image", final_image)
    cv2.waitKey(0)

    return final_image


if __name__ == "__main__":
    data_dir = "..//Data//"
    # low_level_fusion(data_dir, show_random_pcl=True)
    mid_level_fusion(data_dir)