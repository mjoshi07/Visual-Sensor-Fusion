import numpy as np
import cv2
import matplotlib.pyplot as plt


class Detector():
    def __init__(self, conf_threshold=0.4, classes_to_detect=None, nms_threshold=0.4, input_size=(416, 416), scale=1.0/255):
        self.conf_threshold = conf_threshold
        self.classes_to_detect = classes_to_detect
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.scale = scale
        self.model = None
        self.ln = None
        self.names = None
        cmap = plt.cm.get_cmap("hsv", 256)
        self.cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    def get_layers(self):
        ln = self.model.getLayerNames()
        return [ln[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]

    def read_names(self, names_path):
        f = open(names_path, "r")
        return [n.split('\n')[0] for n in f.readlines()]

    def load_weights(self, weights_path, config_path):
        return cv2.dnn.readNet(weights_path, config_path)

    def load_model(self, weights_path, config_path, names_path):
        self.model = self.load_weights(weights_path, config_path)
        self.ln = self.get_layers()
        self.names = self.read_names(names_path)
        a = 0

    def detect(self, image, draw_bboxes=False, display_labels=False):
        img = image.copy()
        (H, W) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, self.scale, self.input_size, swapRB=True, crop=False)
        self.model.setInput(blob)
        layerOutputs = self.model.forward(self.ln)

        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.conf_threshold:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    if self.classes_to_detect is None or self.names[classID] in self.classes_to_detect:
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        detections = []
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                class_id = classIDs[i]
                conf = confidences[i]
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                bbox = [x, y, w, h]
                # draw a bounding box rectangle and label on the image
                if draw_bboxes:
                    color = self.cmap[int(255.0 / (class_id + 1)), :]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color=tuple(color), thickness=2)
                    if display_labels:
                        cv2.putText(img, str(self.names[class_id]), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1,
                                    (0, 0, 255), 2, 16)
                        cv2.putText(img, str(self.names[class_id]), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1,
                                    (255, 255, 255), 1, 16)

                detections.append([class_id, bbox, conf])

        return np.array(detections), img