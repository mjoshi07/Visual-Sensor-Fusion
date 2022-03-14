import cv2


def draw_yolo_detections(image, detections, color=(0,255,0)):
    img = image.copy()
    with open("..//Data//model//yolov4/coco.names", 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    for detect in detections:
        bbox = detect[1]
        category = classes[int(detect[0])]
        cv2.rectangle(img, bbox, color, 2)
        cv2.putText(img, str(category), (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return img