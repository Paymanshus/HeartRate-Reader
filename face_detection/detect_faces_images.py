import numpy as np
import argparse
import cv2
import os


# Parsing the arguments
def parse_args(image=True, prototxt=True, model=True, confidence=True):
    ap = argparse.ArgumentParser()
    if image:
        ap.add_argument("-i", "--image", required=False,
                        help="path to input image")
    if prototxt:
        ap.add_argument("-p", "--prototxt", required=True,
                        help="path to Caffe 'deploy' prototxt file")
    if model:
        ap.add_argument("-m", "--model", required=True,
                        help="path to Caffe pre-trained model")
    if confidence:
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    return args


def image_detector(args, net, image_path, out_path):
    image = cv2.imread(image_path)

    image_name = os.path.basename(image_path).split('.')[0]
    save_path = out_path + '/' + image_name + '_out.jpg'
    print("Saving at: ", save_path)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Applying face detection
    net.setInput(blob)
    detections = net.forward()

    # Looping over detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # confidence of prediction

        # ensuring confidence is greater than the minimum confidence
        if confidence > args["confidence"]:

            # Finding (x,y) coordinates of bounding box
            # Conversion to numpy array of format [w, h, w, h]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype(
                "int")  # [startX, startY, endX, endY]

            # Drawing bounding box of face and showing probability
            text = "{:.2f}%".format(confidence * 100)

            # If face detection occurs at the top of the image, then place text at the bottom of the bounding box(y coordinate for text derived from StartY)
            # Else text placed at the top of the box(y = startY + 10)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            # Syntax: cv2.rectangle(img, pt1, pt2, color, thickness)

            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            # Syntax: cv2.putText(img, text, org, fontFace, fontScale, color)

    # Displaying output image
    # cv2.imshow("Output", image)
    # cv2.waitKey(0)

    # Saving output image
    cv2.imwrite(save_path, image)
    return save_path


def image_datector_haar(image_path):
    face_cascade = cv2.CascadeClassifier(
        'models/haarcascade_frontalface_default.xml')

    img = cv2.imread(image_path)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img_g, 1.1, 1)

    print(faces)

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == "__main__":

    args = parse_args(prototxt=False, model=False)

    # Hardcoding file paths(relative)

    # Body Detection VGG model
    # prototxt_path = r"models\VGG_Body_Detection\deploy.prototxt"
    # model_path = r"models\VGG_Body_Detection\VGG_VOC0712_SSD_300x300_iter_120000.caffemodel"

    # ssd Face Detection Model
    prototxt_path = r"models\SSD_Face_Detection\deploy.prototxt.txt"
    model_path = r"models\SSD_Face_Detection\res10_300x300_ssd_iter_140000.caffemodel"
    if args['image']:
        image_path = args['image']
    else:
        image_path = "Test Images/multitest.jpg"

    out_path = "Test Images/out"
    # Loading Model
    print("-------Loading Model--------")
    # net = cv2.dnn.readNetFromCaffe(
    #     args["prototxt"], args["model"])  # Initialized model net USING ARG PARSER

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Load input image and creating a blob by resiinge it to a fixed 300x300 pixels and then normalizing it
    # image = cv2.imread(args["image"])  # USING ARG PARSER
    # image = cv2.imread(image_path)

    image_detector(args, net, image_path, out_path)
    image_datector_haar(image_path)
