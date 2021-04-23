from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2


# Parsing the arguments
def parse_args_vid(prototxt=True, model=True, confidence=True):
    ap = argparse.ArgumentParser()
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


def video_detector(args, net):

    # Initializing video stream
    print("-------Starting Video Stream--------")
    # VideoStream using imutils
    # vs = VideoStream(src=0).start()
    # time.sleep(2.0)

    cap = cv2.VideoCapture(0)

    # Using input from stream
    while True:
        # frame = vs.read()
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=400)

        # Converting to blob
        (h, w) = frame.shape[:2]  # Getting dimensions
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # Feeding blob through net
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

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                # Syntax: cv2.rectangle(img, pt1, pt2, color, thickness)

                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                # Syntax: cv2.putText(img, text, org, fontFace, fontScale, color)

        # Show the output image frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Break on pressing 'q' key
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
    # vs.stop()
    cap.release()


if __name__ == "__main__":

    args = parse_args_vid(prototxt=False, model=False)

    # Hardcoding file paths(relative)
    prototxt_path = r"models\SSD_Face_Detection\deploy.prototxt.txt"
    model_path = r"models\SSD_Face_Detection\res10_300x300_ssd_iter_140000.caffemodel"
    # image_path = "test1.jpg"

    # Loading Model
    print("-------Loading Model--------")
    # net = cv2.dnn.readNetFromCaffe(
    #     args["prototxt"], args["model"])  # Initialized model net USING ARG PARSER

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    video_detector(args, net)
