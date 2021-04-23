# from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
# import time
import cv2
import matplotlib.pyplot as plt

args = {}
# args['confidence'] = 0.6

vid_url = "https://192.168.0.14:8080/video"
vid_path = "test_vids/fingertip_phone2.mp4"


def process_video(args):

    # Initializing video stream
    video = cv2.VideoCapture(vid_path)
    # video.open(vid_url)

    # Using input from stream
    while True:
        ret, frame = video.read()
        frame = cv2.resize(
            frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

        b, g, r = cv2.split(frame)

        cv2.imshow("red", r)

        # FFT
        # dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        # dft_shift = np.fft.fftshift(dft)

        # magnitude_spectrum = 20 * \
        #     np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        # plt.subplot(121), plt.imshow(gray, cmap='gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        # plt.show()

        pixel = frame

        # Show the output image frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Break on pressing 'q' key
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
    video.release()

    return


if __name__ == "__main__":

    process_video(args)
