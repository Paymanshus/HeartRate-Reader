# from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt
import os

# args = {}
# args['confidence'] = 0.6

# Testing videos
vid_url = "https://192.168.0.14:8080/video"
vid_path = "test_vids/fingertip_phone2.mp4"

out_path = "test_vids/out"


def detect_heartbeat(vid_path, out_path, vid_url=None,):

    # Declaring output path
    vid_name = os.path.basename(vid_path).split('.')[0]
    save_path = out_path + '/' + vid_name + '_plot.jpg'

    # Initializing video stream
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if vid_url:
        cap.open(vid_url)

    # # Image crop
    # x, y, w, h = 800, 500, 100, 100
    # x, y, w, h = 950, 300, 100, 100

    # Creating an array to store average brightness values of video frames
    heartbeat_count = 128
    heartbeat_values = [0]*heartbeat_count
    heartbeat_times = [time.time()]*heartbeat_count

    # Matplotlib graph surface
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Using input from stream
    while True:
        ret, frame = cap.read()
        if ret is False:
            cv2.imwrite(save_path, plot_img_np)
            cv2.waitKey(1)
            return save_path

        frame = cv2.resize(
            frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

        # Splitting into 3 channels: R, G and B
        b, g, r = cv2.split(frame)

        # crop_img = r[y:y + h, x:x + w]

        # Attempting to reduce noise and smoothen the image by apqplying Gaussian Blur on frame
        blur = cv2.GaussianBlur(r, (5, 5), 0)

        # Update the data and timestamps
        heartbeat_values = heartbeat_values[1:] + [np.average(frame)]
        heartbeat_times = heartbeat_times[1:] + [time.time()]

        # Draw matplotlib graph to numpy array
        ax.plot(heartbeat_times, heartbeat_values)
        fig.canvas.draw()
        plot_img_np = np.fromstring(fig.canvas.tostring_rgb(),
                                    dtype=np.uint8, sep='')
        plot_img_np = plot_img_np.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        plt.cla()
        # Display the frames
        cv2.imshow('Blur', blur)
        cv2.imshow('Image', frame)
        cv2.imshow('Graph', plot_img_np)
        cv2.imwrite(save_path, plot_img_np)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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

        # pixel = frame[]

        # Show the output image frame
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF

        # # Break on pressing 'q' key
        # if key == ord("q") or key == 27:
        #     break

    cv2.destroyAllWindows()
    cap.release()

    return save_path


if __name__ == "__main__":

    save_path = detect_heartbeat(vid_path, out_path)
    print(save_path)
