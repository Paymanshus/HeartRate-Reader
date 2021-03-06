# Task List

1. Design and build a simple web UI/Mobile app for below objectives
2. The interface has functionality to upload and display image file from local device
3. The interface has functionality to access device camera
4. Implement face detection using OpenCV or any other computer vision library on uploaded image
5. Implement a boundary box to track face on the streaming video from device camera
6. Calculate heart rate by recording finger tip video

## TODO List

- [x] Web App Skeleton
- [x] Image Upload Functionality
- [x] Image Display on Web App
- [x] Face Detection on Uploaded Image
- [x] Face Detection with Webcam
- [x] Finger tip heart rate detection on uploaded video
- [x] Implementing Finger tip heart rate on website
- [ ] Running live Heart Rate check from Webcam and live graphs
- [ ] Use IP Cam to link with phone camera

## Getting Started

To get a local copy up and running follow these simple steps.

### Running the app

1. Clone the repo
   ```
   git clone https://github.com/Paymanshus/HeartRate-Reader
   ```
2. Install requirements from requirements.txt

   ```
   pip install -r requirements.txt
   ```

3. Run the Flask app
   ```
   python app.py
   ```

### Running the HeartRate Reader Graphs

- To get a live graph of the heart beat reading:

  ```
  cd heartbeat
  python fingertip_heartbeat.py
  ```

  - Currently runs on sample video from test_vids folder
