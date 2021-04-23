import os
from flask import Flask, request, render_template, flash, redirect, send_from_directory, url_for, Response
import requests

# CV
from imutils.video import VideoStream
import imutils
import cv2

# FUNCTIONS
from face_detection.detect_faces_video import VideoCamera
from face_detection.detect_faces_images import image_detector

from heartbeat.fingertip_heartbeat import detect_heartbeat

from werkzeug.serving import run_simple
from werkzeug.utils import secure_filename


# # set to True to inform that the app needs to be re-created
# to_reload = False

# codePath = os.path.dirname(os.path.abspath('preprocessing.py'))

proto_path = r"models\deploy.prototxt.txt"
model_path = r"models\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

args = {}
args['prototxt'] = r"models\deploy.prototxt.txt"
args['model'] = r"models\res10_300x300_ssd_iter_140000.caffemodel"
args['confidence'] = 0.6


net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() \
        in app.config['UPLOAD_EXTENSIONS']


def gen(camera):
    while True:
        # get camera frame
        frame = camera.video_detector(args, net)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# def add_imgpath(image):
#     global args
#     args['image'] = image


app = Flask(__name__, template_folder='templates')

app.config['UPLOAD_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif', 'jfif'])
app.config['UPLOAD_PATH'] = 'static/img/uploads'
# app.config['EXPLAIN_TEMPLATE_LOADING'] = True

out_path = 'static/img/outputs'


@app.route('/')
def home():
    return render_template("index.html")


# ------------------------------
# Switching between Submit Types
# ------------------------------
@app.route('/', methods=['GET', 'POST'])
def home_type():

    if request.method == 'POST':
        if 'image-page' in request.form:
            # return render_template("index.html", image_upload=True)
            return redirect(url_for('image_upload'))
        elif 'video-page' in request.form:
            # return render_template("index.html", video_page=True)
            return redirect(url_for('video_stream'))

        # elif 'video-upload' in request.form:
        #     return render_template("index.html", video_upload=True)

        elif 'image-upload' in request.files:
            print('File Request Passed')

            uploaded_file = request.files['image-upload']
            print(uploaded_file)

            filename = secure_filename(uploaded_file.filename)
            if filename != '':

                image_path = os.path.join(app.config['UPLOAD_PATH'], filename)
                # add_imgpath(image_path)
                print(filename)

                uploaded_file.save(image_path)
                print(os.listdir(app.config['UPLOAD_PATH']))

                # Saves image to out_path
                save_path = image_detector(args, net, image_path, out_path)

                return render_template("index.html", filename=save_path)

            else:
                # Make user reupload image
                return render_template("index.html", image_upload=True)

        else:
            print("Else condition executed")
            return render_template("index.html")


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


@app.route('/image_upload')
def image_upload():
    return render_template("index.html", image_upload=True)


@app.route('/video_stream')
def video_stream():
    return render_template("index.html", video_stream=True)


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Switching To Heartbeat Detector
# @app.route('/heartbeat')
# def heartbeat():
#     return render_template("heartbeat.html")

@app.route('/heartbeat', methods=['GET', 'POST'])
def heartbeat():
    print(request.method)
    print(request.form)
    if request.method == 'POST':
        if 'video-upload' in request.form:
            # return render_template("index.html", image_upload=True)

            return redirect(url_for('heart_video_upload'))

        elif 'video-stream' in request.form:
            # return render_template("index.html", video_page=True)
            return redirect(url_for('heart_video_stream'))

        elif 'video-upload' in request.files:
            print('Video Request Passed')

            uploaded_file = request.files['video-upload']
            print(uploaded_file)

            filename = secure_filename(uploaded_file.filename)
            if filename != '':

                video_path = os.path.join(app.config['UPLOAD_PATH'], filename)
                # add_imgpath(image_path)
                print(filename)

                uploaded_file.save(video_path)
                print(os.listdir(app.config['UPLOAD_PATH']))

                # Saves plot to out_path
                save_path = image_detector(args, net, image_path, out_path)

                return render_template("heartbeat.html", pred=pred)

            else:
                # Make user reupload image
                return render_template("heartbeat.html", video_upload=True)
    else:
        return render_template("heartbeat.html")


# Recording from Live Video Stream page
@app.route('/heartbeat/video_stream')
def heart_video_stream():
    return render_template('heartbeat.html', video_stream=True)


# Recording to be uploaded by user
@app.route('/video_upload')
def heart_video_upload():
    return render_template("heartbeat.html", video_upload=True)


# Routing video feed to display in frame
@app.route('/heartbeat/video_feed')
def heart_video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
