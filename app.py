import os
from flask import Flask, request, render_template, flash, redirect, send_from_directory, url_for
import requests

from bs4 import BeautifulSoup
import urllib.request as urllib
from urllib.request import urlopen

# CV
from imutils.video import VideoStream
import imutils
import cv2

# FUNCTIONS
from face_detection.detect_faces_video import video_detector
from face_detection.detect_faces_images import image_detector

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


# @app.route('/', methods=['GET', 'POST'])
# def upload_image():

#     # if 'file' not in request.files:
#     #     print('No file part')
#     #     flash('No file part')
#     #     return redirect(request.url)

#     if file.filename == '':
#         print('No image selected for uploading')
#         flash('No image selected for uploading')
#         error = 'No image selected for uploading'
#         return render_template("index.html", error=error, scroll='scrollable')

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         print('upload_image filename: ' + filename)
#         # flash('Image successfully uploaded and displayed below')

#         pred = predict_image(filepath)

#         ing_list, recipe = return_details(pred)
#         # recipe = return_recipe(pred)

#         return render_template('index.html', user_img=filename, pred=pred, scroll='scrollable', ingredients_list=ing_list, recipe=recipe)
#     else:
#         error = ('Allowed image types are -> png, jpg, jpeg, gif')

#         return render_template("index.html", error=error, scroll='scrollable')

#     return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
