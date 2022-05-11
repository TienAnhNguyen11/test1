import requests_toolbelt.cookies.forgetful
from flask import Flask, request
from flask_cors import CORS, cross_origin

import cv2
import numpy as np
import base64

def face_detecting(face):
    # khoi tao bo phat hien khuon mat
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # chuyen anh mau thanh anh gray
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # dem so luong khuon mat trong anh
    faces = face_cascade.detectMultiScale(gray, 1.2, 10)

    face_number = len(faces)
    return face_number

def convert_base64_into_image(base64_image):
    try:
        base64_image = np.fromstring(base64.b64decode(base64_image), dtype=np.uint8)
        base64_image = cv2.imdecode(base64_image, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return base64_image

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'

@app.route("/face-counting", methods=['POST'])
@cross_origin(origin="*")


def face_detecting_process():
    face_numbers = 0
    # nhan anh tu client gui len
    facebase64 = request.form.get("facebase64")
    # chuyen base64 ve OpenCV format
    face = convert_base64_into_image(facebase64)
    # dem so luong khuon mat trong anh
    face_numbers = face_detecting(face)

    return "So mat la: " + str(face_numbers)


if __name__ == "__main__":
    app.run(debug=True)
