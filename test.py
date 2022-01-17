import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()

app = Flask(__name__)  # '__main__'
maxSpeed = 30


def preProcess_white(img):
    img = img[60:135, :, :]

    rec = img
    rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    low_white = np.array([0, 40, 70])  # np.array([100, 40, 70])
    up_white = np.array([255, 83, 75])  # np.array([125, 83, 75])

    mask1 = cv2.inRange(hsv, low_white, up_white)

    sensitivity = 65

    low_white = np.array([0, 0, 255 - sensitivity])  # np.array([0, 0, 255 - sensitivity])
    up_white = np.array([230, sensitivity, 255])  # np.array([230, sensitivity, 255])

    mask2 = cv2.inRange(hsv, low_white, up_white)

    masks = cv2.add(mask1, mask2)

    edges = cv2.Canny(masks, 50, 120)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, maxLineGap=4, minLineLength=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(rec, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('camera', rec)
    cv2.waitKey(1)
    img = cv2.resize(img, (200, 66))

    img = img / 255

    return img


def preProcess_yellow(img):
    rec = img
    img = img[60:135, :, :]

    img = cv2.GaussianBlur(img, (7, 7), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    low_yellow = np.array([22, 40, 0])  # [22, 40, 0] [30, 36, 120]
    up_yellow = np.array([45, 255, 255])  # [45, 255, 255] [45, 45, 160]

    mask = cv2.inRange(hsv, low_yellow, up_yellow)

    edges = cv2.Canny(mask, 50, 120)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, maxLineGap=2, minLineLength=5)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(rec, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img = cv2.resize(img, (200, 66))

    img = img / 255

    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcess_yellow(image)
    image = np.array([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed - 2*abs(steering)
    # print('{} {} {}'.format(steering, throttle, speed))
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
