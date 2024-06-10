import base64
import logging
import time

import cv2
import numpy as np
from flask import jsonify, request

from model import GLOBAL_ACCOUNTS_FOR_RIGHT, GLOBAL_ACCOUNTS_FOR_LEFT
from model.faceIdentify import FaceIdentify
from model.headPoseEstimation import Head_Pose
from resources import app, api
from flask_restful import Resource
from resources import api
import dlib

detector = dlib.get_frontal_face_detector()


def hasFace(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if faces:
        return True
    else:
        return False


class faceDetecter(Resource):

    def __init__(self):
        self.FaceDetectResult = None

    def get(self):
        return self.FaceDetectResult

    def post(self):
        frameJson = request.json
        frame_data = frameJson['frame']
        jpg_original = base64.b64decode(frame_data)
        # 转换为numpy数组以供OpenCV使用
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        # 将图像数据解码为图像矩阵
        frame = cv2.imdecode(jpg_as_np, flags=cv2.IMREAD_COLOR)
        faceidentify = FaceIdentify()
        # cv2.namedWindow("camera", 1)
        # cv2.imshow("camera", frame)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        self.FaceDetectResult = faceidentify.faceIdentifier(2, frame).get_json()
        return self.FaceDetectResult


logging.basicConfig(level=logging.INFO)
api.add_resource(faceDetecter, '/face/detect')
