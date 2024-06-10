import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging

from flask import jsonify
from PIL import Image

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()


class FaceModify:
    def modifyPicByFace(self, picturePath):
        global facePathAfterProcess
        TIME = time.time()
        logging.info(f'{picturePath}')
        if os.path.exists(picturePath):
            parentpicturePath = os.path.dirname(picturePath)
            for image in os.listdir(picturePath):
                name = image.split('.')[0]
                # 1. 使用dlib检测器检测人脸
                img_rd = cv2.imread(os.path.join(picturePath, image))
                faces = detector(img_rd)
                if len(faces) > 1:
                    return jsonify({"status": -1, "message": "more than one face detected"})
                elif len(faces) < 1:
                    return jsonify({"status": -1, "message": "no face detected"})
                else:
                    # 设置图片路径
                    for k, d in enumerate(faces):
                        facePathAfterProcess = os.path.join(parentpicturePath, 'data', 'face')
                        if not os.path.exists(facePathAfterProcess):
                            os.makedirs(facePathAfterProcess)
                        height = (d.bottom() - d.top())
                        width = (d.right() - d.left())
                        hh = int(height / 2)
                        ww = int(width / 2)
                        # 根据人脸大小生成空的图像
                        image_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)
                        picturePathAfterProcessId = os.path.join(facePathAfterProcess, name + ".jpg")
                        img_height, img_width, _ = img_rd.shape
                        for ii in range(height * 2):
                            for jj in range(width * 2):
                                y = d.top() - hh + ii
                                x = d.left() - ww + jj
                                # 检查坐标是否在图像范围内
                                if 0 <= y < img_height and 0 <= x < img_width:
                                    image_blank[ii][jj] = img_rd[y][x]
                        cv2.imwrite(picturePathAfterProcessId, image_blank)
            TIME = time.time() - TIME
            return jsonify({"status": 1, "url": facePathAfterProcess, "time": f"{int(TIME*1000)}ms"})
        else:
            return jsonify({'status': -1, 'message': "faceLocalPath not exist"})
