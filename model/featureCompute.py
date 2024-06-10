import dlib
import os
import logging
import csv
import numpy as np
import cv2
from flask import jsonify

face_reco_model = dlib.face_recognition_model_v1("dlib/dlib_face_recognition_resnet_model_v1.dat")
predictor = dlib.shape_predictor("dlib/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


class featureCompute:
    # 计算单张的图像的128D特征
    def return_128D_feature_single(self, path_picture_personX):
        img = cv2.imread(path_picture_personX)
        # 检测人脸
        # todo
        faces = detector(img)
        logging.info("检测到人脸图片{0:40}".format(path_picture_personX))
        # 用检测到人脸的原图片去计算128D特征
        if len(faces):
            # 获得人脸特征点
            shape = predictor(img, faces[0])
            # 获得人脸128D特征
            face_descriptor = face_reco_model.compute_face_descriptor(img, shape)
        else:
            face_descriptor = 0
            logging.warning("录入失败，未检测到人脸")
        return face_descriptor

    # def tansform_pic_to_csv(self):
    # 获得person的128D特征 - 平均特征
    def return_128D_features_ave(self, path_personX):
        features_128D_Of_PersonX = []
        faces = os.listdir(path_personX)
        if faces:
            for face in faces:
                logging.info("正在读取{0:40}下的图片：{1:20}".format(path_personX, face))
                # 计算该图片的128D特征
                face_feature_128D = self.return_128D_feature_single(os.path.join(path_personX, face))
                # 跳过识别失败的图片
                if face_feature_128D == 0:
                    continue
                else:
                    features_128D_Of_PersonX.append(face_feature_128D)

        else:
            logging.warning("文件夹{0:40}为空".format(path_personX))

        if features_128D_Of_PersonX:
            features_128D_Of_PersonX_ave = np.array(features_128D_Of_PersonX, dtype=object).mean(axis=0)
        else:
            features_128D_Of_PersonX_ave = np.zeros(128, dtype=object, order='C')
        return features_128D_Of_PersonX_ave

    def featureCompute(self, facePath):
        datapath = os.path.dirname(facePath)
        csvpath = os.path.join(datapath, "features.csv")
        userId = os.path.basename(os.path.dirname(os.path.dirname(facePath)))
        with open(csvpath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            logging.info("正在处理{0:40}{1:20}".format(facePath, userId))
            features_128D_Of_PersonX = self.return_128D_features_ave(facePath)
            #获得person的128D特征
            # features_128D_Of_PersonX = np.insert(features_128D_Of_PersonX, 0, userId, axis=0)
            writer.writerow(features_128D_Of_PersonX)
            logging.info("{0:40}{1:20}处理完毕".format(facePath, userId))
        logging.info("全部处理完毕, 全部人脸数据存入{0:40}".format(csvpath))
        if os.path.exists(csvpath):
            return jsonify({"status": 1, "url": csvpath})
        else:
            return jsonify({"status": -1, "message": "csv处理失败"})
