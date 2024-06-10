import logging
import os
import time

import cv2
import dlib
import numpy as np
import pandas as pd
from PIL import ImageFont
from flask import jsonify

from model.headPoseEstimation import Head_Pose
from services import userService

from model import GLOBAL_TOTAL_ACCOUNTS, GLOBAL_ACCEPT_ACCOUNTS, GLOBAL_ACCOUNTS_FOR_LEFT, \
    GLOBAL_ACCOUNTS_FOR_RIGHT, GLOBAL_ACCOUNTS_FOR_UP

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor("dlib/shape_predictor_68_face_landmarks.dat")

# Dlib Resnet 人脸识别模型, 提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("dlib/dlib_face_recognition_resnet_model_v1.dat")

GLOBAL_FEATURES = []
abc = 0


class FaceIdentify:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        self.font_chinese = ImageFont.truetype("simsun.ttc", 30)

        # 用来存储所有录入人脸特征的数组
        self.features_known_list = []
        # 用来存储录入人脸名字
        self.face_name_known_list = []

        # 用来存储上一帧和当前帧 ROI 的质心坐标
        self.last_frame_centroid_list = []
        self.current_frame_centroid_list = []

        # 用来存储当前帧检测出目标的名字
        self.current_frame_name_list = []

        # 上一帧和当前帧中人脸数的计数器
        self.last_frame_faces_cnt = 0
        self.current_frame_face_cnt = 0

        # 用来存放进行识别时候对比的欧氏距离
        self.current_frame_face_X_e_distance_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字
        self.current_frame_face_position_list = []
        # 存储当前摄像头中捕获到的人脸特征
        self.current_frame_face_feature_list = []

        # 控制再识别的后续帧数
        # 如果识别出 "unknown" 的脸, 将在 reclassify_interval_cnt 计数到 reclassify_interval 后, 对于人脸进行重新识别
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    # 接收 单张图片
    # 返回 单张人脸特征
    def return_128D_feature_single(self, image):
        # 检测人脸
        print("正在检测单张图片的128D特征向量")
        faces = detector(image)
        # 用检测到人脸的原图片去计算128D特征
        if len(faces):
            # 获得人脸特征点
            shape = predictor(image, faces[0])
            # 获得人脸128D特征
            face_descriptor = face_reco_model.compute_face_descriptor(image, shape)
            print("获得了特征向量")
            return face_descriptor
        else:
            print("图片中没有人脸")
            return 0

    # def tansform_pic_to_csv(self):
    # 接受 imageList：裁剪后的图片
    # 返回 平均特征向量
    def return_128D_features_ave(self, imageList):
        features_128D_Of_PersonX = []
        for face in imageList:
            # 计算该图片的128D特征
            face_feature_128D = self.return_128D_feature_single(face)
            # 跳过识别失败的图片
            if face_feature_128D == 0:
                continue
            else:
                features_128D_Of_PersonX.append(face_feature_128D)

        if features_128D_Of_PersonX:
            features_128D_Of_PersonX_ave = np.array(features_128D_Of_PersonX, dtype=object).mean(axis=0)
        else:
            features_128D_Of_PersonX_ave = np.zeros(128, dtype=object, order='C')
        return features_128D_Of_PersonX_ave

    # 接受  ImageQeueue 原始图片
    # 返回  ImageList   裁剪后的图片
    def computeFeature(self, frame):
        img_rd = frame
        faces = detector(img_rd)
        if len(faces) > 1:
            print("实时特征计算失败 多人脸")
        elif len(faces) < 1:
            print("实时特征计算失败 无人脸")
        else:
            # 设置图片路径
            GLOBAL_TOTAL_ACCOUNTS[0] += 1
            for k, d in enumerate(faces):
                height = (d.bottom() - d.top())
                width = (d.right() - d.left())
                hh = int(height / 2)
                ww = int(width / 2)
                # 根据人脸大小生成空的图像
                image_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)
                img_height, img_width, _ = img_rd.shape
                for ii in range(height * 2):
                    for jj in range(width * 2):
                        y = d.top() - hh + ii
                        x = d.left() - ww + jj
                        # 检查坐标是否在图像范围内
                        if 0 <= y < img_height and 0 <= x < img_width:
                            image_blank[ii][jj] = img_rd[y][x]
                print("实时人脸计算成功")
                tmp = self.return_128D_feature_single(image_blank)
                return tmp
            print("实时特征计算失败 tag")

    # 从 "features_all.csv" 读取录入人脸特征
    def get_face_database(self, CsvPath):
        if os.path.exists(CsvPath):
            csv_rd = pd.read_csv(CsvPath, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(0, 128):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.features_known_list.append(features_someone_arr)
            print("数据库中的特征向量已加载")
            return features_someone_arr
        else:
            print("数据库: 特征向量不存在")
            return []

    # 计算两个128D向量间的欧式距离
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        if feature_2 is None:
            return 1
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def getLoginUseryId(self, userId):
        loginUser = userService.getUserById(userId)
        return loginUser

    # 处理获取的视频流, 进行人脸识别
    def faceIdentifier(self, userId, frame):  # 用户账号
        # 1. 通过用户账号从数据库中获得人脸特征存储路径
        #   1.1）判断 是否完成注册
        #   1.2）查找 人脸特征存储信息
        # 2. 计算人脸特征
        #   2.1）使用featureCompute的 "return_128D_features_ave" 计算特征
        #   2.2）计算 List<Image>的平均特征
        # 3. 计算欧式距离
        #   if True:
        #       todo 计算List<Image>头部姿势
        #           if ok:
        #               ok
        #           else:
        #               作弊检测
        #   else:
        #       识别失败
        loggingUser = self.getLoginUseryId(userId)
        if loggingUser.isRegisterFace == 1:
            featuresLocalPath = loggingUser.FeaturesLocalPath
            feature_store_in_database = self.get_face_database(featuresLocalPath)
            TIME = time.time()
            feature_current_frame = self.computeFeature(frame)
            euclidean_distance = self.return_euclidean_distance(feature_store_in_database, feature_current_frame)
            print("耗时",time.time() - TIME, " 欧式距离:", euclidean_distance)
            if euclidean_distance < 0.55:
                GLOBAL_ACCEPT_ACCOUNTS[0] += 1
                return jsonify({"status": 1, "message": f"{euclidean_distance}人脸认证通过"})
            return jsonify({"status": -1,
                            "message": f"{euclidean_distance}人脸认证未通过"})
        else:
            return jsonify({"status": -2, "message": "未进行人脸未注册"})
