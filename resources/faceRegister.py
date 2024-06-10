import os

from flask import request
from flask.json import jsonify

from resources import app, api
from flask_restful import Resource
from resources import api
from model.faceModify import FaceModify
from model.featureCompute import featureCompute
import logging


class FaceRegister(Resource):
    def get(self):
        return {'name': 'csp', 'id': 1}, 200

    def post(self):
        # 拿到json {}
        RegisterJson = request.json
        userId = RegisterJson.get('userId')
        FacePicLocalPath = RegisterJson.get('FacePicLocalPath')

        if os.path.exists(FacePicLocalPath):
            # 1. 裁剪人脸图片
            ModifyJson = FaceModify().modifyPicByFace(FacePicLocalPath).get_json()
            if ModifyJson.get('status') == -1:
                return jsonify({"status": -1, "message": "Picture Process error"})
            elif ModifyJson.get('status') == 1:
                # 2. 提取人脸图片的特征
                FeatureComputer = featureCompute()
                FeatureComputeJson = FeatureComputer.featureCompute(ModifyJson.get('url')).get_json()
                FacePicLocalPathTmp = str(ModifyJson.get('url'))
                FeaturesLocalPathTmp = str(FeatureComputeJson.get('url'))
                return jsonify({"status": 1,
                                "FacePicLocalPath": FacePicLocalPathTmp,
                                "FeaturesLocalPath": FeaturesLocalPathTmp})
            else:
                return jsonify({"status": -1, "message": "other ModifyJson"})
        else:
            return jsonify({"status": -1, "message": "FacePicLocalPath does not exist"})


api.add_resource(FaceRegister, '/face/register')
