import time

import schedule
from flask import jsonify
from flask_restful import Resource

from model import GLOBAL_ACCEPT_ACCOUNTS, GLOBAL_TOTAL_ACCOUNTS, GLOBAL_ACCOUNTS_FOR_LEFT, \
    GLOBAL_ACCOUNTS_FOR_RIGHT, GLOBAL_ACCOUNTS_FOR_UP, cleanFace, lasttime

from resources import api


class AntiCheatingApp(Resource):
    def get(self):
        return self.return_detect_result()

    def return_detect_result(self):
        if (lasttime[0] == 0):
            lasttime[0] = time.time()
        else:
            print("need " + str(time.time() - lasttime[0]))
            lasttime[0] = time.time()
        print(" 进行人脸认证任务 ")
        print(f" SUCCESS {GLOBAL_ACCEPT_ACCOUNTS[0]} / {GLOBAL_TOTAL_ACCOUNTS[0]}")
        if (GLOBAL_TOTAL_ACCOUNTS[0]) and (GLOBAL_ACCEPT_ACCOUNTS[0] / GLOBAL_TOTAL_ACCOUNTS[0]) > 0.4:
            res = jsonify(
                {"status": 1, "message": f"人脸认证成功 {GLOBAL_ACCEPT_ACCOUNTS[0] / GLOBAL_TOTAL_ACCOUNTS[0]}"})
            cleanFace()
            return res
        elif GLOBAL_TOTAL_ACCOUNTS[0] == 0 and GLOBAL_ACCEPT_ACCOUNTS[0] != 0:
            return jsonify({"status": 1, "message": "定时任务频繁"})
        else:
            cleanFace()
            print(" FAILED ")
            return jsonify({"status": -1, "message": "人脸认证失败"})

    def doJob(self):
        schedule.every(10).seconds.do(self.return_detect_result)


api.add_resource(AntiCheatingApp, '/face/detect/result')
