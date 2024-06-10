from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Boolean
from resources import db


class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(Integer, primary_key=True, comment='id')
    userAccount = db.Column(String(256), nullable=False, comment='账号')
    userPassword = db.Column(String(512), nullable=False, comment='密码')
    unionId = db.Column(String(256), comment='微信开放平台id')
    mpOpenId = db.Column(String(256), comment='公众号openId')
    userName = db.Column(String(256), comment='用户昵称')
    userAvatar = db.Column(String(1024), comment='用户头像')
    userProfile = db.Column(String(512), comment='用户简介')
    userRole = db.Column(String(256), nullable=False, default='user', comment='用户角色：user/admin/ban')
    isRegisterFace = db.Column(Boolean, nullable=False, default=0, comment='是否人脸认证')
    FacePicLocalPath = db.Column(String(256), comment='人脸图片存储路径')
    FeaturesLocalPath = db.Column(String(256), comment='人脸特征csv存储路径')
    createTime = db.Column(DateTime, nullable=False, default=datetime.utcnow, comment='创建时间')
    updateTime = db.Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow,
                           comment='更新时间')
    isDelete = db.Column(Boolean, nullable=False, default=0, comment='是否删除')

    def __repr__(self):
        return f'<User {self.userName}>'
