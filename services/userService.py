import logging

from resources.USERDB import User
from resources import db


def getUserById(UserId):
    user = db.session.get(User, UserId)
    if user is not None:
        return user
    else:
        logging.error('User does not exist')
        return User()


