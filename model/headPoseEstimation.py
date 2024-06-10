import logging
import math

import cv2
import mediapipe as mp
import numpy as np
from flask import jsonify

class Head_Pose:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)
    def rotation_matrix_to_angles(self, rotation_matrix):
        """
        Calculate Euler angles from rotation matrix.
        :param rotation_matrix: A 3*3 matrix with the following structure
        [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
        [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
        [  -Siny             CosySinx                   Cosy*Cosx         ]
        :return: Angles in degrees for each axis
        """
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                         rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([x, y, z]) * 180. / math.pi

    def getHeadPose(self, frame):
        # Convert the color space from BGR to RGB and get Mediapipe results
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        # Convert the color space from RGB to BGR to display well with Opencv
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face_coordination_in_real_world = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)

        h, w, _ = image.shape
        face_coordination_in_image = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 9, 57, 130, 287, 359]:
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_coordination_in_image.append([x, y])

                face_coordination_in_image = np.array(face_coordination_in_image,
                                                      dtype=np.float64)

                # The camera matrix
                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, w / 2],
                                       [0, focal_length, h / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Use solvePnP function to get rotation vector
                success, rotation_vec, transition_vec = cv2.solvePnP(
                    face_coordination_in_real_world, face_coordination_in_image,
                    cam_matrix, dist_matrix)

                # Use Rodrigues function to convert rotation vector to matrix
                rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

                result = self.rotation_matrix_to_angles(rotation_matrix)
                if result is not None:
                    print("HeadPose: ", result)
                    return jsonify({'status': 1, 'P': result[0], 'Y': result[1], 'R': result[2]})
                else:
                    print("HeadPose: Failed")
                    return jsonify({'status': -1})
            return jsonify({'status': 1, 'P': 0, 'Y': 0, 'R': 0})
        return jsonify({'status': 1, 'P': 0, 'Y': 0, 'R': 0})