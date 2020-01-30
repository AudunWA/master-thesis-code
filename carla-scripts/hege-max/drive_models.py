from abc import ABC, abstractmethod
import cv2
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.backend import set_session
import tensorflow.keras.losses
import re
import time
from scipy.optimize import curve_fit

class ModelInterface(ABC):

    @abstractmethod
    def get_prediction(self, images, info):
        pass


def encoder(x, angle):
    return np.sin(((2*np.pi*(x-1))/(9))-((angle*np.pi)/(2*1)))


class LSTMKeras(ModelInterface):
    def __init__(self, path, seq_length, sampling_interval, capture_rate=3):
        self._model = None
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        sess = tf.Session(config=config)
        set_session(sess)

        # Initialize network input history
        self._img_center_history = []
        self._img_left_history = []
        self._img_right_history = []
        self._info_history = [] 
        self._hlc_history = []
        self._environment_history = []
        
        # Network parameters 
        self._seq_length = seq_length
        self._sampling_interval = sampling_interval + capture_rate - 1
        self.hlc_one_hot = { 1: [1,0,0,0,0,0], 2:[0,1,0,0,0,0], 3:[0,0,1,0,0,0], 4:[0,0,0,1,0,0], 5:[0,0,0,0,1,0], 6:[0,0,0,0,0,1]}
        self.environment_one_hot = { 0: [1,0], 1:[0,1]}

        self.loaded_at = time.time()
        self.brake_hist = []

        # Load model
        self._load_model(path)

        self._frame = 0
        
        self._last_pred = None

    def _init_history(self):
        self._img_center_history = []
        self._img_left_history = []
        self._img_right_history = []
        self._info_history = [] 
        self._hlc_history = []
        self._environment_history = []
    
    def _load_model(self, path):
        self._model = tf.keras.models.load_model(path, compile=False)

        
    def restart(self):
        self._img_center_history = []
        self._img_left_history = []
        self._img_right_history = []
        self._info_history = [] 
        self._hlc_history = []
        self._environment_history = []
        self.loaded_at = time.time()
        self._frame = 0
        self._last_pred = None
        print("Restart")

    def get_prediction(self, images, info):
  
        self._frame += 1
        
        if self._model is None:
            return False
        req = (self._seq_length - 1) * (self._sampling_interval + 1) + 1

        img_center = cv2.cvtColor(images["forward_center_rgb"], cv2.COLOR_BGR2LAB)
        """img_left = cv2.cvtColor(images["left_center_rgb"], cv2.COLOR_BGR2LAB)
        img_right = cv2.cvtColor(images["right_center_rgb"], cv2.COLOR_BGR2LAB)"""
        info_input = [
            max(float(info["speed"] * 3.6 / 100),0.2 ),
            float(info["speed_limit"] * 3.6 / 100),
            info["traffic_light"]
        ]
        hlc_input = self.hlc_one_hot[(info["hlc"].value)]
        environment_input = self.environment_one_hot[(info["environment"].value)]

        self._img_center_history.append(np.array(img_center))
        """self._img_left_history.append(np.array(img_left))
        self._img_right_history.append(np.array(img_right))"""
        self._info_history.append(np.array(info_input))
        self._hlc_history.append(np.array(hlc_input))
        self._environment_history.append(np.array(environment_input))

        sinus = True
        
        if len(self._img_center_history) > req:
            self._img_center_history.pop(0)
            """self._img_left_history.pop(0)
            self._img_right_history.pop(0)"""
            self._info_history.pop(0)
            self._hlc_history.pop(0)
            self._environment_history.pop(0)
                
        if len(self._img_center_history) == req:
            imgs_center = np.array([self._img_center_history[0::self._sampling_interval + 1]])
            """imgs_left = np.array([self._img_left_history[0::self._sampling_interval + 1]])
            imgs_right = np.array([self._img_right_history[0::self._sampling_interval + 1]])"""

            infos = np.array([self._info_history[0::self._sampling_interval + 1]])
            hlcs = np.array([self._hlc_history[0::self._sampling_interval + 1]])
            environments = np.array([self._environment_history[0::self._sampling_interval + 1]])

            prediction = self._model.predict({
                "forward_image_input": imgs_center,
                "info_input": infos,
                "hlc_input": hlcs,
                "environment_input": environments
            })

            """if info["hlc"].value == 4:
                prediction = prediction[0]
            elif info["hlc"].value == 5:
                prediction = prediction[1]
            elif info["hlc"].value == 6:
                prediction = prediction[2]"""


            """steer, acc = prediction[0][0], prediction[1][0]

            if sinus:
                steer_curve_parameters = curve_fit(encoder, np.arange(1, 11, 1), steer)[0]
                steer_angle = steer_curve_parameters[0]

            brake = 1 if acc < -0.1 else 0
            throttle = 0.5 if acc > 0 else 0

        
            print(acc)

            return (steer_angle, throttle, brake)"""

            # print(brake)
            steer, throttle, brake = prediction[0][0], prediction[1][0], prediction[2][0]
            self.brake_hist.append(brake)
            if len(self.brake_hist)>7:
                self.brake_hist.pop(0)
            if sinus:
                steer_curve_parameters = curve_fit(encoder, np.arange(1, 11, 1), steer)[0]
                steer_angle = steer_curve_parameters[0]

            avg_brake = np.max(self.brake_hist)
            step_brake = 1 if avg_brake > 0.5 else 0

            """if self._frame % 30 != 0 and self._last_pred:
                return self._last_pred"""

            self._last_pred  = (steer_angle, throttle, step_brake) if sinus else (steer, throttle, step_brake) 
            return self._last_pred
    
        
        return (0, 0.5, 0)