from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np
import os

import keras
from keras import Model
from keras_segmentation.data_utils.data_loader import get_image_array
from keras_segmentation.models._pspnet_2 import Interp
from keras_segmentation.models.mobilenet import relu6
from keras_segmentation.predict import model_from_checkpoint_path

from hegemax_model import get_hegemax_model
from utils.pspnet import model_from_checkpoint_path as custom_model_from_checkpoint_path, parse_depth_pred

from util import get_model_params

import tensorflow.python.keras.losses
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class ModelInterface(ABC):

    @abstractmethod
    def get_prediction(self, images, info):
        pass


def encoder(x, angle):
    return np.sin(((2 * np.pi * (x - 1)) / (9)) - ((angle * np.pi) / (2 * 1)))


class SegmentationModel:
    def __init__(self, model_path):
        checkpoints_path = "seg_nets/" + model_path
        self.model = custom_model_from_checkpoint_path(checkpoints_path)
        # self.model = model_from_checkpoint_path(checkpoints_path)
        f, [self.ax1, self.ax2] = plt.subplots(2, 1)
        self.im = None
        self.im2 = None
        self.model.summary()

    def plot_segmentation(self, images):
        seg_prediction, depth_prediction = self.model.predict_segmentation(images["forward_center_rgb"])
        (width, height, _) = images["forward_center_rgb"].shape
        seg_prediction = cv2.resize(seg_prediction.astype('float32'), dsize=(height, width), interpolation=cv2.INTER_NEAREST)
        depth_prediction = cv2.resize((parse_depth_pred(depth_prediction) / 255).astype('float32'), dsize=(height, width), interpolation=cv2.INTER_NEAREST)
        if self.im is None:
            self.im = self.ax1.imshow(seg_prediction)
            # self.im = plt.imshow(np.concatenate(seg_prediction, depth_prediction))
        else:
            self.im.set_data(seg_prediction)
            # self.im.set_data(np.concatenate(seg_prediction, depth_prediction))

        if self.im2 is None:
            self.im2 = self.ax2.imshow(depth_prediction)
            # self.im = plt.imshow(np.concatenate(seg_prediction, depth_prediction))
        else:
            self.im2.set_data(depth_prediction)
            # self.im.set_data(np.concatenate(seg_prediction, depth_prediction))

        plt.pause(0.00001)
        plt.draw()


class LSTMKeras(ModelInterface):
    def __init__(self, path, sampling_interval, capture_rate=3):
        self._model = None  # type: Optional[keras.models.Model]

        # Initialize network input history
        self._img_center_history = []
        self._img_left_history = []
        self._img_right_history = []
        self._info_history = []
        self._hlc_history = []
        self._steer_history = []
        self._environment_history = []

        # Network parameters
        self._sampling_interval = sampling_interval + capture_rate - 1
        # Uncomment for legacy models
        # self.hlc_one_hot = {1: [1, 0, 0, 0, 0, 0], 2: [0, 1, 0, 0, 0, 0], 3: [0, 0, 1, 0, 0, 0], 4: [0, 0, 0, 1, 0, 0], 5: [0, 0, 0, 0],
        #                     6: [0, 0, 0, 0]}
        # Normal
        self.hlc_one_hot = {1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 1, 0], 4: [0, 0, 0, 1], 5: [0, 0, 0, 0],
                            6: [0, 0, 0, 0]}
        # SPURV
        # self.hlc_one_hot = {1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 0, 1], 4: [0, 0, 0, 1], 5: [0, 0, 0, 0],
        #                     6: [0, 0, 0, 0]}
        self.environment_one_hot = {0: [1, 0], 1: [0, 1]}

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
        self._steer_history = []
        self._environment_history = []

    def _load_model(self, path: str):
        # Uncomment for legacy models
        # model: tensorflow.python.keras.Model = get_hegemax_model(1, True)
        # model.load_weights(path)
        # self._model = model
        self._model = keras.models.load_model(path, compile=False, custom_objects={'Interp': Interp, 'relu6': relu6})
        (self.height, self.width, self.channels), self.sequence_length, self.sine_steering = get_model_params(
            self._model)
        print("Image shape: " + str((self.height, self.width, self.channels)) + ", sequence length: " + str(
            self.sequence_length) + ", sine steering? " + str(self.sine_steering))

    def restart(self):
        self._img_center_history = []
        self._img_left_history = []
        self._img_right_history = []
        self._info_history = []
        self._hlc_history = []
        self._steer_history = []
        self._environment_history = []
        self.loaded_at = time.time()
        self._frame = 0
        self._last_pred = None
        print("Restart")

    def get_prediction(self, images, info):

        self._frame += 1

        if self._model is None:
            return False
        req = (self.sequence_length - 1) * (self._sampling_interval + 1) + 1
        # Uncomment for legacy models
        # img_center = cv2.cvtColor(cv2.resize(images["forward_center_rgb"], (self.width, self.height)), cv2.COLOR_BGR2LAB)
        # info_input = [
        #     max(float(info["speed"] * 3.6 / 100), 0.2),
        #     float(info["speed_limit"] * 3.6 / 100),
        #     info["traffic_light"]
        # ]

        img_center = get_image_array(images["forward_center_rgb"], height=self.height, width=self.width, imgNorm="sub_mean",
                                     ordering='channels_last')
        info_input = [
            float(info["speed"] * 3.6 / 30 - 1),
            float(info["speed_limit"] * 3.6 / 30 - 1),
            info["traffic_light"]
        ]

        hlc_input = self.hlc_one_hot[info["hlc"].value]
        environment_input = self.environment_one_hot[info["environment"].value]

        self._img_center_history.append(np.array(img_center))
        """self._img_left_history.append(np.array(img_left))
        self._img_right_history.append(np.array(img_right))"""
        self._info_history.append(np.array(info_input))
        self._hlc_history.append(np.array(hlc_input))
        self._steer_history.append(np.array([self._last_pred[0] if self._last_pred is not None else 0.0]))
        self._environment_history.append(np.array(environment_input))

        if len(self._img_center_history) > req:
            self._img_center_history.pop(0)
            """self._img_left_history.pop(0)
            self._img_right_history.pop(0)"""
            self._info_history.pop(0)
            self._hlc_history.pop(0)
            self._steer_history.pop(0)
            self._environment_history.pop(0)

        if len(self._img_center_history) == req:
            imgs_center = np.array([self._img_center_history[0::self._sampling_interval + 1]])
            """imgs_left = np.array([self._img_left_history[0::self._sampling_interval + 1]])
            imgs_right = np.array([self._img_right_history[0::self._sampling_interval + 1]])"""

            infos = np.array([self._info_history[0::self._sampling_interval + 1]])
            hlcs = np.array([self._hlc_history[0::self._sampling_interval + 1]])
            last_steers = np.array([self._steer_history[0::self._sampling_interval + 1]])
            environments = np.array([self._environment_history[0::self._sampling_interval + 1]])

            prediction = self._model.predict({
                "forward_image_input": imgs_center,
                "info_input": infos,
                "hlc_input": hlcs,
                "environment_input": environments,
                "prev_steer_input": last_steers
            })

            steer = prediction[0][0]
            steer_angle = steer[0]
            if self.sine_steering:
                steer_curve_parameters = curve_fit(encoder, np.arange(1, 11, 1), steer)[0]
                steer_angle = steer_curve_parameters[0]
            else:
                # pass
                steer_angle = steer_angle * 2 - 1

            # Target speed
            if len(prediction) == 2:
                target_speed = prediction[1][0][0] * 100.0
                if target_speed < 4:
                    target_speed = 0
                self._last_pred = (steer_angle, 0, 0, target_speed)
                print("Steer: ", steer_angle, ", target speed: ", target_speed)
            else:
                throttle = prediction[1][0][0]
                if len(prediction) == 3:
                    brake = prediction[2][0][0]
                else:
                    brake = 0

                self.brake_hist.append(brake)
                avg_brake = np.max(self.brake_hist)
                step_brake = 1 if avg_brake > 0.3 else 0
                #step_brake = brake  # avg_brake  # 1 if avg_brake > 0.5 else 0
                #step_brake = avg_brake  # 1 if avg_brake > 0.5 else 0

                if len(self.brake_hist) > 7:
                    self.brake_hist.pop(0)

                self._last_pred = (steer_angle, throttle, step_brake, None)
                print("Steer: ", steer_angle, ", throttle: ", throttle, ", brake: ", brake)

            return self._last_pred

        return 0, 0.5, 0, None


class CNNKeras(ModelInterface):
    def __init__(self, path):
        self._model = None  # type: Optional[keras.models.Model]

        self.hlc_one_hot = {1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 1, 0], 4: [0, 0, 0, 1]}
        self.environment_one_hot = {0: [1, 0], 1: [0, 1]}

        self.loaded_at = time.time()
        self.brake_hist = []

        # Load model
        self._load_model(path)

        self._frame = 0

        self._last_pred = None

    def _load_model(self, path):
        self._model = keras.models.load_model(path, compile=False, custom_objects={'Interp': Interp, 'relu6': relu6})
        (self.height, self.width, self.channels), _, self.sine_steering = get_model_params(self._model)
        self._model.summary()
        x = self._model.layers[-7].get_output_at(0)
        self.cut_model = Model(inputs=self._model.layers[-7].get_input_at(0), outputs=x)

    def restart(self):
        self.loaded_at = time.time()
        self._frame = 0
        self._last_pred = None
        print("Restart")

    def get_prediction(self, images, info):

        self._frame += 1

        if self._model is None:
            return False

        img_center = get_image_array(images["forward_center_rgb"], self.height, self.width, imgNorm="sub_mean",
                                     ordering='channels_last')
        """img_left = cv2.cvtColor(images["left_center_rgb"], cv2.COLOR_BGR2LAB)
        img_right = cv2.cvtColor(images["right_center_rgb"], cv2.COLOR_BGR2LAB)"""
        info_input = [
            max(float(info["speed"] * 3.6 / 100), 0.2),
            float(info["speed_limit"] * 3.6 / 100),
            info["traffic_light"]
        ]
        hlc_input = self.hlc_one_hot[(info["hlc"].value)]
        environment_input = self.environment_one_hot[(info["environment"].value)]

        prediction = self._model.predict({
            "forward_image_input": np.array([img_center]),
            "info_input": np.array([info_input]),
            "hlc_input": np.array([hlc_input]),
            "environment_input": np.array([environment_input])
        })

        prediction_cut = self.cut_model(np.array(img_center))
        print("Cut: ", prediction_cut)

        steer, throttle, brake = prediction[0][0], prediction[1][0][0], 0
        self.brake_hist.append(brake)
        avg_brake = np.max(self.brake_hist)
        step_brake = 1 if avg_brake > 0.5 else 0

        if len(self.brake_hist) > 7:
            self.brake_hist.pop(0)

        if self.sine_steering:
            steer_curve_parameters = curve_fit(encoder, np.arange(1, 11, 1), steer)[0]
            steer_angle = steer_curve_parameters[0]
            self._last_pred = (steer_angle, throttle, step_brake)
            print("(Sine-)Steer: ", steer_angle, ", throttle: ", throttle)
        else:
            steer = steer[0]
            self._last_pred = (steer, throttle, step_brake)
            print("Steer: ", steer, ", throttle: ", throttle)

        return self._last_pred
