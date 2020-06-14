## Imports
from glob import glob

import configparser
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
from ast import literal_eval
from distutils.util import strtobool
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import load_model
from keras.layers import Input, Flatten, Dense, concatenate, TimeDistributed, CuDNNLSTM, \
    Lambda, BatchNormalization, Activation, Conv2D, ZeroPadding2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import Sequence, plot_model
from keras_segmentation.data_utils.data_loader import get_image_array
from keras_segmentation.models._pspnet_2 import Interp
from keras_segmentation.models.mobilenet import relu6
from keras_segmentation.predict import model_from_checkpoint_path
from keras_segmentation.pretrained import pspnet_101_cityscapes
from pathlib import Path
from typing import Tuple, List

from prediction_pipeline.unet_depth_segm import unet_model_from_checkpoint_path
from prediction_pipeline.utils.hegemax_augment import change_light, change_hue, add_rain, add_shadow, gaussian_blur

three_class_checkpoints_path = "data/segmentation_models/pspnet_3_classes/pspnet_3_classes"
seven_class_checkpoints_path = "data/segmentation_models/resnet50_pspnet_8_classes/2020-02-26_17-05-57/resnet50_pspnet_8_classes"
seven_class_mobile_checkpoints_path = "data/segmentation_models/mobilenet_eight/mobilenet_eight"
seven_class_vanilla_psp_path = "data/segmentation_models/pspnet_8_classes/2020-02-14_14-09-24/pspnet_8_classes"
seven_class_vanilla_psp_depth_path = "data/segmentation_models/pspnet_8_classes/2020-02-26_12-00-11/pspnet_8_classes"
resnet50_pspnet_8_classes = "data/segmentation_models/resnet50_pspnet_8_classes/2020-02-26_17-05-57/resnet50_pspnet_8_classes"
resnet50_pspnet_3_classes = "/hdd/audun/master-thesis-code/training/model_checkpoints/pspnet_checkpoints_best/pspnet_50_three"
mobilenet_unet_5_classes_augmentFalse = "data/segmentation_models/mobilenet_unet_5_classes_augmentFalse/2020-04-18_02-52-11/mobilenet_unet_5_classes_augmentFalse"
mobilenet_unet_5_classes_augment_true_combined_data_false = "data/segmentation_models/mobilenet_unet_5_classes_augment_true_combined_data_false/2020-04-20_18-56-35/mobilenet_unet_5_classes_augment_true_combined_data_false"
mobilenet_unet_5_classes_augment_true_combined_data_true = "data/segmentation_models/mobilenet_unet_5_classes_augment_true_combined_data_true/2020-04-21_14-00-05/mobilenet_unet_5_classes_augment_true_combined_data_true"
mobilenet_unet_5_classes_augment_true_data_carla = "data/segmentation_models/mobilenet_unet_5_classes_augment_true_data_carla/2020-04-23_15-08-04/mobilenet_unet_5_classes_augment_true_data_carla"
mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm = "data/segmentation_models/mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm/2020-04-27_16-27-21/mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm"
mobilenet_unet_depth_segm_5_classes_carla_only_depth_segm_depth_w05 = "data/segmentation_models/mobilenet_unet_depth_segm_5_classes_carla_only_depth_segm_depth_w0.5/2020-05-01_15-06-39/mobilenet_unet_depth_segm_5_classes_carla_only_depth_segm_depth_w0.5"
mapillary_carla_segm_depth = "data/segmentation_models/mobilenet_unet_depth_segm_5_classes_carla_mapillary_depth_segm/2020-04-28_10-12-38/mobilenet_unet_depth_segm_5_classes_carla_mapillary_depth_segm"

import tensorflow as tf
import keras.backend as K

# import pydotplus

print(tf.__version__)
print(cv2.__version__)
print(os.environ["CONDA_DEFAULT_ENV"])

## Helper functions
# Left, right, straight, follow lane, change lane left, change lane right
# We map change lane left to left, and change lane right to right
hlc_one_hot = {1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 1, 0], 4: [0, 0, 0, 1], 5: [1, 0, 0, 0], 6: [0, 1, 0, 0]}


def flatten_list(items_list):
    """ Removes the episode dimension from a list of data points """
    ret = []
    for items in items_list:
        for item in items:
            ret.append(item)
    return ret


# Custom loss function
def weighted_mse(y_true, y_pred, weight_mask):
    """
    Custom loss fucntion, different weigted loss to steer/throttle
    Used when all outputs is evaluated by same loss function
    """
    return K.mean(K.square(y_pred - y_true) * weight_mask, axis=-1)


def root_mean_squared_error(y_true, y_pred):
    """ Custom loss function, RMSE """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def mean_squared_error(y_true, y_pred):
    """ Custom loss function, RMSE """
    return K.mean(K.square(y_pred - y_true))


def steer_loss():
    """ Loss function for steering, RMSE """

    def custom(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)

    return custom


def get_hlc_one_hot(hlc):
    """ One-hot encode HLC values """
    return hlc_one_hot[hlc]


def translate_spurv_hlc_to_one_hot(hlc):
    """SPURV training data has different HLCs"""
    if hlc == 0:  # left
        return hlc_one_hot[1]
    elif hlc == 1:
        return hlc_one_hot[4]  # follow lane
    elif hlc == 2:
        return hlc_one_hot[2]  # right


def sine_encode(angle):
    """ Encode steering angle as sine wave """
    angle_max = 1
    N = 10
    ret = []

    for i in range(1, N + 1, 1):
        Y = np.sin(((2 * np.pi * (i - 1)) / (N - 1)) - ((angle * np.pi) / (2 * angle_max)))
        ret.append(Y)

    return np.array(ret)


def create_input_dict():
    return {
        "forward_imgs": [],
        "info_signals": [],
        "hlcs": [],
    }


def create_target_dict():
    return {
        "steer": [],
        "target_speed": [],
    }


def split_dict(dictionary, split_pos):
    """ Split data into training and validation """
    train_dict = {}
    val_dict = {}

    for key in dictionary:
        train_dict[key] = dictionary[key][:split_pos]
        val_dict[key] = dictionary[key][split_pos:]

    return train_dict, val_dict


def adjust_hlcs(hlcs, info_signals):
    # Ensure same results for same input
    random.seed(123)
    """
    Randomly adjust HLC backwards,
    such that the HLC data is given before the car is in the intersection
    """

    # Iterate over all episodes
    for e in range(len(hlcs)):
        active_change_hlc = None
        changed = 0
        adjust_num = random.randint(20, 40)
        # Iterate over all hlcs in episode
        for i in range(len(hlcs[e]) - 2, -1, -1):
            next_hlc = np.argmax(hlcs[e][i + 1]) + 1
            current_hlc = np.argmax(hlcs[e][i]) + 1
            if active_change_hlc == False:
                active_change_hlc = None
                continue

            if active_change_hlc is None:
                if next_hlc == 1 or next_hlc == 2 or next_hlc == 3:
                    if current_hlc == 4:
                        active_change_hlc = hlcs[e][i + 1]
            if active_change_hlc is not None:
                hlcs[e][i] = active_change_hlc
                if info_signals[e][i][2] != 0:
                    changed += 1
                if changed == adjust_num:
                    active_change_hlc = False
                    changed = 0
    return hlcs


#### BALANCING DATA - Helper functions ####
def get_hlc_dist_label(hlc):
    " Get distributions of HLC in data "
    if hlc == 1:
        return "left"
    elif hlc == 2:
        return "right"
    elif hlc == 3:
        return "straight"
    elif hlc == 4:
        return "follow_lane"
    elif hlc == 5:
        return "change_lane_left"
    elif hlc == 6:
        return "change_lane_right"


def get_speed_dist_label(speed):
    """ Get distributions of speed in data """

    if speed < 0.001:
        return "low_speed"
    else:
        return "high_speed"


def get_speed_limit_dist_label(speed_limit):
    """ Get distributions of speed limits in data """

    speed_limit = round(speed_limit, 1)
    if speed_limit == 0.3:
        return "30km/h"
    elif speed_limit == 0.6:
        return "60km/h"
    elif speed_limit == 0.9:
        return "90km/h"


def get_percentage_dist(dist):
    """ Get distribution of data in percentage """
    N = dist["traffic_light"]["red"] + dist["traffic_light"]["green"]
    for dist_key in dist:
        for key, dist_val in dist[dist_key].items():
            dist[dist_key][key] = dist_val / N
    return dist


def get_drop_num(tot_num, num, keep_fraction):
    """ Calculates how many datasamples one should drop to get the right amount of data samples """
    return int((num - keep_fraction * tot_num) / (1 - keep_fraction))


def shuffle_data(inputs_flat, targets_flat):
    # Shuffle
    indices = np.arange(len(inputs_flat["forward_imgs"]))
    np.random.shuffle(indices)

    for key in inputs_flat:
        inputs_flat[key] = np.array(inputs_flat[key])[indices]

    for key in targets_flat:
        targets_flat[key] = np.array(targets_flat[key])[indices]

    return inputs_flat, targets_flat


# ## Load driving logs
# - Loads data from all csv file in a lits of folders
# - Stores the data in a input dictionary and a target dictionary
# - Normalizes and coverts data to correct format
# - Does not use side cameraes for lane change data
#

import random


def get_path(episode_path, image_path):
    return str(episode_path / Path("imgs") / image_path.split("/")[-1])


def load_driving_logs(dataset_folders: List[str], use_side_cameras: bool, steering_correction: float):
    """
    input:
        dataset_folders: list of paths to folders to load data from
        steering_correction: float, adjusts steering angles of forward facing side cameras

    Loads data from all csv file in a lits of folders
    Stores the data in a input dictionary and a target dictionary
    Normalizes and coverts data to correct format
    Does not use side cameraes for lane change data

    return:
        inputs: dictionary of all input data (image paths, info signals, HLCs, control signals)
        targets: dictionary of all target data (steer, throttle)
    """
    inputs = create_input_dict()

    targets = create_target_dict()
    # Loads data
    for folder in dataset_folders:
        folder_path = Path("data/carla_data") / folder
        for episode in glob(str(folder_path / "*")):

            temp_forward = {"center": [], "left": [], "right": []}
            temp_hlcs = {"center": [], "left": [], "right": []}
            temp_info_signals = {"center": [], "left": [], "right": []}

            temp_steer = {"center": [], "left": [], "right": []}
            temp_target_speed = {"center": [], "left": [], "right": []}

            episode_path = Path(episode)

            use_side_cameras_here = use_side_cameras
            df = pd.read_csv(str(episode_path / "driving_log.csv"))
            for index, row in df.iterrows():
                if index == 0:
                    continue

                if row.get("speed") is not None:
                    # Normalize max speed (60km/h) between -1 and 1
                    speed = row["speed"] * 3.6 / 30 - 1
                    speed_limit = (row["speed"] * 3.6 / 30) / 0.75 - 1  # Mimic speed limit
                    steer = -row["angle"]  # SPURV angle is opposite of CARLA
                    target_speed = row["speed"] * 3.6 / 100
                    traffic_light = 1 if row["speed"] > 0.01 else 0
                    if df["high_level_command"].min() == 0 or df["high_level_command"].max() == 2:
                        # Old format, convert
                        hlc = translate_spurv_hlc_to_one_hot(row["high_level_command"])
                    else:
                        hlc = get_hlc_one_hot(row["high_level_command"])

                    temp_forward["center"].append(get_path(episode_path, row["image_path"]))
                    use_side_cameras_here = False  # Explicitly state that we don't use side cameras

                else:
                    [_, steer, _] = literal_eval(row["ClientAutopilotControls"])
                    # Normalize max speed (60km/h) between -1 and 1
                    speed = float(row["Velocity"]) * 3.6 / 30 - 1
                    speed_limit = float(row["SpeedLimit"]) * 3.6 / 30 - 1
                    traffic_light = int(row["NewTrafficLight"])
                    target_speed = float(row["APTargetSpeed"]) / 100.0  # percentage of 100 km/h

                    hlc = row["HLC"]
                    if hlc == 0 or hlc == -1:
                        hlc = 4
                    hlc = get_hlc_one_hot(hlc)
                    temp_forward["center"].append(get_path(episode_path, row["ForwardCenter"]))

                temp_steer["center"].append(steer)
                temp_target_speed["center"].append(target_speed)
                temp_info_signals["center"].append([speed, speed_limit, traffic_light])
                temp_hlcs["center"].append(hlc)

                if use_side_cameras_here:
                    # Only use the left/right shifted images for non lange-change data
                    # if row["HLC"] != 5 and row["HLC"] != 6:
                    temp_forward["left"].append(get_path(episode_path, row["ForwardLeft"]))
                    temp_forward["right"].append(get_path(episode_path, row["ForwardRight"]))

                    temp_target_speed["left"].append(target_speed)
                    temp_target_speed["right"].append(target_speed)

                    temp_steer["left"].append(steer + steering_correction)
                    temp_steer["right"].append(steer - steering_correction)

                    temp_info_signals["left"].append([speed, speed_limit, traffic_light])
                    temp_info_signals["right"].append([speed, speed_limit, traffic_light])

                    temp_hlcs["left"].append(hlc)
                    temp_hlcs["right"].append(hlc)

            inputs["forward_imgs"].append(temp_forward["center"])
            inputs["info_signals"].append(temp_info_signals["center"])
            inputs["hlcs"].append(temp_hlcs["center"])
            targets["steer"].append(temp_steer["center"])
            targets["target_speed"].append(temp_target_speed["center"])

            if use_side_cameras_here:
                inputs["forward_imgs"].append(temp_forward["left"])
                inputs["forward_imgs"].append(temp_forward["right"])
                inputs["info_signals"].append(temp_info_signals["left"])
                inputs["info_signals"].append(temp_info_signals["right"])
                inputs["hlcs"].append(temp_hlcs["left"])
                inputs["hlcs"].append(temp_hlcs["right"])
                targets["steer"].append(temp_steer["left"])
                targets["steer"].append(temp_steer["right"])
                targets["target_speed"].append(temp_target_speed["left"])
                targets["target_speed"].append(temp_target_speed["right"])

    print("Done, {:d} episode(s) loaded.".format(len(inputs["forward_imgs"])))
    # if use_side_cameras:
    #     print("Using side cameras as well, real episode count is ", len(inputs["forward_imgs"]) / 3)

    return (inputs, targets)


# ## Plot Data


def plot_data(dist, title=""):
    """ Plots distribution of HLC, speed and traffic lights """
    print("Plotting...")
    tot_num = 0
    for key in dist["hlc"]:
        tot_num += dist["hlc"][key]

    fig = plt.figure(figsize=(16, 6))

    # HLC
    labels = ["Left", "Right", "Straight", "Follow lane"]
    sizes = [dist["hlc"]["left"], dist["hlc"]["right"], dist["hlc"]["straight"], dist["hlc"]["follow_lane"]]

    ax1 = fig.add_subplot(1, 4, 1)
    wedges, texts, autotexts = ax1.pie(sizes, autopct='%.1f%%')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax1.set_title("Distrubution of HLC")
    ax1.legend(wedges, labels, loc="best")
    ax1.axis('equal')

    # Speed
    labels = ["Low Speed", "High Speed"]
    sizes = [dist["speed"]["low"], dist["speed"]["high"]]

    ax2 = fig.add_subplot(1, 4, 2)
    wedges, texts, autotexts = ax2.pie(sizes, autopct='%.1f%%')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title("Distrubution of speed")
    ax2.legend(wedges, labels, loc="best")
    ax2.axis('equal')

    # Steering
    labels = ["Left", "Straight", "Right"]
    print("Dist: ", dist)
    sizes = [dist["steering"]["left"], dist["steering"]["straight"], dist["steering"]["right"]]

    ax2 = fig.add_subplot(1, 4, 3)
    wedges, texts, autotexts = ax2.pie(sizes, autopct='%.1f%%')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title("Distrubution of steering angle")
    ax2.legend(wedges, labels, loc="best")
    ax2.axis('equal')

    fig.suptitle(title + ": " + str(tot_num) + " sequences", fontsize=18)
    plt.show()


# ## Balance Data
#
# - Get distribution of HLC, speed, speed limit, and traffic lights
# - Balance data for LSTM

STEERING_ANGLE_THRESHOLD = 0.05


def get_dist(inputs, targets):
    """
        input:
            inputs: dictionary of input data
            targets: dictionary of result data

        return:
            dist: dictionary of distributions for HLC, speed, angle
    """
    dist = {
        "hlc": {
            "left": 0,
            "right": 0,
            "straight": 0,
            "follow_lane": 0,
        },
        "steering": {
            "left": 0,  # More than 20 deg left
            "right": 0,  # More than 20 deg right
            "straight": 0  # Between left and right
        },
        "speed": {
            "high": 0,  # More than 0.5m/s
            "low": 0  # Less than 0.5 m/s
        }

    }

    # print("getting dist", inputs, targets)
    # HLC distribution
    for hlcs in inputs["hlcs"]:

        # Iterate over all HLC in sequence
        for hlc in hlcs:
            hlc_value = np.argmax(hlc) + 1
            dist["hlc"][get_hlc_dist_label(hlc_value)] += 1

    # Speed distribution
    for speed in targets["target_speed"]:
        if speed > 30:
            dist["speed"]["high"] += 1
        else:
            dist["speed"]["low"] += 1

    # Steering distribution
    for angle in targets["steer"]:
        if angle < -STEERING_ANGLE_THRESHOLD:
            dist["steering"]["left"] += 1
        elif angle > STEERING_ANGLE_THRESHOLD:
            dist["steering"]["right"] += 1
        else:
            dist["steering"]["straight"] += 1

    return dist


def balance_steering_angle(inputs, targets, dist, target_straight_fraction):
    """ Balance steer angle such that target fraction is correct """

    # Find the steering with least amount of values
    least_vals = min(dist["steering"]["straight"], dist["steering"]["left"], dist["steering"]["right"])

    inputs_bal = create_input_dict()
    targets_bal = create_target_dict()

    left_count = 0
    right_count = 0
    forward_count = 0

    for i in range(len(inputs["hlcs"])):
        angle = targets["steer"][i]
        is_left = angle < -STEERING_ANGLE_THRESHOLD
        is_right = angle > STEERING_ANGLE_THRESHOLD

        if is_left and left_count >= least_vals:
            continue

        elif is_right and right_count >= least_vals:
            continue

        elif not is_left and not is_right and forward_count >= least_vals:
            continue

        if is_left:
            left_count += 1
        elif is_right:
            right_count += 1
        else:
            forward_count += 1

        # Keep
        for key in inputs_bal:
            inputs_bal[key].append(inputs[key][i])
        for key in targets_bal:
            targets_bal[key].append(targets[key][i])

    return inputs_bal, targets_bal


def balance_hlc(inputs, targets, dist):
    """ Balance HLCs such that target fraction is correct """

    # Find the steering with least amount of values
    least_vals = min(dist["hlc"]["straight"], dist["hlc"]["left"], dist["hlc"]["right"], dist["hlc"]["follow_lane"])
    # SPURV
    # least_vals = min(dist["hlc"]["left"], dist["hlc"]["right"], dist["hlc"]["follow_lane"])

    inputs_bal = create_input_dict()
    targets_bal = create_target_dict()

    left_count = 0
    right_count = 0
    straight_count = 0
    follow_lane_count = 0

    for i in range(len(inputs["hlcs"])):
        hlc = np.argmax(inputs["hlcs"][i]) + 1
        if hlc == 1:
            if left_count >= least_vals:
                continue
            left_count += 1
        elif hlc == 2:
            if right_count >= least_vals:
                continue
            right_count += 1
        elif hlc == 3:
            if straight_count >= least_vals:
                continue
            straight_count += 1
        elif hlc == 4:
            if follow_lane_count >= least_vals:
                continue
            follow_lane_count += 1

        # Keep
        for key in inputs_bal:
            inputs_bal[key].append(inputs[key][i])
        for key in targets_bal:
            targets_bal[key].append(targets[key][i])

    return inputs_bal, targets_bal


def balance_data_lstm(inputs, targets, straight_angle_frac=0.2):
    """ Balance dataset for LSTM data """
    print("Balancing data:")

    # Get distribution
    dist = get_dist(inputs, targets)
    print("dist: ", dist)

    inputs_bal = inputs
    targets_bal = targets
    dist_bal = dist

    # Balance steering angle
    # print("   - Balancing steering angle")
    # inputs_bal, targets_bal = balance_steering_angle(inputs_bal, targets_bal, dist_bal, straight_angle_frac)
    # dist_bal = get_dist(inputs_bal, targets_bal)
    # print("dist_bal: ", dist_bal)

    print("   - Balancing HLCs")
    inputs_bal, targets_bal = balance_hlc(inputs_bal, targets_bal, dist_bal)
    dist_bal = get_dist(inputs_bal, targets_bal)
    print("dist_bal: ", dist_bal)

    # Shuffle
    inputs_bal, targets_bal = shuffle_data(inputs_bal, targets_bal)

    dist_bal = get_dist(inputs_bal, targets_bal)

    return inputs_bal, targets_bal, dist_bal


# # LSTM

# ## Prepare dataset format

def get_episode_sequences(data, sampling_interval, seq_length):
    sequences = []
    slices = []
    for o in range(sampling_interval + 1):
        slices.append(data[o::sampling_interval + 1])
    for s in slices:
        for o in range(0, len(s)):
            if o + seq_length <= len(s):
                sequences.append(s[o:o + seq_length])
    return sequences


def prepare_dataset_lstm(inputs, targets, sampling_interval, seq_length):
    inputs_flat = create_input_dict()
    targets_flat = create_target_dict()

    for e in range(len(inputs["forward_imgs"])):
        [inputs_flat["forward_imgs"].append(sequence) for sequence in
         get_episode_sequences(inputs["forward_imgs"][e], sampling_interval, seq_length)]
        [inputs_flat["hlcs"].append(sequence) for sequence in
         get_episode_sequences(inputs["hlcs"][e], sampling_interval, seq_length)]
        [inputs_flat["info_signals"].append(sequence) for sequence in
         get_episode_sequences(inputs["info_signals"][e], sampling_interval, seq_length)]
        [targets_flat["steer"].append(sequence[-1]) for sequence in
         get_episode_sequences(targets["steer"][e], sampling_interval, seq_length)]
        [targets_flat["target_speed"].append(sequence[-1]) for sequence in
         get_episode_sequences(targets["target_speed"][e], sampling_interval, seq_length)]

    # Shuffle
    inputs_flat, targets_flat = shuffle_data(inputs_flat, targets_flat)

    return (inputs_flat, targets_flat)


# ## Define model

def get_layer_with_name(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer


def get_segmentation_model(model_type, freeze=True):
    x = None
    segmentation_model = None
    if model_type == "three_class_trained":
        from utils.pspnet import model_from_checkpoint_path as custom_model_from_checkpoint_path
        segmentation_model = custom_model_from_checkpoint_path(three_class_checkpoints_path)
        x = segmentation_model.get_layer("activation_10").output

    elif model_type == "seven_class_trained":
        segmentation_model = model_from_checkpoint_path(seven_class_checkpoints_path)

        x = segmentation_model.get_layer("conv2d_6").output


    elif model_type == "pretrained":
        segmentation_model = pspnet_101_cityscapes()
        x = segmentation_model.get_layer("conv6").output

    elif model_type == "seven_class_vanilla_psp":
        segmentation_model = model_from_checkpoint_path(seven_class_vanilla_psp_path)
        x = segmentation_model.get_layer("activation_10").output

    elif model_type == "seven_class_vanilla_psp_depth":
        from utils.pspnet import model_from_checkpoint_path as custom_model_from_checkpoint_path
        segmentation_model = custom_model_from_checkpoint_path(seven_class_vanilla_psp_depth_path)
        x = segmentation_model.get_layer("activation_10").output

    elif model_type == "seven_class_mobile":
        segmentation_model = model_from_checkpoint_path(seven_class_mobile_checkpoints_path)
        output_layer = get_layer_with_name(segmentation_model, "conv_dw_6_relu")

        x = output_layer.output

    elif model_type == "resnet50_pspnet_8_classes":
        segmentation_model = model_from_checkpoint_path(resnet50_pspnet_8_classes)
        output_layer = segmentation_model.get_layer(name="conv2d_6")
        x = output_layer.output
    elif model_type == "resnet50_pspnet_3_classes":
        segmentation_model = model_from_checkpoint_path(resnet50_pspnet_3_classes)
        output_layer = segmentation_model.get_layer(name="conv6")
        x = output_layer.output
    elif model_type == "mobilenet_unet_5_classes_augmentFalse":
        segmentation_model = model_from_checkpoint_path(mobilenet_unet_5_classes_augmentFalse)
        output_layer = segmentation_model.get_layer(name="conv2d_5")
        x = output_layer.output
    elif model_type == "mobilenet_unet_5_classes_augment_true_combined_data_true":
        segmentation_model = model_from_checkpoint_path(mobilenet_unet_5_classes_augment_true_combined_data_true)
        output_layer = segmentation_model.get_layer(name="conv2d_5")
        x = output_layer.output
    elif model_type == "mobilenet_unet_5_classes_augment_true_data_carla":
        segmentation_model = model_from_checkpoint_path(mobilenet_unet_5_classes_augment_true_data_carla)
        output_layer = segmentation_model.get_layer(name="conv2d_5")
        x = output_layer.output
    elif model_type == "mobilenet_unet_5_classes_augment_true_combined_data_false":
        segmentation_model = model_from_checkpoint_path(mobilenet_unet_5_classes_augment_true_combined_data_false)
        output_layer = segmentation_model.get_layer(name="conv2d_5")
        x = output_layer.output
    elif model_type == "mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm":
        segmentation_model = unet_model_from_checkpoint_path(mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm)

        # Concat segmentation and depth output
        segm_output_layer = segmentation_model.get_layer(name="conv2d_5").output
        depth_output_layer = segmentation_model.get_layer(name="conv2d_10").output
        x = concatenate([segm_output_layer, depth_output_layer], axis=3)
    elif model_type == "mobilenet_unet_depth_segm_5_classes_carla_only_depth_segm_depth_w05":
        segmentation_model = unet_model_from_checkpoint_path(
            mobilenet_unet_depth_segm_5_classes_carla_only_depth_segm_depth_w05)

        # Concat segmentation and depth output
        segm_output_layer = segmentation_model.get_layer(name="conv2d_5").output
        depth_output_layer = segmentation_model.get_layer(name="conv2d_10").output
        x = concatenate([segm_output_layer, depth_output_layer], axis=3)
    elif model_type == "mapillary_carla_segm_depth":
        segmentation_model = unet_model_from_checkpoint_path(
            mapillary_carla_segm_depth)

        # Concat segmentation and depth output
        segm_output_layer = segmentation_model.get_layer(name="conv2d_5").output
        depth_output_layer = segmentation_model.get_layer(name="conv2d_10").output
        x = concatenate([segm_output_layer, depth_output_layer], axis=3)
    elif model_type == "mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm_common_cut":
        segmentation_model = unet_model_from_checkpoint_path(mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm)
        output_layer = segmentation_model.get_layer(name="conv_pw_11_relu")
        x = output_layer.output
    elif model_type == "mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm_only_segm":
        segmentation_model = unet_model_from_checkpoint_path(mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm)
        output_layer = segmentation_model.get_layer(name="conv2d_5")
        x = output_layer.output

    plot_model(segmentation_model, "segmentation_model.png", show_shapes=True)
    # Explicitly define new model input and output by slicing out old model layers
    model_new = Model(inputs=segmentation_model.layers[0].input,
                      outputs=x)

    if freeze:
        for layer in model_new.layers:
            layer.trainable = False

    return model_new


def get_lstm_model(seq_length, freeze_segmentation, sine_steering=False, segm_model="seven_class_trained",
                   print_summary=True):
    hlc_input = Input(shape=(seq_length, 4), name="hlc_input")
    info_input = Input(shape=(seq_length, 3), name="info_input")

    segmentation_model = get_segmentation_model(segm_model, freeze_segmentation)
    [_, height, width, _] = segmentation_model.input.shape.dims
    forward_image_input = Input(shape=(seq_length, height.value, width.value, 3), name="forward_image_input")
    segmentation_output = TimeDistributed(segmentation_model)(forward_image_input)

    # Vanilla encoder
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    x = segmentation_output
    x = TimeDistributed(ZeroPadding2D((pad, pad)))(x)
    x = TimeDistributed(Conv2D(filter_size, (kernel, kernel),
                               padding='valid'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(MaxPooling2D((pool_size, pool_size)))(x)

    x = TimeDistributed(ZeroPadding2D((pad, pad)))(x)
    x = TimeDistributed(Conv2D(128, (kernel, kernel),
                               padding='valid'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(MaxPooling2D((pool_size, pool_size)))(x)

    for _ in range(3):
        x = TimeDistributed(ZeroPadding2D((pad, pad)))(x)
        x = TimeDistributed(Conv2D(256, (kernel, kernel), padding='valid'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D((pool_size, pool_size)))(x)
    segmentation_output = x

    segmentation_output = TimeDistributed(Flatten())(segmentation_output)

    x = concatenate([segmentation_output, hlc_input, info_input])

    x = TimeDistributed(Dense(100, activation="relu"))(x)
    x = concatenate([x, hlc_input])
    x = CuDNNLSTM(10, return_sequences=False)(x)
    hlc_latest = Lambda(lambda x: x[:, -1, :])(hlc_input)
    x = concatenate([x, hlc_latest])

    if sine_steering:
        steer_pred = Dense(10, activation="tanh", name="steer_pred")(x)
    else:
        steer_pred = Dense(1, activation="relu", name="steer_pred")(x)

    target_speed_pred = Dense(1, name="target_speed_pred", activation="sigmoid")(x)
    model = Model(inputs=[forward_image_input, hlc_input, info_input], outputs=[steer_pred, target_speed_pred])

    if print_summary:
        model.summary()

    return model


# # Training

# ## Define generator

class generator(Sequence):
    def __init__(self, inputs, targets, batch_size, img_width, img_height, validation=False, sine_steering=False,
                 augment=False):
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.validation = validation
        self.sine_steering = sine_steering
        self.augment = augment
        self.img_width = img_width
        self.img_height = img_height

        random.seed()
        # Convert to np array
        for key in inputs:
            inputs[key] = np.array(self.inputs[key])

        for key in targets:
            targets[key] = np.array(self.targets[key])

    def __len__(self):
        return int(np.ceil(len(self.inputs["forward_imgs"]) / float(self.batch_size)))

    def __getitem__(self, idx):
        subset = np.arange(idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.inputs["forward_imgs"])))
        forward_imgs = []
        read_images = [[cv2.imread(path) for path in seq] for seq in self.inputs["forward_imgs"][subset]]
        steer_pred = self.targets["steer"][subset]
        if self.sine_steering:
            steer_pred = np.array([sine_encode(steer) for steer in steer_pred])
        else:
            # pass
            steer_pred = (steer_pred + 1) / 2  # Normalize in [0,1]

        if self.validation:
            for seq in read_images:
                forward_imgs.append(
                    [get_image_array(image, width=self.img_width, height=self.img_height, imgNorm="sub_mean",
                                     ordering='channels_last') for
                     image in seq])
        else:
            if self.augment:
                augment = random.randint(0, 3)
                if augment == 0:
                    augment_type = random.randint(0, 4)

                    # Lightness
                    if augment_type == 0:
                        light_coeff = random.random() * 0.9 + 0.5
                        for seq in read_images:
                            forward_imgs.append(
                                [get_image_array(change_light(image, light_coeff=light_coeff), width=self.img_width,
                                                 height=self.img_height,
                                                 imgNorm="sub_mean", ordering='channels_last') for image in seq])
                    # Hue
                    elif augment_type == 1:
                        hue_coeff = random.random() * 1.4 + 0.2
                        for seq in read_images:
                            forward_imgs.append([get_image_array(change_hue(image, hue_coeff=hue_coeff),
                                                                 width=self.img_width, height=self.img_height,
                                                                 imgNorm="sub_mean",
                                                                 ordering='channels_last') for image in seq])
                    # Rain
                    elif augment_type == 2:
                        for seq in read_images:
                            forward_imgs.append(
                                [get_image_array(add_rain(image), width=self.img_width, height=self.img_height,
                                                 imgNorm="sub_mean", ordering='channels_last') for image
                                 in seq])

                    # Shadows
                    elif augment_type == 3:
                        for seq in read_images:
                            forward_imgs.append(
                                [get_image_array(add_shadow(image), width=self.img_width, height=self.img_height,
                                                 imgNorm="sub_mean", ordering='channels_last') for image
                                 in seq])
                    # Gaussian blur
                    elif augment_type == 4:
                        for seq in read_images:
                            forward_imgs.append(
                                [get_image_array(gaussian_blur(image), width=self.img_width, height=self.img_height,
                                                 imgNorm="sub_mean", ordering='channels_last') for image
                                 in seq])
                else:
                    for seq in read_images:
                        forward_imgs.append(
                            [get_image_array(image, width=self.img_width, height=self.img_height, imgNorm="sub_mean",
                                             ordering='channels_last')
                             for
                             image in seq])
            else:
                for seq in read_images:
                    forward_imgs.append(
                        [get_image_array(image, width=self.img_width, height=self.img_height, imgNorm="sub_mean",
                                         ordering='channels_last') for
                         image in seq])
        return {
                   "forward_image_input": np.array(forward_imgs),
                   "hlc_input": self.inputs["hlcs"][subset],
                   "info_input": self.inputs["info_signals"][subset],
               }, {
                   "steer_pred": steer_pred,
                   "target_speed_pred": self.targets["target_speed"][subset],
               }


def build_or_load_model(
        first_iteration: bool,
        model_name: str,
        seq_length: int,
        sampling_interval: int,
        sine_steering: bool,
        segmentation_model_name: str,
        use_side_cameras: bool,
        parameters_string: str,
        freeze_segmentation: bool,
        augment: bool
) -> Tuple[Model, Path, configparser.ConfigParser, int]:
    model_candidates = sorted(glob("data/driving_models/" + model_name + "/*/*.h5"), reverse=True)

    resume_training = 'n'
    if first_iteration and len(model_candidates) > 0:
        resume_training = input(
            "Found existing model(s) with same name. Do you want to continue the last session of " + model_name + "? (Y/n)").strip()

    if resume_training.lower() == '' or resume_training.lower() == 'y':
        model_path = model_candidates[0]
        model_folder = Path(os.path.dirname(model_path))
        initial_epoch = int(model_path.split("/")[-1].split('_')[0])  # 0-based in Keras, so no need to +1
        config = configparser.ConfigParser()
        config.read(model_folder / "config.ini")
        return load_model(model_path, compile=True,
                          custom_objects={"custom": root_mean_squared_error,
                                          "root_mean_squared_error": root_mean_squared_error,
                                          "Interp": Interp,
                                          "relu6": relu6}), model_folder, config, initial_epoch
    else:
        # Prepare for logging
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        path = Path('data/driving_models') / model_name / timestamp
        if not os.path.exists(str(path)):
            os.makedirs(str(path))

        # Save parmeters to disk
        with open(str(path / "parameters.txt"), "w") as text_file:
            text_file.write(parameters_string)

        # Save config file to disk
        model_type = "lstm"
        config = configparser.ConfigParser()
        config["ModelConfig"] = {
            'Model': model_type,
            'sequence_length': seq_length,
            'sampling_interval': sampling_interval,
            'segmentation_model_name': segmentation_model_name,
            'sine_steering': sine_steering,
            'use_side_cameras': use_side_cameras,
            'augment': augment,
        }
        with open(str(path / 'config.ini'), 'w') as configfile:
            config.write(configfile)

        # Build model
        model = get_lstm_model(seq_length, freeze_segmentation=freeze_segmentation, sine_steering=sine_steering,
                               segm_model=segmentation_model_name, print_summary=True)

        # Compile model
        model.compile(loss=[steer_loss(), mean_squared_error], optimizer=Adam())
        plot_model(model, str(path / "driving_model.png"), show_shapes=True)
        return model, path, config, 0


def get_model_params(model):
    """
    Extracts some info about the model from its input and output layers
    """

    forward_image_input_layer = model.get_layer('forward_image_input')
    steer_pred_output_layer = model.get_layer('steer_pred')

    if len(forward_image_input_layer.input_shape) == 5:
        (_, sequence_length, height, width, channels) = forward_image_input_layer.input_shape
    else:
        # CNN
        (_, height, width, channels) = forward_image_input_layer.input_shape
        sequence_length = None

    sine_steering = steer_pred_output_layer.output_shape[1] > 1
    return (height, width, channels), sequence_length, sine_steering


# Parameters
val_split = 0.8
adjust_hlc_list = [True]

epochs_list = [100]

dataset_folders_lists = \
    [
        # ["/home/audun/fast_training_data/town01_easy_traffic_lights_clear_099",
        #  "/home/audun/fast_training_data/town01_easy_traffic_lights_rain_099"
        #  ],
        # [
        # "/home/audun/fast_training_data/town01_easy_traffic_lights_multi_angle_099",
        # "/home/audun/fast_training_data/town07_easy_traffic_lights_multi_angle_099",
        # "/home/audun/fast_training_data/first_eberg_day",
        # ],
        # [
        #     "/home/audun/fast_training_data/first_eberg_day",
        # ],
        [
            "/home/audun/fast_training_data/town01_easy_traffic_lights_multi_angle_099",
            "/home/audun/fast_training_data/town07_easy_traffic_lights_multi_angle_099",
        ],
        # ["/home/audun/fast_training_data/town01_low_angles_099",
        #  "/home/audun/fast_training_data/town07_low_angles_099",
        #  "/home/audun/fast_training_data/town07_easy_traffic_lights_multi_angle_099"],
        # [
        #     "/home/audun/fast_training_data/town01_low_angles_099",
        #     "/home/audun/fast_training_data/town07_low_angles_099",
        #     "/home/audun/fast_training_data/town07_easy_traffic_lights_multi_angle_099",
        #     "/home/audun/fast_training_data/first_eberg_day"],
        # [
        #     "/home/audun/fast_training_data/town01_low_angles_099",
        #     "/home/audun/fast_training_data/town07_low_angles_099",
        #     "/home/audun/fast_training_data/town01_easy_traffic_lights_multi_angle_099",
        #     "/home/audun/fast_training_data/town07_easy_traffic_lights_multi_angle_099",
        #     "/home/audun/fast_training_data/first_eberg_day"],
    ]

steering_corrections = [0.05]

batch_sizes = [32]

sampling_intervals = [3]

seq_lengths = [1]

sine_steering_list = [False]

balance_data_list = [True]

use_side_cameras_list = [True]
segmentation_model_name_list = [
    "mapillary_carla_segm_depth",
    "mobilenet_unet_depth_segm_5_classes_carla_only_depth_segm_depth_w05",
    "mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm",
]
# segmentation_model_name_list = ["mobilenet_unet_5_classes_augmentFalse"]

freeze_segmentation_list = [False]

augment_list = [True]

# ## Training loop
parameter_permutations = itertools.product(epochs_list,
                                           dataset_folders_lists,
                                           steering_corrections,
                                           batch_sizes,
                                           sampling_intervals,
                                           seq_lengths,
                                           sine_steering_list,
                                           balance_data_list,
                                           use_side_cameras_list,
                                           segmentation_model_name_list,
                                           adjust_hlc_list,
                                           freeze_segmentation_list,
                                           augment_list)

# Train a new model for each parameter permutation, and save the best models
model_name = input("Enter name of model: ").strip()
parameter_permutations_list = [p for p in parameter_permutations]

# Uncomment to do everything twice
# parameter_permutations_list.extend(parameter_permutations_list)

print(f"Preparing to train {len(parameter_permutations_list)} models:")
print(parameter_permutations_list)

# Set up Tensorflow to not use all memory if not needed
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# Used to only ask about continuing training on the first iteration
first_iteration = True
for parameters in parameter_permutations_list:
    # Get parameters
    epochs, dataset_folders, steering_correction, batch_size, sampling_interval, seq_length, sine_steering, balance_data, use_side_cameras, segmentation_model_name, adjust_hlc, freeze_segmentation, augment = parameters

    parameters_string = (
        f"epochs:\t\t\t{epochs}\n"
        f"dataset folders:\t{str(dataset_folders)}\n"
        f"steering correction:\t{steering_correction}\n"
        f"batch size:\t\t{batch_size}\n"
        f"balance:\t\t{balance_data}\n"
        f"sine_steer:\t\t{sine_steering}\n"
        f"sampling interval:\t{sampling_interval}\n"
        f"seq lenght: \t\t{seq_length}\n"
        f"use_side_cameras:\t\t{use_side_cameras}\n"
        f"segmentation_model_name:\t{segmentation_model_name}\n"
        f"adjust_hlc:\t{adjust_hlc}\n"
        f"freeze_segmentation:\t{freeze_segmentation}\n"
        f"augment:\t{augment}\n"
        f"\n"
    )

    model, path, config, initial_epoch = build_or_load_model(first_iteration, model_name, seq_length, sampling_interval,
                                                             sine_steering,
                                                             segmentation_model_name, use_side_cameras,
                                                             parameters_string, freeze_segmentation, augment)

    first_iteration = False

    # We only use the most essential parameters if we resume training,
    # as we may want to continue training with other params.
    model_config = config["ModelConfig"]
    seq_length = int(model_config["sequence_length"])
    sampling_interval = int(model_config["sampling_interval"])
    sine_steering = bool(strtobool(model_config.get("sine_steering", "True")))
    augment = bool(strtobool(model_config.get("augment", "False")))
    use_side_cameras = bool(strtobool(model_config.get("use_side_cameras", "False")))

    checkpoint_val = ModelCheckpoint(
        str(path / (
            '{epoch:02d}_s{val_steer_pred_loss:.4f}_ts{val_target_speed_pred_loss:.4f}_l{val_loss}.h5')),
        monitor='val_steer_pred_loss',
        verbose=1, save_best_only=True, mode="min")

    # Load drive logs and paths
    inputs, targets = load_driving_logs(dataset_folders, use_side_cameras, steering_correction)

    # Adjust hlcs
    if adjust_hlc:
        print("Moving some HLCs back randomly")
        inputs["hlcs"] = adjust_hlcs(inputs["hlcs"], inputs["info_signals"])

    # Prepare dataset for LSTM
    inputs_flat, targets_flat = prepare_dataset_lstm(inputs, targets, sampling_interval, seq_length)

    # Balance data
    if balance_data:
        # Plot data before balancing
        title_before = "Before Balancing"
        # print("inputs_flat", inputs_flat)
        plot_data(get_dist(inputs_flat, targets_flat), title=title_before)

        # Balance data
        inputs_flat, targets_flat, dist_bal1 = balance_data_lstm(inputs_flat, targets_flat)

        # Plot data after balancing
        title_after = "After Balancing"
        plot_data(dist_bal1, title=title_after)

    inputs_flat_dict = create_input_dict()
    targets_flat_dict = create_target_dict()

    for key in inputs_flat:
        inputs_flat_dict[key] = inputs_flat[key]

    for key in targets_flat:
        targets_flat_dict[key] = targets_flat[key]

    # Shuffle data
    inputs_flat, targets_flat = shuffle_data(inputs_flat, targets_flat)

    # Split into val and train
    split_pos = int(val_split * len(inputs_flat["forward_imgs"]))
    inputs_train, inputs_val = split_dict(inputs_flat, split_pos)
    targets_train, targets_val = split_dict(targets_flat, split_pos)

    # Print training info
    parameters_string = (
        f"epochs:\t\t\t{epochs}\n"
        f"dataset folders:\t{str(dataset_folders)}\n"
        f"steering correction:\t{steering_correction}\n"
        f"batch size:\t\t{batch_size}\n"
        f"balance:\t\t{balance_data}\n"
        f"sine_steer:\t\t{sine_steering}\n"
        f"sampling interval:\t{sampling_interval}\n"
        f"seq lenght: \t\t{seq_length}\n"
        f"use_side_cameras:\t\t{use_side_cameras}\n"
        f"segmentation_model_name:\t{segmentation_model_name}\n"
        f"adjust_hlc:\t{adjust_hlc}\n"
        f"freeze_segmentation:\t{freeze_segmentation}\n"
        f"augment:\t{augment}\n"
        f"\n"
    )

    train_num = len(inputs_train["forward_imgs"])
    val_num = len(inputs_val["forward_imgs"])
    print("Initiate training loop with the following parameters:")
    print("---")
    print(parameters_string)
    print("---")
    print("Training set size: " + str(train_num))
    print("Validation set size: " + str(val_num))

    # Create image of model architecture
    # plot_model(model, str(path/'model.png'))

    steps = int(train_num / batch_size)
    steps_val = int(val_num / batch_size)

    # Define early stopping params
    es = EarlyStopping(monitor='val_steer_pred_loss', mode='min', verbose=1, patience=8)
    # es = EarlyStopping(monitor='val_steer_pred_loss', mode='min', verbose=1, patience=16)

    (img_height, img_width, _), _, _ = get_model_params(model)
    # Train model
    history_object = model.fit_generator(
        generator(inputs_train, targets_train, batch_size, sine_steering=sine_steering, augment=augment,
                  img_height=img_height, img_width=img_width),
        validation_data=generator(inputs_val, targets_val, batch_size, validation=True,
                                  sine_steering=sine_steering, img_height=img_height, img_width=img_width),
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint_val, es],
        steps_per_epoch=steps,
        validation_steps=steps_val,
        use_multiprocessing=True,
        max_queue_size=12,
        workers=6,
        initial_epoch=initial_epoch,
        shuffle=True,
    )

    # Prepare plot and save it to disk
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(history_object.history['steer_pred_loss'], color="blue")
    ax.plot(history_object.history['val_steer_pred_loss'], color="blue", linestyle="--")
    ax.plot(history_object.history['target_speed_pred_loss'], color="green")
    ax.plot(history_object.history['val_target_speed_pred_loss'], color="green", linestyle="--")

    ax.set_title("Mean squared loss of: target_speed and steer")
    ax.set_xlabel("epochs")
    ax.set_ylabel("mse")

    lgd = ax.legend(['steer loss',
                     'steer validation loss',
                     'target speed loss',
                     'target speed validation loss'], bbox_to_anchor=(1.1, 1.05))

    plt.show()
    fig.savefig(str(path / 'loss.png'), bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('\n\n\n\n')

    # Very important: Clear Tensorflow session before starting permutation
    K.clear_session()
