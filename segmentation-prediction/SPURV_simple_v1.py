#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:
from ast import literal_eval

import sys

from keras_segmentation.data_utils.data_loader import get_image_array

print(sys.version_info)

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser
import tensorflow as tf
import time
import itertools
from glob import glob
from pathlib import Path
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Input, Flatten, Dense, Cropping2D, Conv2D, concatenate, TimeDistributed, CuDNNLSTM
# from tensorflow.keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Flatten, Dense, Cropping2D, Conv2D, concatenate, TimeDistributed, CuDNNLSTM
from keras.utils import Sequence, plot_model
import random
from keras_segmentation.predict import model_from_checkpoint_path
from keras.backend import tf

checkpoints_path = "/hdd/audun/master-thesis-code/training/model_checkpoints/pspnet_checkpoints_best/pspnet_50_three"

import tensorflow.keras.backend as K
import scipy

# import pydotplus

print(tf.__version__)
print(cv2.__version__)
print(os.environ["CONDA_DEFAULT_ENV"])

# ## Helper functions
#

# In[2]:


# hlc_one_hot = { 0: [1,0,0], 1:[0,1,0], 2:[0,0,1] }
hlc_one_hot = {1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 1, 0], 4: [0, 0, 0, 1]}


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
        "hlcs": [],
    }


def create_target_dict():
    return {
        "steer": [],
        "throttle": [],
    }


def split_dict(dictionary, split_pos):
    """ Split data into training and validation """
    train_dict = {}
    val_dict = {}

    for key in dictionary:
        train_dict[key] = dictionary[key][:split_pos]
        val_dict[key] = dictionary[key][split_pos:]

    return train_dict, val_dict


"""
def adjust_hlcs(hlcs, info_signals):
    Randomly adjust HLC backwards, 
    such that the HLC data is given before the car is in the intersection 
    
    # Iterate over all episodes 
    for e in range(len(hlcs)): 
        active_change_hlc = None
        changed = 0
        adjust_num = random.randint(20, 40)
        # Iterate over all hlcs in episode 
        for i in range(len(hlcs[e])-2,-1, -1):
            next_hlc = np.argmax(hlcs[e][i+1]) + 1 
            current_hlc = np.argmax(hlcs[e][i]) +1 
            if active_change_hlc == False:
                active_change_hlc = None
                continue

            if active_change_hlc is None:
                if next_hlc == 1 or next_hlc == 2 or next_hlc == 3:                     
                    if current_hlc == 4: 
                        active_change_hlc = hlcs[e][i+1]
            if active_change_hlc is not None: 
                hlcs[e][i] = active_change_hlc
                if info_signals[e][i][2] != 0: 
                    changed += 1
                if changed == adjust_num:
                    active_change_hlc = False
                    changed = 0 
    return hlcs 
"""


#### BALANCING DATA - Helper functions ####
def get_hlc_dist_label(hlc):
    " Get distributions of HLC in data "
    if hlc == 0:
        return "left"
    elif hlc == 2:
        return "right"
    elif hlc == 1:
        return "straight"


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


# ## Data Augmentations
# - Lightness
# - Hue
# - Gaussian Blur
# - Shadows
# - Rain

# ## Load driving logs
# - Loads data from all csv file in a lits of folders
# - Stores the data in a input dictionary and a target dictionary
# - Normalizes and coverts data to correct format
# - Does not use side cameraes for lane change data
#

# In[3]:


import random


def get_path(episode_path, image_path):
    return str(episode_path / Path("imgs") / image_path.split("/")[-1])


def load_driving_logs(dataset_folders, steering_correction):
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
    img_paths_center = []

    inputs = create_input_dict()

    targets = create_target_dict()
    # Loads data
    for folder in dataset_folders:
        folder_path = Path("dataset") / folder
        for episode in glob(str(folder_path / "*")):

            temp_forward = {"center": [], "left": [], "right": []}
            temp_hlcs = {"center": [], "left": [], "right": []}

            temp_steer = {"center": [], "left": [], "right": []}
            temp_throttle = {"center": [], "left": [], "right": []}

            episode_path = Path(episode)

            df = pd.read_csv(str(episode_path / "driving_log.csv"))
            for index, row in df.iterrows():
                if index == 0:
                    continue

                [throttle, steer, brake] = literal_eval(row["ClientAutopilotControls"])
                # steer = row["angle"]
                # throttle = row["speed"]

                hlc = row["HLC"]
                if hlc == 0 or hlc == -1:
                    hlc = 4
                hlc = get_hlc_one_hot(hlc)

                temp_forward["center"].append(get_path(episode_path, row["ForwardCenter"]))
                temp_steer["center"].append(steer)
                temp_throttle["center"].append(throttle)
                temp_hlcs["center"].append(hlc)

            inputs["forward_imgs"].append(temp_forward["center"])

            inputs["hlcs"].append(temp_hlcs["center"])

            targets["steer"].append(temp_steer["center"])

            targets["throttle"].append(temp_throttle["center"])

    print("Done, {:d} episode(s) loaded.".format(len(inputs["forward_imgs"])))

    return (inputs, targets)


# ## Plot Data

# In[4]:


def plot_data(dist, title=""):
    """ Plots distribution of HLC, speed and traffic lights """
    print("Plotting...")
    tot_num = 0

    fig = plt.figure(figsize=(16, 6))

    # HLC
    labels = ["Left", "Right", "Straight"]
    sizes = [dist["hlc"]["left"], dist["hlc"]["right"], dist["hlc"]["straight"]]

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

# In[5]:

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
            hlc_value = np.argmax(hlc)
            if hlc_value == 0:
                dist["hlc"]["left"] += 1
            elif hlc_value == 1:
                dist["hlc"]["straight"] += 1
            elif hlc_value == 2:
                dist["hlc"]["right"] += 1

    # Speed distribution
    for speed in targets["throttle"]:
        if speed > 0.5:
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


def balance_data_lstm(inputs, targets, straight_angle_frac=0.2, debug=False, drop=False):
    """ Balance dataset for LSTM data """
    print("Balancing data:")

    # Get distribution
    dist = get_dist(inputs, targets)
    print("dist: ", dist)

    inputs_bal = inputs
    targets_bal = targets
    dist_bal = dist

    # Balance steering angle
    print("   - Balancing steering angle")
    inputs_bal, targets_bal = balance_steering_angle(inputs_bal, targets_bal, dist_bal, straight_angle_frac)
    dist_bal = get_dist(inputs_bal, targets_bal)
    print("dist_bal: ", dist_bal)

    # Shuffle
    inputs_bal, targets_bal = shuffle_data(inputs_bal, targets_bal)

    if drop:
        inputs_bal, targets_bal = drop_follow_lane(inputs_bal, targets_bal, 3)

    dist_bal = get_dist(inputs_bal, targets_bal)

    return inputs_bal, targets_bal, dist_bal


# ## Drop random data
# Randomly drop data that is follow lane without brake values or turns

# In[6]:


def drop_follow_lane(inputs, targets, keep_every):
    print("Dropping...")

    inputs_dropped = create_input_dict()
    targets_dropped = create_target_dict()

    for i in range(len(inputs["forward_imgs"])):

        follow_lane = True
        # Iterate over all HLC in sequence
        for hlc in inputs["hlcs"][i]:
            hlc_value = np.argmax(hlc) + 1
            # Only mark sequence as follow lane if all hlcs are follow lane
            if hlc_value != 4:
                follow_lane = False
                break

        steer = targets["steer"][i]

        # Only drop if FOLLOW_LANE, no brake, and no turn
        if follow_lane and abs(steer) < 0.1:
            # Drop based on drop_rate

            if np.random.randint(keep_every) != 0:
                continue

        # Keep
        for key in inputs_dropped:
            inputs_dropped[key].append(inputs[key][i])
        for key in targets_dropped:
            targets_dropped[key].append(targets[key][i])

    return inputs_dropped, targets_dropped


# # LSTM

# ## Prepare dataset format

# In[7]:


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
        [targets_flat["steer"].append(sequence[-1]) for sequence in
         get_episode_sequences(targets["steer"][e], sampling_interval, seq_length)]
        [targets_flat["throttle"].append(sequence[-1]) for sequence in
         get_episode_sequences(targets["throttle"][e], sampling_interval, seq_length)]

    # Shuffle
    inputs_flat, targets_flat = shuffle_data(inputs_flat, targets_flat)

    return (inputs_flat, targets_flat)


# ## Define model

# In[8]:


def get_segmentation_model():
    segmentation_model = model_from_checkpoint_path(checkpoints_path)
    x = segmentation_model.layers[-14].output
    x = Flatten()(x)

    # Explicitly define new model input and output by slicing out old model layers
    model_new = Model(inputs=segmentation_model.layers[0].input,
                      outputs=x)
    plot_model(model_new, "pspnet50_cut.png")
    print(model_new.summary())

    for layer in model_new.layers:
        layer.trainable = False

    return model_new


def get_lstm_model(seq_length, sine_steering=False, print_summary=True):
    forward_image_input = Input(shape=(seq_length, 473, 473, 3), name="forward_image_input")
    hlc_input = Input(shape=(seq_length, 4), name="hlc_input")

    segmentation_model = get_segmentation_model()
    segmentation_output = TimeDistributed(segmentation_model)(forward_image_input)

    # x = TimeDistributed(Cropping2D(cropping=((50,0),(0,0))))(forward_image_input)
    # x = TimeDistributed(Conv2D(24,(5,5),strides=(2,2), activation="relu"))(x)
    # x = TimeDistributed(Conv2D(36,(5,5),strides=(2,2), activation="relu"))(x)
    # x = TimeDistributed(Conv2D(48,(5,5),strides=(2,2), activation="relu"))(x)
    # x = TimeDistributed(Conv2D(64,(3,3),strides=(2,2), activation="relu"))(x)
    # x = TimeDistributed(Conv2D(64,(3,3), activation="relu"))(x)
    # x = TimeDistributed(Conv2D(64,(3,3), activation="relu"))(x)
    # conv_output = TimeDistributed(Flatten())(x)

    # x = concatenate([conv_output, hlc_input])
    x = concatenate([segmentation_output, hlc_input])

    x = TimeDistributed(Dense(100, activation="relu"))(x)
    x = CuDNNLSTM(10, return_sequences=False)(x)
    steer_dim = 1 if not sine_steering else 10
    steer_pred = Dense(steer_dim, activation="tanh", name="steer_pred")(x)
    """
    segmentation_output = TimeDistributed(segmentation_model)(forward_image_input)
    #x = TimeDistributed(Cropping2D(cropping=((50,0),(0,0))))(forward_image_input)
    #x = TimeDistributed(Conv2D(24,(5,5),strides=(2,2), activation="relu"))(x)
    #x = TimeDistributed(Conv2D(36,(5,5),strides=(2,2), activation="relu"))(x)
    #x = TimeDistributed(Conv2D(48,(5,5),strides=(2,2), activation="relu"))(x)
    #x = TimeDistributed(Conv2D(64,(3,3),strides=(2,2), activation="relu"))(x)
    #x = TimeDistributed(Conv2D(64,(3,3), activation="relu"))(x)
    #x = TimeDistributed(Conv2D(64,(3,3), activation="relu"))(x)
    #conv_output = TimeDistributed(Flatten())(x)

    #x = concatenate([conv_output, hlc_input])
    x = concatenate([segmentation_output, hlc_input])
    
    x = TimeDistributed(Dense(100, activation="relu"))(x)
    x = CuDNNLSTM(10, return_sequences = False)(x)
    """
    throtte_pred = Dense(1, name="throttle_pred")(x)
    model = Model(inputs=[forward_image_input, hlc_input], outputs=[steer_pred, throtte_pred])

    if print_summary:
        model.summary()
    plot_model(model, to_file='model.png')

    return model


# mol = get_lstm_model(10)
# get_segmentation_model()


# # Training

# ## Define generator

# In[9]:


class generator(Sequence):
    def __init__(self, inputs, targets, batch_size, validation=False, sine_steering=False):
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.validation = validation
        self.sine_steering = sine_steering

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
        if not self.validation:
            for seq in read_images:
                forward_imgs.append(
                    [get_image_array(image, 473, 473, imgNorm="sub_mean", ordering='channels_last') for image in seq])
        else:
            for seq in read_images:
                forward_imgs.append(
                    [get_image_array(image, 473, 473, imgNorm="sub_mean", ordering='channels_last') for image in seq])

        # forward_imgs = np.array(forward_imgs)/255.0 - 0.5
        return {
                   "forward_image_input": np.array(forward_imgs),
                   "hlc_input": self.inputs["hlcs"][subset]
               }, {
                   "steer_pred": steer_pred,
                   "throttle_pred": self.targets["throttle"][subset],
               }


# ## Parameters

# In[10]:


val_split = 0.8
adjust_hlc = False

epochs_list = [100]
# dataset_folders_lists = [["Town01", "Town04"]]
dataset_folders_lists = [["Town01_simple_noise", "Town01_simple_roaming_noise"]]
"""dataset_folders_lists = [([
    "etron/Town01/ClientAP/no_cars_no_rain",
    "etron/Town01/ClientAP/no_cars_no_rain_noise15",
    "etron/Town01/ClientAP/no_cars_rain_noise10",
    "etron/Town01/ClientAP/cars_no_rain_noise15",
    "etron/Town01/ClientAP/cars_rain_noise10",    
], [
    "etron/Town04/ClientAP/no_cars_no_rain", 
    "etron/Town04/ClientAP/no_cars_rain_noise5", 
    "etron/Town04/ClientAP/cars_no_rain_noise5", 
    "etron/Town04/ClientAP/cars_rain_noise5", 
    "etron/Town01/ClientAP/brake_all_weather_noise15",
    "etron/Town01/ClientAP/brake_all_weather",
])]"""

steering_corrections = [0.05]

batch_sizes = [32]

sampling_intervals = [2, 3]

seq_lengths = [1, 2]

sine_steering_list = [True, False]

balance_data_list = [True, False]

# ## Training loop

# In[ ]:


parameter_permutations = itertools.product(epochs_list,
                                           dataset_folders_lists,
                                           steering_corrections,
                                           batch_sizes,
                                           sampling_intervals,
                                           seq_lengths,
                                           sine_steering_list,
                                           balance_data_list)

# Train a new model for each parameter permutation, and save the best models
model_name = input("Name of model test: ").strip()
# balance_data = True if input("Balance data y/[n]: ").lower() == "y" else False
# drop_data = True if input("Drop data y/[n]: ").lower() == "y" else False

parameter_permutations_list = [p for p in parameter_permutations]
# glos_cycle_full_folders = ["glos_cycle_track_wet_clouded", "glos_cycle_noise_mini"]
glos_cycle_full_folders = ["triangular_noise_glos_train"]
# parameter_permutations_list.append([epochs_list[0],
#                                 glos_cycle_full_folders,
#                                 steering_corrections[0],
#                                 batch_sizes[0],
#                                 sampling_intervals[0],
#                                 seq_lengths[0],
#                                 False,
#                                 False])
#
# parameter_permutations_list.append([epochs_list[0],
#                                 glos_cycle_full_folders,
#                                 steering_corrections[0],
#                                 batch_sizes[0],
#                                 sampling_intervals[0],
#                                 seq_lengths[0],
#                                 True,
#                                 False])


for parameters in parameter_permutations_list:
    # Get parameters
    epochs, dataset_folders, steering_correction, batch_size, sampling_interval, seq_length, sine_steering, balance_data = parameters
    parameters_string = (
        "epochs:\t\t\t{}\ndataset folders:\t{}\nsteering correction:\t{}\nbatch size:\t\t{}\nbalance:\t\t{}\nsine_steer:\t\t{}\nsampling interval:\t{}\nseq lenght: \t\t{}\n\n"
        .format(epochs, str(dataset_folders), steering_correction, batch_size, balance_data, sine_steering,
                sampling_interval, seq_length))

    # town1_dataset_folders, town4_dataset_folders = dataset_folders

    # Prepare for logging
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    path = Path('models') / model_name / timestamp
    if not os.path.exists(str(path)):
        os.makedirs(str(path))

    # Save parmeters to disk
    with open(str(path / "parameters.txt"), "w") as text_file:
        text_file.write(parameters_string)

    # Save config file to disk
    model_name = "lstm"
    config = configparser.ConfigParser()
    config["ModelConfig"] = {'Model': model_name, 'sequence_length': seq_length, 'sampling_interval': sampling_interval}
    with open(str(path / 'config.ini'), 'w') as configfile:
        config.write(configfile)

    # Load drive logs and paths
    inputs, targets = load_driving_logs(dataset_folders, steering_correction)
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
    train_num = len(inputs_train["forward_imgs"])
    val_num = len(inputs_val["forward_imgs"])
    print("Initiate training loop with the following parameters:")
    print("---")
    print(parameters_string)
    print("---")
    print("Training set size: " + str(train_num))
    print("Validation set size: " + str(val_num))

    # Get model
    model = get_lstm_model(seq_length, print_summary=False, sine_steering=sine_steering)

    # Compile model
    model.compile(loss=[steer_loss(), mean_squared_error], optimizer=Adam())
    checkpoint_val = ModelCheckpoint(
        str(path / ('{epoch:02d}_s{val_steer_pred_loss:.4f}_t{val_throttle_pred_loss:.4f}.h5')), monitor='val_loss',
        verbose=1, save_best_only=False, mode="min")

    # Create image of model architecture
    # plot_model(model, str(path/'model.png'))

    steps = int(train_num / batch_size)
    steps_val = int(val_num / batch_size)

    # Define early stopping params
    es = EarlyStopping(monitor='val_steer_pred_loss', mode='min', verbose=1, patience=8)

    # Train model
    print(model.summary())
    history_object = model.fit_generator(
        generator(inputs_train, targets_train, batch_size, sine_steering=sine_steering),
        validation_data=generator(inputs_val, targets_val, batch_size, validation=True, sine_steering=sine_steering),
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint_val, es],
        steps_per_epoch=steps,
        validation_steps=steps_val,
        use_multiprocessing=False,
        workers=1
    )

    # Prepare plot and save it to disk
    fig = plt.figure()
    ax = fig.add_subplot(111)

    print("History", history_object.history)
    ax.plot(history_object.history['steer_pred_loss'], color="blue")
    ax.plot(history_object.history['val_steer_pred_loss'], color="blue", linestyle="--")
    ax.plot(history_object.history['throttle_pred_loss'], color="green")
    ax.plot(history_object.history['val_throttle_pred_loss'], color="green", linestyle="--")

    ax.set_title("Mean squared loss of: throttle and steer")
    ax.set_xlabel("epochs")
    ax.set_ylabel("mse")

    lgd = ax.legend(['steer loss',
                     'steer validation loss',
                     'throttle loss',
                     'throttle validation loss'], bbox_to_anchor=(1.1, 1.05))

    plt.show()
    fig.savefig(str(path / 'loss.png'), bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('\n\n\n\n')

# In[ ]:


# In[ ]:
