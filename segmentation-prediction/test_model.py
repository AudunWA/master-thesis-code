

from keras.models import load_model, Model
from keras.backend import tensorflow_backend as K
from keras.layers import InputLayer
from keras_segmentation.data_utils.data_loader import get_image_array
from keras_segmentation.models._pspnet_2 import Interp
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd

def get_model_params(model):
    """
    Extracts some info about the model from its input and output layers
    """

    forward_image_input_layer = model.get_layer('forward_image_input')
    steer_pred_output_layer = model.get_layer('steer_pred')

    (_, sequence_length, height, width, channels) = forward_image_input_layer.input_shape
    sine_steering = (steer_pred_output_layer.output_shape == (None, 10))
    return (height, width, channels), sequence_length, sine_steering


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def steering_loss(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

print("Loading model")

model = load_model("13_s0.1882_t0.1748.h5", custom_objects={'custom': steering_loss, 'Interp': Interp})

top_model = Model(inputs=model.layers[0].input, outputs=model.layers[1].output)

img_shape, sequence_length,sine_steering = get_model_params(model)

SEQUENCE_SPACE = 3
print("Image shape: " + str(img_shape) + ", sequence length: " + str(
    sequence_length) + ", sine steering? " + str(sine_steering))


hlc_history = []
image_history = []

def get_prediction(image):

    image_history.append(image)

    hlc = [0, 1, 0, 0]
    hlc_history.append(hlc)

    req_len = (sequence_length - 1) * (SEQUENCE_SPACE + 1) + 1
    if len(image_history) > req_len:
        hlc_history.pop(0)
        image_history.pop(0)

    if len(image_history) < req_len:
        return 0, 0

    image_sequence = np.array([image_history[0::SEQUENCE_SPACE + 1]])
    hlc_sequence = np.array([hlc_history[0::SEQUENCE_SPACE + 1]])

    prediction = model.predict({'forward_image_input': image_sequence,
                            'hlc_input': hlc_sequence})

    prediction_2 = top_model.predict({'forward_image_input': image_sequence,
                            'hlc_input': hlc_sequence})

    print("prediction_2", prediction_2.shape, prediction_2)
    steer = prediction[0][0]
    throttle = prediction[1][0][0]

    steer_angle = steer[0]

    return throttle, steer_angle

folder = "/home/audun/master-thesis-code/segmentation-prediction/dataset/2020-02-06_12-06-15/imgs/"
filenames = os.listdir(folder)
filenames.sort()
for filename in filenames:
    img = cv2.imread(folder + filename)


    formatted_img = get_image_array(img, 473, 473, imgNorm='sub_mean', ordering='')

    print("Getting prediction for", filename, formatted_img.shape)

    throttle, steer_angle = get_prediction(formatted_img)
    #print(formatted_img)
    plt.imshow(formatted_img)
    plt.show()
    print("Throttle", throttle, "Steer angle", steer_angle)

    if filename == "forward_center_rgb_00000062.png":
        print(formatted_img)

