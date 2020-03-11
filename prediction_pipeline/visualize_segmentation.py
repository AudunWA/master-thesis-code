import os
import random
import cv2
from keras import Model
from keras.engine.saving import load_model
from keras_segmentation.data_utils.data_loader import get_image_array
from keras_segmentation.predict import model_from_checkpoint_path
from pathlib import Path
import keras.backend as K
from keras_segmentation.pretrained import pspnet_101_cityscapes
from scipy.optimize import curve_fit

from prediction_pipeline.utils.helpers import verify_folder_exists

#from prediction_pipeline.utils.pspnet import model_from_checkpoint_path, predict
import numpy as np
random.seed(1)

data_folder = Path("data/segmentation_test_images")

def segment_images_psp(path_to_checkpoint, folder_name):

    output_folder_name = folder_name + "_segm"
    checkpoint = model_from_checkpoint_path(path_to_checkpoint)
    #checkpoint = pspnet_101_cityscapes()
    output_folder = data_folder / output_folder_name
    verify_folder_exists(output_folder)

    for filename in os.listdir(str(data_folder / folder_name)):
        out_name = str(output_folder / filename).replace("jpg", "png")

        in_name = str(data_folder / folder_name / filename)

        checkpoint.predict_segmentation(
            out_fname=out_name,
            inp=in_name
        )

"""
def segment_images(path_to_checkpoint, folder_name):

    output_folder_name = folder_name + "_segm"
    checkpoint = model_from_checkpoint_path(path_to_checkpoint)
    output_folder = data_folder / output_folder_name
    verify_folder_exists(output_folder)

    for filename in os.listdir(str(data_folder / folder_name)):
        out_name = str(output_folder / filename).replace("jpg", "png")

        in_name = str(data_folder / folder_name / filename)

        predict(
            model=checkpoint,
            out_fname=out_name,
            inp=in_name
        )
"""



def root_mean_squared_error(y_true, y_pred):
    """ Custom loss function, RMSE """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_model_cut_at(model: Model, cut_layer_name):
    cut_layer_output = model.get_layer(cut_layer_name).output
    cut_model = Model(inputs=[model.get_layer("forward_image_input").input, model.get_layer("info_input").input, model.get_layer("hlc_input").input], outputs=cut_layer_output)
    return cut_model

def print_output():
    np.set_printoptions(edgeitems=100)

    path = "data/driving_models/psp50/2020-02-27_13-20-12/01_s0.3457_t0.1132_b0.1909.h5"
    path = "data/driving_models/psp50_no_lstm/2020-02-27_15-49-43/01_s1.2058_t0.5040_b0.7655.h5"
    path = "data/driving_models/psp50_no_lstm_batch_norm/2020-02-28_10-37-18/01_s0.1479_t0.1044_b0.1299.h5"
    #path = "data/driving_models/baseline_segm_only/2020-02-25_15-14-11/39_s0.0456_t0.0364_b0.0210.h5"
    path = "data/driving_models/resnet50_psp_early_cut/2020-03-09_15-43-19/03_s0.1595_t0.1111_b0.1673.h5"

    imgs = [cv2.imread("data/carla_data/cleaned_clear_traffic/2020-02-20_16-04-09/imgs/forward_center_rgb_00000096.png"), cv2.imread("data/carla_data/cleaned_clear_traffic/2020-02-20_16-04-09/imgs/forward_center_rgb_00000001.png")]

    model = load_model(path, custom_objects={"custom": root_mean_squared_error})
    model.summary()
    cut_model = get_model_cut_at(model, "concatenate_2")
    for img in imgs:
        img_center = get_image_array(img, 192, 192, imgNorm="sub_mean", ordering='channels_last')
        #img_center = get_image_array(img, 224, 224, imgNorm="sub_mean", ordering='channels_last')
        info_input = [
            max(float(60 * 3.6 / 100), 0.2),
            float(60 * 3.6 / 100),
            1
        ]
        hlc_input = [0, 0, 0, 1]

        prediction = model.predict({
            "forward_image_input": np.array([[img_center]]),
            "info_input": np.array([[info_input]]),
            "hlc_input": np.array([[hlc_input]]),
        })

        cut_prediction = cut_model.predict({
            "forward_image_input": np.array([[img_center]]),
            "info_input": np.array([[info_input]]),
            "hlc_input": np.array([[hlc_input]]),
        })

        print("Pred: ", prediction)
        print("Cut pred: ", cut_prediction)

print_output()
checkpoint_path = "data/segmentation_models/resnet50_pspnet_8_classes/2020-02-26_17-05-57/resnet50_pspnet_8_classes"

#segment_images_psp(checkpoint_path, "carla")

