import os
import random
import cv2
from keras import Model
from keras.engine.saving import load_model
from keras_segmentation.data_utils.data_loader import get_image_array
from keras_segmentation.predict import evaluate
from pathlib import Path
import keras.backend as K
from keras_segmentation.pretrained import pspnet_101_cityscapes
from scipy.optimize import curve_fit

from prediction_pipeline.depth_modifications.evaluate import evaluate_depth_segm
from prediction_pipeline.utils.helpers import verify_folder_exists

#from prediction_pipeline.utils.pspnet import model_from_checkpoint_path, predict
import numpy as np
random.seed(1)

data_folder = Path("data/segmentation_data/")

def evaluate_mapillary_segmentation_model(path_to_checkpoint, data_folder):

    segm_dir = data_folder / "mapillary_depth_segm/segm_annotations_prepped_val"
    img_dir = data_folder / "mapillary_depth_segm/images_prepped_val"

    results = evaluate(checkpoints_path=path_to_checkpoint, inp_images_dir=img_dir, annotations_dir=segm_dir)
    return results

def evaluate_carla_segmentation_model(path_to_checkpoint, data_folder):

    segm_dir = data_folder / "segmentation"
    img_dir = data_folder / "rgb"

    results = evaluate(checkpoints_path=path_to_checkpoint, inp_images_dir=img_dir, annotations_dir=segm_dir)
    return results

def evaluate_depth_segm_model(path_to_checkpoint, data_folder):

    segm_dir = data_folder / "prepped_segmentation_seven_classes"
    depth_dir = data_folder / "depth_annotation"
    img_dir = data_folder / "rgb"

    results = evaluate_depth_segm(checkpoints_path=path_to_checkpoint, inp_images_dir=img_dir, segm_annotations_dir=segm_dir, depth_annotations_dir=depth_dir)
    return results

def evalate_all_carla_towns(path_to_checkpoint, evaluation_method=evaluate_carla_segmentation_model):
    print("Evaluating towns ...")
    res = evaluation_method(path_to_checkpoint, data_folder=data_folder/"carla_test/Town01")
    print("Town 1:", res)
    res = evaluation_method(path_to_checkpoint, data_folder=data_folder/"carla_test/Town02")
    print("Town 2:", res)
    res = evaluation_method(path_to_checkpoint, data_folder=data_folder/"carla_test/Town03")
    print("Town 3:", res)
    res = evaluation_method(path_to_checkpoint, data_folder=data_folder/"carla_test/Town04")
    print("Town 4:", res)

def evaluate_mapillary_data(path_to_checkpoint, evaluation_method=evaluate_mapillary_segmentation_model):
    print("Evaluating mapillary ...")
    res = evaluation_method(path_to_checkpoint, data_folder=data_folder)
    print("Mapillary:", res)

vanilla_psp_five_class = "data/segmentation_models/pspnet_5_classes/2020-04-17_11-19-11/pspnet_5_classes"
vanilla_segnet_five_class = "data/segmentation_models/segnet_5_classes_augmentFalse/2020-04-17_23-50-09/segnet_5_classes_augmentFalse"
vanilla_unet_five_class = "data/segmentation_models/unet_5_classes_augmentFalse/2020-04-18_01-31-13/unet_5_classes_augmentFalse"
vanilla_fcn32_five_class = "data/segmentation_models/fcn_32_5_classes_augmentFalse/2020-04-18_03-25-07/fcn_32_5_classes_augmentFalse"

mobilenet_segnet_five_class = "data/segmentation_models/mobilenet_segnet_5_classes_augmentFalse/2020-04-18_01-00-59/mobilenet_segnet_5_classes_augmentFalse"
mobilenet_unet_five_class = "data/segmentation_models/mobilenet_unet_5_classes_augmentFalse/2020-04-18_02-52-11/mobilenet_unet_5_classes_augmentFalse"
mobilenet_fcn32_five_class = "data/segmentation_models/fcn_32_mobilenet_5_classes_augmentFalse/2020-04-20_10-52-01/fcn_32_mobilenet_5_classes_augmentFalse"
resnet50_segnet_five_class = "data/segmentation_models/resnet50_segnet_5_classes_augmentFalse/2020-04-18_06-33-43/resnet50_segnet_5_classes_augmentFalse"
resnet50_unet_five_class = "data/segmentation_models/resnet50_unet_5_classes_augment_false_data_mapillary/2020-04-29_10-05-30/resnet50_unet_5_classes_augment_false_data_mapillary"

mobilenet_unet = "data/segmentation_models/mobilenet_unet_5_classes_augment_false_data_mapillary/2020-05-05_15-22-07/mobilenet_unet_5_classes_augment_false_data_mapillary"
mobilenet_unet_augment = "data/segmentation_models/mobilenet_unet_5_classes_augment_true_combined_data_false/2020-04-20_18-56-35/mobilenet_unet_5_classes_augment_true_combined_data_false"
mobilenet_unet_combined = "data/segmentation_models/mobilenet_unet_5_classes_augment_false_combined_data_true/2020-04-20_18-24-16/mobilenet_unet_5_classes_augment_false_combined_data_true"
mobilenet_augment_combined = "data/segmentation_models/mobilenet_unet_5_classes_augment_true_combined_data_true/2020-04-21_14-00-05/mobilenet_unet_5_classes_augment_true_combined_data_true"
mobilenet_carla_only = "data/segmentation_models/mobilenet_unet_5_classes_augment_true_data_carla/2020-04-23_15-08-04/mobilenet_unet_5_classes_augment_true_data_carla"

mapillary_segm_depth = "data/segmentation_models/mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm/2020-04-27_16-27-21/mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm"
mapillary_carla_segm_depth = "data/segmentation_models/mobilenet_unet_depth_segm_5_classes_carla_mapillary_depth_segm/2020-04-28_10-12-38/mobilenet_unet_depth_segm_5_classes_carla_mapillary_depth_segm"
carla_only_segm_depth = "data/segmentation_models/mobilenet_unet_depth_segm_5_classes_carla_only_depth_segm_depth_w0.5/2020-05-01_15-06-39/mobilenet_unet_depth_segm_5_classes_carla_only_depth_segm_depth_w0.5"
carla_only_no_aug = "data/segmentation_models/mobilenet_unet_5_classes_augment_false_data_carla/2020-06-05_19-09-33/mobilenet_unet_5_classes_augment_false_data_carla"

evalate_all_carla_towns(carla_only_no_aug)
#evaluate_mapillary_data(mobilenet_unet_augment)
#evaluate_mapillary_data(mobilenet_unet_combined)
#evaluate_mapillary_data(mobilenet_augment_combined)
#evaluate_mapillary_data(mobilenet_carla_only)

#evalate_all_carla_towns(mapillary_segm_depth, evaluation_method=evaluate_depth_segm_model)
#evalate_all_carla_towns(mapillary_carla_segm_depth, evaluation_method=evaluate_depth_segm_model)
