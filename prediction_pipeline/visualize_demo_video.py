"""
This script runs a given set of images through a perception model, creating a video of its output.
"""
import random
from glob import glob

from prediction_pipeline.unet_depth_segm import unet_model_from_checkpoint_path

random.seed(1)


def visualize_segm_depth(path_to_checkpoint):
    checkpoint = unet_model_from_checkpoint_path(path_to_checkpoint)
    for filename in glob(
            "/hdd/audun/master-thesis-code/carla-scripts/hege-max/output/more_demo_video/**/*/forward_center_*"):
        out_segm_name = filename.split(".")[0] + "_seg_depth.png"
        checkpoint.predict_segmentation(
            out_fname=out_segm_name,
            inp=filename
        )


def create_videos():
    import os
    for dir in next(os.walk("/hdd/audun/master-thesis-code/carla-scripts/hege-max/output/more_demo_video/."))[1]:
        os.chdir("/hdd/audun/master-thesis-code/carla-scripts/hege-max/output/more_demo_video/" + dir + "/imgs")
        os.system(f"ffmpeg -framerate 20 -pattern_type glob -i \"*_seg_depth.png\" {dir}_seg_depth.mp4")
        os.system(f"ffmpeg -framerate 20 -pattern_type glob -i \"hq_*.png\" {dir}_hq.mp4")


checkpoint_path = "data/segmentation_models/pspnet_5_classes/2020-04-17_11-19-11/pspnet_5_classes"
mobilenet_path = "data/segmentation_models/mobilenet_unet_5_classes/2020-04-17_12-27-34/mobilenet_unet_5_classes"
vanilla_fcn32_five_class = "data/segmentation_models/fcn_32_5_classes_augmentFalse/2020-04-18_03-25-07/fcn_32_5_classes_augmentFalse"
vanilla_psp_five_class = "data/segmentation_models/pspnet_5_classes/2020-04-17_11-19-11/pspnet_5_classes"
mobilenet_unet_five_class = "data/segmentation_models/mobilenet_unet_5_classes_augmentFalse/2020-04-18_02-52-11/mobilenet_unet_5_classes_augmentFalse"
mobilenet_unet_combined = "data/segmentation_models/mobilenet_unet_5_classes_augment_false_combined_data_true/2020-04-20_18-24-16/mobilenet_unet_5_classes_augment_false_combined_data_true"
mobilenet_augment_combined = "data/segmentation_models/mobilenet_unet_5_classes_augment_true_combined_data_true/2020-04-21_14-00-05/mobilenet_unet_5_classes_augment_true_combined_data_true"
mobilenet_carla_only = "data/segmentation_models/mobilenet_unet_5_classes_augment_true_data_carla/2020-04-23_15-08-04/mobilenet_unet_5_classes_augment_true_data_carla"

mapillary_segm_depth = "data/segmentation_models/mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm/2020-04-27_16-27-21/mobilenet_unet_depth_segm_5_classes_mapillary_depth_segm"
maillary_carla_segm_depth = "data/segmentation_models/mobilenet_unet_depth_segm_5_classes_carla_mapillary_depth_segm/2020-04-28_10-12-38/mobilenet_unet_depth_segm_5_classes_carla_mapillary_depth_segm"
mapilarry_carla_segm_depth_w01 = "data/segmentation_models/mobilenet_unet_depth_segm_5_classes_carla_mapillary_depth_segm_depth_w0.1/2020-04-30_04-25-13/mobilenet_unet_depth_segm_5_classes_carla_mapillary_depth_segm_depth_w0.1"
mapillary_carla_only = "data/segmentation_models/mobilenet_unet_depth_segm_5_classes_carla_only_depth_segm_depth_w0.5/2020-05-01_15-06-39/mobilenet_unet_depth_segm_5_classes_carla_only_depth_segm_depth_w0.5"

visualize_segm_depth(mapillary_carla_only)
create_videos()
