from keras.utils import plot_model
import time
import os
from pathlib import Path

from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_101_voc12

from prediction_pipeline.utils.pspnet import pspnet, train
from keras_segmentation.models import fcn, pspnet as default_pspnet, segnet
from keras_segmentation.pretrained import resnet_pspnet_VOC12_v0_1
from keras.utils import plot_model
model_folder = Path('data/segmentation_models')
data_folder = Path('data/segmentation_data')



def get_segmentation_depth_model(num_classes):

    print("get_segmentation_depth_model")
    model = pspnet(num_classes, input_height=192, input_width=192)
    print(model.summary())
    plot_model(model, "model.png")

    return model




def train_segmentation_model(num_classes=8):

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

    model = get_segmentation_depth_model(num_classes)

    model_name = model.model_name + "_" + str(num_classes) + "_classes"
    checkpoint_path = model_folder / model_name / timestamp / model_name

    if not os.path.exists(str(model_folder / model_name / timestamp)):
        os.makedirs(str(model_folder / model_name / timestamp))

    train(
        model,
        train_images=str(data_folder / "images_prepped_train"),
        train_annotations_segm=str(data_folder / "segm_annotations_prepped_train"),
        train_annotations_depth=str(data_folder / "depth_annotations_prepped_train"),
        val_images=str(data_folder / "images_prepped_val"),
        val_annotations_segm = str(data_folder / "segm_annotations_prepped_val"),
        val_annotations_depth = str(data_folder / "depth_annotations_prepped_val"),
        checkpoints_path=str(checkpoint_path),
        validate=True,
        do_augment=True,
        epochs=100,
    )

def train_resnet_50_psp(num_classes=8):

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    pretrained = resnet_pspnet_VOC12_v0_1()
    model = default_pspnet.resnet50_pspnet(n_classes=8)

    transfer_weights(pretrained, model)

    model_name = model.model_name + "_" + str(num_classes) + "_classes"
    checkpoint_path = model_folder / model_name / timestamp / model_name

    if not os.path.exists(str(model_folder / model_name / timestamp)):
        os.makedirs(str(model_folder / model_name / timestamp))

    model.train(
        train_images=str(data_folder / "images_prepped_train"),
        train_annotations=str(data_folder / "segm_annotations_prepped_train"),
        val_images=str(data_folder / "images_prepped_val"),
        val_annotations=str(data_folder / "segm_annotations_prepped_val"),
        checkpoints_path=str(checkpoint_path),
        validate=True,
        do_augment=True,
        epochs=150,
    )

def print_model_summary():
    model = fcn.fcn_32_resnet50(n_classes=8)
    psp = default_pspnet.resnet50_pspnet(n_classes=8)
    seg = segnet.resnet50_segnet(n_classes=8)

    print(seg.summary())
    print(psp.summary())
    print(model.summary())
    plot_model(seg, "seg_model.png")
    plot_model(psp, "psp_model.png")



#train_segmentation_model(num_classes=8)
train_resnet_50_psp(num_classes=8)

import os, shutil
"""
dir = "data/segmentation_data/segm_annotations_prepped_train/"
filenames = os.listdir(dir)

for file in filenames:
    src = dir + file
    dst = dir + file.replace("_gtCoarse_color", "")
    os.rename(src, dst)
"""