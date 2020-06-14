import time
import os
from pathlib import Path

from keras_segmentation.models import unet, segnet
from prediction_pipeline.depth_modifications.unet_depth_segm import unet_depth_segm
from keras.utils import plot_model
model_folder = Path('data/segmentation_models')
data_folder = Path('data/segmentation_data')






# Data can be one of -> "mapillary" | "combined" | "carla"
def train_segmentation_model(model, num_classes, augment=False, data="mapillary"):
    try:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        model = model(n_classes=num_classes)

        model_name = model.model_name + "_" + str(num_classes) + "_classes_augment_" + str(augment).lower() + "_data_" + str(data).lower()
        checkpoint_path = model_folder / model_name / timestamp / model_name

        if not os.path.exists(str(model_folder / model_name / timestamp)):
            os.makedirs(str(model_folder / model_name / timestamp))


        folder = data_folder

        if data == "combined":
            folder = data_folder / "carla_mapillary"
        elif data == "carla":
            folder = data_folder / "carla_only"

        model.train(
            train_images=str(folder / "images_prepped_train"),
            train_annotations=str(folder / "annotations_prepped_train"),
            val_images=str(folder / "images_prepped_val"),
            val_annotations=str(folder / "annotations_prepped_val"),
            checkpoints_path=str(checkpoint_path),
            validate=False,
            verify_dataset=False,
            do_augment=augment,
            epochs=90,
        )
    except Exception as e:
        print("Exception occurred", e)

#dataset can be -> mapillary_depth_segm, carla_mapillary_depth_segm, carla_only_depth_segm
def train_segm_depth_model(model, num_classes, dataset="carla_mapillary_depth_segm", depth_weight=0.5):
    try:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        model = model(n_classes=num_classes, depth_weight=depth_weight)

        model_name = model.model_name + "_" + str(num_classes) + "_classes_"+dataset + "_depth_w"+str(depth_weight)
        checkpoint_path = model_folder / model_name / timestamp / model_name

        if not os.path.exists(str(model_folder / model_name / timestamp)):
            os.makedirs(str(model_folder / model_name / timestamp))


        folder = data_folder / dataset

        model.train(
            train_images=str(folder / "images_prepped_train"),
            train_annotations_segm=str(folder / "segm_annotations_prepped_train"),
            train_annotations_depth=str(folder / "depth_annotations_prepped_train"),
            val_images=str(folder / "images_prepped_val"),
            val_annotations_depth=str(folder / "depth_annotations_prepped_val"),
            val_annotations_segm=str(folder / "segm_annotations_prepped_val"),
            checkpoints_path=str(checkpoint_path),
            validate=True,
            do_augment=False,
            epochs=90,
            batch_size=16,
            val_batch_size=16
        )
    except Exception as e:
        print("Exception occurred", e)

def print_model_summary():

    model = unet_depth_segm(5)
    print(model.summary())
    plot_model(model, "unet_depth_segm_model.png")



#train_segm_depth_model(unet_depth_segm, 5, dataset="carla_only_depth_segm", depth_weight=0.5)

#train_segm_depth_model(unet_depth_segm, 5, depth_weight=0.1)

#train_segmentation_model(segnet.segnet, 5, data="mapillary")
#train_segmentation_model(segnet.mobilenet_segnet, 5, data="mapillary")
train_segmentation_model(unet.mobilenet_unet, 5, data="carla")

