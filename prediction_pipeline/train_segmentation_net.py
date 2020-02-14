from keras_segmentation.models.pspnet import pspnet
import time
import os
from pathlib import Path

model_folder = Path('data/segmentation_models')
data_folder = Path('data/segmentation_data')

def train_segmentation_model(num_classes=8):

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

    model = pspnet(num_classes, input_height=192, input_width=192)

    model_name = model.model_name + "_" + str(num_classes) + "_classes"
    checkpoint_path = model_folder / model_name / timestamp / model_name

    if not os.path.exists(str(model_folder / model_name / timestamp)):
        os.makedirs(str(model_folder / model_name / timestamp))

    model.train(
        train_images=str(data_folder / "images_prepped_train"),
        train_annotations=str(data_folder / "annotations_prepped_train"),
        val_images=str(data_folder / "images_prepped_val"),
        val_annotations=str(data_folder / "annotations_prepped_val"),
        checkpoints_path=str(checkpoint_path),
        validate=True,
        do_augment=True,
        epochs=100,
    )

train_segmentation_model(num_classes=8)