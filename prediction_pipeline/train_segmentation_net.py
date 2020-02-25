from keras.utils import plot_model
import time
import os
from pathlib import Path
from prediction_pipeline.utils.pspnet import pspnet, train

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

train_segmentation_model(num_classes=8)