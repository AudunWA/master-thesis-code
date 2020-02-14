import os
import random
from keras_segmentation.predict import model_from_checkpoint_path
from pathlib import Path
from prediction_pipeline.utils.helpers import verify_folder_exists
random.seed(1)

data_folder = Path("data/segmentation_test_images")

def segment_images(path_to_checkpoint, folder_name):

    output_folder_name = folder_name + "_segm"
    checkpoint = model_from_checkpoint_path(path_to_checkpoint)
    output_folder = data_folder / output_folder_name
    verify_folder_exists(output_folder)

    for filename in os.listdir(str(data_folder / folder_name)):
        in_name = str(data_folder / folder_name / filename)
        out_name = str(output_folder / filename.replace("jpg", "png"))

        checkpoint.predict_segmentation(
            inp=in_name,
            out_fname=out_name
        )


checkpoint_path = "data/segmentation_models/pspnet_8_classes/2020-02-13_15-29-13/pspnet_8_classes"

segment_images(checkpoint_path, "glos_cycle")