import os
import cv2
import random
from keras_segmentation.predict import model_from_checkpoint_path

random.seed(1)
checkpoint = model_from_checkpoint_path("pspnet_checkpoints_best/pspnet_50_three")

filenames = os.listdir("predict_test_data/carla_images")
print("Sorting", len(filenames))
filenames.sort()
print("Done sorting", len(filenames))

for filename in filenames:
    in_name = os.path.join("./predict_test_data/carla_images/", filename)
    out_name = os.path.join("./predict_test_data/carla_images_segm/", filename.replace("jpg", "png"))

    print("Segmenting:", in_name, out_name)
    out = checkpoint.predict_segmentation(
        inp=in_name,
        out_fname=out_name
    )
   # print(out)
    #cv2.imwrite(out_name, out)
