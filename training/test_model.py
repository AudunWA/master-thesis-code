import os
import random
from keras_segmentation.predict import model_from_checkpoint_path
from keras.backend import tf

random.seed(1)
<<<<<<< Updated upstream
checkpoints_path = "/home/audun/master-thesis-code/training/psp_checkpoints_best/pspnet_50_three"

checkpoint = model_from_checkpoint_path(checkpoints_path)

filenames = os.listdir("dataset/images")
=======
checkpoint = model_from_checkpoint_path("model_checkpoints/pspnet_checkpoints_best/pspnet_50_three")

filenames = os.listdir("predict_test_data/carla_images_2")
>>>>>>> Stashed changes
print("Sorting", len(filenames))
filenames.sort(key=lambda x: int(x.replace(".jpg", "")))
print("Done sorting", len(filenames))

for filename in filenames:
<<<<<<< Updated upstream
    in_name = os.path.join("./dataset/images/", filename)
    out_name = os.path.join("./dataset/images_segm/", filename.replace("jpg", "png"))
=======
    in_name = os.path.join("./predict_test_data/carla_images_2/", filename)
    out_name = os.path.join("./predict_test_data/carla_images_2_segm/", filename.replace("jpg", "png"))
>>>>>>> Stashed changes

    print("Segmenting:", in_name, out_name)
    out = checkpoint.predict_segmentation(
        inp=in_name,
        out_fname=out_name
    )
   # print(out)
    #cv2.imwrite(out_name, out)
