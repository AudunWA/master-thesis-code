import os
import random
from keras_segmentation.predict import model_from_checkpoint_path

random.seed(1)
#checkpoints_path = "/home/audun/master-thesis-code/training/psp_checkpoints_best/mobilenet_eight"

checkpoints_path = "/home/audun/master-thesis-code/training/psp_checkpoints_best/pspnet_50_three"

checkpoint = model_from_checkpoint_path(checkpoints_path)

filenames = os.listdir("predict_test_data/carla_images_2")
#print("Sorting", len(filenames))
#filenames.sort(key=lambda x: int(x.replace(".jpg", "")))
#print("Done sorting", len(filenames))



for filename in filenames:
    in_name = os.path.join("./dataset/images/", filename)
    out_name = os.path.join("./dataset/images_segm_psp/", filename.replace("jpg", "png"))

    print("Segmenting:", in_name, out_name)
    out = checkpoint.predict_segmentation(
        inp=in_name,
        out_fname=out_name
    )
    # print(out)
    # cv2.imwrite(out_name, out)
