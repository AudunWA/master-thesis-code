import os
import random
from keras_segmentation.predict import model_from_checkpoint_path
from keras.backend import tf

random.seed(1)
checkpoints_path = "/home/audun/master-thesis-code/training/psp_checkpoints_best/pspnet_50_eight"
#checkpoints_path = "/home/audun/master-thesis-code/training/psp_checkpoints_best/mobilenet_eight"
print(open(checkpoints_path + "_config.json").readlines())

checkpoint = model_from_checkpoint_path(checkpoints_path)

filenames = os.listdir("dataset/images")
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
