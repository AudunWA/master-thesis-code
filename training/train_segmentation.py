 #%%
from keras import Model
from keras_segmentation.models.unet import vgg_unet


model = vgg_unet(n_classes=2 ,  input_height=3456, input_width=4808)

#model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset

# model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset


# load any of the 3 pretrained models
image_dir = "../prepare_segmentation_data/data/input"
segmentation_dir = "../prepare_segmentation_data/data/results"

model.train(
    train_images =  image_dir,
    train_annotations = segmentation_dir,
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)

out = model.predict_segmentation(
    inp=image_dir + "/aSqVUgt36gddhmJdI1lXNA.jpg",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

# evaluating the model
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )

