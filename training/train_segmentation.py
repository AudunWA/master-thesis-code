 #%%
from keras import Model
from keras_segmentation.models.pspnet import pspnet_50


model = pspnet_50(n_classes=7 ,  input_height=3456, input_width=4808)



# load any of the 3 pretrained models
image_dir = "../prepare_segmentation_data/data/dataset/images_prepped_train"
segmentation_dir = "../prepare_segmentation_data/data/dataset/annotations_prepped_train"

model.train(
    train_images =  image_dir,
    train_annotations = segmentation_dir,
    checkpoints_path = "psp" , epochs=5
)

out = model.predict_segmentation(
    inp=image_dir + "/aSqVUgt36gddhmJdI1lXNA.jpg",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

# evaluating the model
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )

