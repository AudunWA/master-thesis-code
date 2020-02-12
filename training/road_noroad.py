
# coding: utf-8

# In[3]:


from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.pretrained import  pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_101, pspnet_50
import keras_segmentation.models.all_models as models

import cv2
img = cv2.imread("dataset/annotations_prepped_train/a0vGGO5MIsQW1K_-sulhAg.png")
print(img.max())
new_model = models.model_from_name["mobilenet_segnet"](n_classes=8)

#transfer_weights( pretrained_model , new_model  ) # transfer weights from pre-trained model to your model

new_model.train(
    train_images =  "dataset/images_prepped_train/",
    train_annotations = "dataset/annotations_prepped_train/",
    val_images="dataset/images_prepped_val/",
    val_annotations="dataset/annotations_prepped_val/",
    checkpoints_path = "mobilenet_checkpoints/pspnet_50_eight",
    validate=True,
    do_augment=True,
    epochs=100,
)
exit(0)

# In[ ]:


model = vgg_unet( n_classes=7, input_height=256, input_width=256 )

model.train(
    train_images =  "dataset/images_prepped_train/",
    train_annotations = "dataset/annotations_prepped_train/",
    checkpoints_path = "checkpoints/psp_50_seven" ,
    epochs=10
)

out = model.predict_segmentation(
    inp="carla.png",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

# evaluating the model
print(model.evaluate_segmentation( inp_images_dir="dataset2/images_prepped_test/"  , annotations_dir="dataset2/annotations_prepped_test/" ) )



# In[ ]:


out = model.predict_segmentation(
    inp="../../training/images/-RvUfordKSu3eFmYIBbt_A.jpg",
    out_fname="/tmp/out.png"
)
plt.imshow(out)


# In[ ]:


out = model.predict_segmentation(
    inp="carla.png",
    out_fname="eberg_out.png"
)
plt.imshow(out)


# In[4]:


import matplotlib.pyplot as plt
from keras_segmentation.predict import model_from_checkpoint_path
checkpoint = model_from_checkpoint_path("pspnet_checkpoints_best/pspnet_50")
out = checkpoint.predict_segmentation(
    inp="carla.png",
    out_fname="carla_pspnet50.png"
)
plt.imshow(out)

