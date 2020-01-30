 #%%
from keras import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12
from keras_segmentation.models import pspnet
base_model = pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset

#model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset

# model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset


# load any of the 3 pretrained models

out = base_model.predict_segmentation(
    inp="carla.png",
    out_fname="out3.png"
)

print(base_model.summary())


predictions = base_model.layers[-4].output
model = Model(inputs=base_model.inputs, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())