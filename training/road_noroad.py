from keras_segmentation.predict import model_from_checkpoint_path
checkpoint = model_from_checkpoint_path("pspnet_checkpoints/pspnet_50")
out = checkpoint.predict_segmentation(
    inp="eberg.jpg",
    out_fname="eberg_pspnet50.png"
)