import json
import os
from types import MethodType

import cv2
from keras import Model
from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, UpSampling2D, concatenate, Reshape, Permute, \
    Activation
from keras_segmentation.data_utils.data_loader import class_colors, get_image_array
from keras_segmentation.models.config import IMAGE_ORDERING
from keras_segmentation.models.mobilenet import get_mobilenet_encoder
from keras_segmentation.models.unet import MERGE_AXIS
from keras_segmentation.train import find_latest_checkpoint

from prediction_pipeline.depth_modifications.helpers import depth_loss_function, parse_depth_pred, \
    image_segm_depth_generator
import numpy as np

def train(model,
          train_images,
          train_annotations_segm,
          train_annotations_depth,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations_segm=None,
          val_annotations_depth=None,
          val_batch_size=2,
          steps_per_epoch=18000 / 2,
          do_augment=False,
          n_classes=8
          ):
    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations_depth is not None
        assert val_annotations_segm is not None

    if checkpoints_path is not None:
        with open(checkpoints_path + "_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    train_gen = image_segm_depth_generator(
        train_images, train_annotations_segm, train_annotations_depth, batch_size, n_classes,
        input_height, input_width, output_height, output_width, do_augment=do_augment)

    if validate:
        val_gen = image_segm_depth_generator(
            val_images, val_annotations_segm, val_annotations_depth, val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, epochs=1)
            if checkpoints_path is not None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            history = model.fit_generator(train_gen, steps_per_epoch,
                                          validation_data=val_gen,
                                          validation_steps=2000 / val_batch_size, epochs=1, workers=8, use_multiprocessing=True).history
            if checkpoints_path is not None:
                model.save_weights(checkpoints_path + "." + str(ep))
                with open("history.txt", "a+") as file:
                    file.write(
                        f"""Epoch {ep}: val_loss: {history["val_loss"][-1]} - val_segm_pred_loss: {history["val_segm_pred_loss"][-1]} - val_depth_pred_loss: {history["val_depth_pred_loss"][-1]} - val_segm_pred_acc: {history["val_segm_pred_acc"][-1]} - val_depth_pred_acc: {history["val_depth_pred_acc"][-1]}""")
                print("saved ", checkpoints_path + ".model." + str(ep))

            print("Finished Epoch", ep)
def predict(model=None, inp=None, out_fname=None, checkpoints_path=None):

    if model is None and (checkpoints_path is not None):
        model = unet_model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)

    inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))
    pr_segm = pr[0].reshape((output_height,  output_width, n_classes)).argmax(axis=2)

    pr_depth = pr[1].reshape((output_height,  output_width))



    depth_img = parse_depth_pred(pr_depth)



    seg_img = np.zeros((output_height, output_width, 3))
    colors = class_colors

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr_segm[:, :] == c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr_segm[:, :] == c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr_segm[:, :] == c)*(colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))
    depth_img = cv2.resize(depth_img, (orininal_w, orininal_h))

    if out_fname is not None:
        combined = np.concatenate((inp, seg_img, depth_img), axis=1)
        print("Writing", out_fname, combined.shape)
        cv2.imwrite(out_fname, combined)

    return pr_segm, pr_depth


def get_segm_depth_model(input, output_segm, output_depth):
    img_input = input

    o_shape = Model(img_input, output_segm).output_shape
    i_shape = Model(img_input, output_segm).input_shape

    n_classes = 0
    output_height = 0
    output_width = 0
    input_height = 0
    input_width = 0

    if IMAGE_ORDERING == 'channels_first':
        output_height = o_shape[2]
        output_width = o_shape[3]
        input_height = i_shape[2]
        input_width = i_shape[3]
        output_segm = (Reshape((-1, output_height * output_width)))(output_segm)
        output_segm = (Permute((2, 1)))(output_segm)
        n_classes = o_shape[1]

    elif IMAGE_ORDERING == 'channels_last':
        output_height = o_shape[1]
        output_width = o_shape[2]
        input_height = i_shape[1]
        input_width = i_shape[2]
        n_classes = o_shape[3]

        output_segm = (Reshape((output_height * output_width, -1)))(output_segm)



    output_segm = (Activation('softmax', name="segm_pred"))(output_segm)
    output_depth = (Activation('sigmoid', name="depth_pred"))(output_depth)

    model = Model(inputs=img_input, outputs=[output_segm, output_depth])

    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.output_height = output_height
    model.output_width = output_width
    model.train = MethodType(train, model)
    model.predict_segmentation = MethodType(predict, model)
    #model.predict_multiple = MethodType(predict_multiple, model)
    #model.evaluate_segmentation = MethodType(evaluate, model)

    return model

def unet_model_from_checkpoint_path(checkpoints_path):

    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = _unet_depth_segm(
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'], encoder=get_mobilenet_encoder)
    model.load_weights(latest_weights)
    return model


def _get_unet_decoder(levels, l1_skip_conn, n_classes):
    [f1, f2, f3, f4, f5] = levels
    o = f4
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    return o

def _unet_depth_segm(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608, depth_weight=0.5):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width)

    segm_output = _get_unet_decoder(levels, l1_skip_conn, n_classes)

    depth_output = _get_unet_decoder(levels, l1_skip_conn, 1)

    model = get_segm_depth_model(img_input, segm_output, depth_output)
    print(Model(img_input, depth_output).output_shape)
    print(model.output_shape)

    # Weights from https://arxiv.org/pdf/1705.07115.pdf
    #loss_weights = {"depth_pred": 0.1, "segm_pred": 0.9}
    loss_weights = {"depth_pred": depth_weight, "segm_pred": 1 - depth_weight}

    model.compile(loss=['categorical_crossentropy', depth_loss_function], loss_weights=loss_weights,
                  optimizer="adadelta",
                  metrics=['accuracy'])

    return model

def unet_depth_segm(n_classes, input_height=224, input_width=224,
                   depth_weight=None):

    model = _unet_depth_segm(n_classes, get_mobilenet_encoder,
                  input_height=input_height, input_width=input_width, depth_weight=depth_weight)
    model.model_name = "mobilenet_unet_depth_segm"
    return model
