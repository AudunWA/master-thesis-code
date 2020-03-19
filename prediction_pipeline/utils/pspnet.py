from types import MethodType

from keras.layers import AveragePooling2D, Conv2D, BatchNormalization, Activation, Concatenate, Reshape, Permute, \
    Flatten
from keras.utils import plot_model
from keras_segmentation.data_utils.augmentation import augment_seg
from keras_segmentation.data_utils.data_loader import get_image_array, get_segmentation_array, class_colors
from keras_segmentation.models.basic_models import vanilla_encoder
from keras_segmentation.models.config import IMAGE_ORDERING
import numpy as np
import tensorflow as tf
import json
import keras.backend as K
from keras_segmentation.models.model_utils import resize_image, Model
from keras_segmentation.predict import predict, predict_multiple, evaluate
from keras_segmentation.train import train, find_latest_checkpoint
import os
import random
import itertools
import cv2
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1


def get_segmentation_model(input, output_segm, output_depth):

    img_input = input
    o = output_segm

    o_shape = Model(img_input, o).output_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height = o_shape[2]
        output_width = o_shape[3]
        o = (Reshape((-1, output_height*output_width)))(o)
        o = (Permute((2, 1)))(o)
    elif IMAGE_ORDERING == 'channels_last':
        output_height = o_shape[1]
        output_width = o_shape[2]
        o = (Reshape((output_height*output_width, -1)))(o)

    o = (Activation('softmax', name="segm_pred"))(o)
    model = Model(inputs=img_input, outputs=[o, output_depth])

    model.train = MethodType(train, model)
    model.predict_segmentation = MethodType(predict, model)
    model.predict_multiple = MethodType(predict_multiple, model)
    model.evaluate_segmentation = MethodType(evaluate, model)


    return model


def pool_block(feats, pool_factor):

    if IMAGE_ORDERING == 'channels_first':
        h = K.int_shape(feats)[2]
        w = K.int_shape(feats)[3]
    elif IMAGE_ORDERING == 'channels_last':
        h = K.int_shape(feats)[1]
        w = K.int_shape(feats)[2]

    pool_size = strides = [
        int(np.round(float(h) / pool_factor)),
        int(np.round(float(w) / pool_factor))]

    x = AveragePooling2D(pool_size, data_format=IMAGE_ORDERING,
                         strides=strides, padding='same')(feats)
    x = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING,
               padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = resize_image(x, strides, data_format=IMAGE_ORDERING)

    return x

# From https://github.com/ialhashim/DenseDepth/blob/ed044069eb99fa06dd4af415d862b3b5cbfab283/loss.py#L4
def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0 / 10.0):
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    print("y_true",y_true,y_true.shape)
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))

def _pspnet(n_classes, encoder,  input_height=384, input_width=576):

    assert input_height % 192 == 0
    assert input_width % 192 == 0

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]

    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)

    o = Concatenate(axis=MERGE_AXIS)(pool_outs)

    o = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING, use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    segmentation_model = Conv2D(n_classes, (3, 3), data_format=IMAGE_ORDERING,
               padding='same')(o)
    segmentation_model = resize_image(segmentation_model, (8, 8), data_format=IMAGE_ORDERING)

    depth_model = Conv2D(1, (3, 3), data_format=IMAGE_ORDERING,
               padding='same')(o)

    print(Model(img_input, depth_model).summary())
    depth_model = resize_image(depth_model, (8, 8), data_format=IMAGE_ORDERING)

    depth_model = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(depth_model)
    print(Model(img_input, depth_model).output_shape)

    # Simplified: View https://arxiv.org/pdf/1704.07813v2.pdf
    depth_model = Activation("tanh", name="depth_pred")(depth_model)

    model = get_segmentation_model(img_input, segmentation_model, depth_model)

    # Weights from https://arxiv.org/pdf/1705.07115.pdf
    loss_weights = {"depth_pred": 0.1, "segm_pred": 0.9}
    model.compile(loss=['categorical_crossentropy', depth_loss_function], loss_weights=loss_weights,
                  optimizer="adadelta",
                  metrics=['accuracy'])

    print(model.summary())
    return model


def pspnet(n_classes,  input_height=384, input_width=576):

    model = _pspnet(n_classes, vanilla_encoder,
                    input_height=input_height, input_width=input_width)
    model.model_name = "pspnet"
    return model


def get_triplets_from_paths(images_path, segs_path, deph_path, ignore_non_matching=False):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png" , ".bmp"]
    ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp"]

    image_files = []
    segmentation_files = {}
    depth_files = {}

    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension, os.path.join(images_path, dir_entry)))

    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            if file_name in segmentation_files:
                raise Exception("Segmentation file with filename {0} already exists and is ambiguous to resolve with path {1}. Please remove or rename the latter.".format(file_name, os.path.join(segs_path, dir_entry)))
            segmentation_files[file_name] = (file_extension, os.path.join(segs_path, dir_entry))

    for dir_entry in os.listdir(deph_path):
        if os.path.isfile(os.path.join(deph_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            if file_name in depth_files:
                raise Exception("Depth file with filename {0} already exists and is ambiguous to resolve with path {1}. Please remove or rename the latter.".format(file_name, os.path.join(segs_path, dir_entry)))
            depth_files[file_name] = (file_extension, os.path.join(deph_path, dir_entry))


    return_value = []
    # Match the images and segmentations
    i = 0
    for image_file, _, image_full_path in image_files:
        i += 1
        if image_file in segmentation_files and image_file in depth_files:
            return_value.append((image_full_path, segmentation_files[image_file][1], depth_files[image_file][1]))
        else:
            continue
        """else:
            # Error out
            raise Exception("No corresponding segmentation found for image {0}.".format(image_full_path))
        """
    return return_value

def get_depth_array(image_input, width, height):
    """ Load depth array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    else:
        if not os.path.isfile(image_input):
            raise Exception("get_segmentation_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    img = img[:, :, 0]
    img = np.reshape(img, (width, height, 1))

    return (img / 255) * 2 - 1

def image_segm_depth_generator(images_path, segs_path, depth_path, batch_size,
                               n_classes, input_height, input_width,
                               output_height, output_width,
                               do_augment=False):

    img_seg_depth_pairs = get_triplets_from_paths(images_path, segs_path, depth_path)
    random.shuffle(img_seg_depth_pairs)
    zipped = itertools.cycle(img_seg_depth_pairs)

    while True:
        X = []
        Y = []
        Y_segm = []
        Y_depth = []
        for _ in range(batch_size):
            im, seg, depth = next(zipped)

            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)
            depth = cv2.imread(depth, 1)

            X.append(get_image_array(im, input_width,
                                   input_height, ordering=IMAGE_ORDERING))
            Y_segm.append(get_segmentation_array(
                seg, n_classes, output_width, output_height))
            Y_depth.append(get_depth_array(depth, output_width, output_height))


        yield np.array(X), {
                "segm_pred": np.array(Y_segm),
                "depth_pred": np.array(Y_depth)
            }

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
          steps_per_epoch=512,
          do_augment=False
          ):
    input_height = 192
    input_width = 192

    n_classes = 8

    output_height = 48
    output_width = 48

    if validate:
        assert val_images is not None
        assert val_annotations_depth is not None
        assert val_annotations_segm is not None


    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)


    train_gen = image_segm_depth_generator(
        train_images, train_annotations_segm, train_annotations_depth,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width , do_augment=do_augment )

    if validate:
        val_gen = image_segm_depth_generator(
            val_images, val_annotations_segm, val_annotations_depth,  val_batch_size,
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
            model.fit_generator(train_gen, steps_per_epoch,
                                validation_data=val_gen,
                                validation_steps=200,  epochs=1)
            if checkpoints_path is not None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))
            print("Finished Epoch", ep)


def model_from_checkpoint_path(checkpoints_path):

    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = pspnet(
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    model.summary()
    model.load_weights(latest_weights)
    return model


def parse_depth_pred(pred):
    vmax = np.percentile(pred, 95)
    normalizer = mpl.colors.Normalize(vmin=pred.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(pred)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im

def predict(model=None, inp=None, out_fname=None, checkpoints_path=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)

    inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = 48
    output_height = 48
    input_width = 192
    input_height = 192
    n_classes = 8

    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    plot_model(model, "model.png")
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

    return pr
