from keras_segmentation.data_utils.augmentation import augment_seg
from keras_segmentation.data_utils.data_loader import get_image_array, get_segmentation_array
from keras_segmentation.models.config import IMAGE_ORDERING
import numpy as np
import tensorflow as tf
import keras.backend as K
import os
import random
import itertools
import cv2
import matplotlib as mpl
import matplotlib.cm as cm


if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1




# From https://github.com/ialhashim/DenseDepth/blob/ed044069eb99fa06dd4af415d862b3b5cbfab283/loss.py#L4
def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1):
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    print("y_true", y_true, y_true.shape)
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


def get_triplets_from_paths(images_path, segs_path, deph_path, ignore_non_matching=False):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
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
                raise Exception(
                    "Segmentation file with filename {0} already exists and is ambiguous to resolve with path {1}. Please remove or rename the latter.".format(
                        file_name, os.path.join(segs_path, dir_entry)))
            segmentation_files[file_name] = (file_extension, os.path.join(segs_path, dir_entry))

    for dir_entry in os.listdir(deph_path):
        if os.path.isfile(os.path.join(deph_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            if file_name in depth_files:
                raise Exception(
                    "Depth file with filename {0} already exists and is ambiguous to resolve with path {1}. Please remove or rename the latter.".format(
                        file_name, os.path.join(segs_path, dir_entry)))
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
            raise Exception("get_depth_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    img = img[:, :, 0]
    img = np.reshape(img, (width, height, 1))

    return (img / 255)


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

            if do_augment:
                im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0])

            X.append(get_image_array(im, input_width,
                                     input_height, ordering=IMAGE_ORDERING))
            Y_segm.append(get_segmentation_array(
                seg, n_classes, output_width, output_height))
            Y_depth.append(get_depth_array(depth, output_width, output_height))

        yield np.array(X), {
            "segm_pred": np.array(Y_segm),
            "depth_pred": np.array(Y_depth)
        }



MIN_DEPTH = 1e-3
MAX_DEPTH = 80


def parse_depth_pred(pred):
    pred[pred > MAX_DEPTH] = MAX_DEPTH
    pred[pred < MIN_DEPTH] = MIN_DEPTH
    pred[pred > MAX_DEPTH] = MAX_DEPTH
    pred[pred < MIN_DEPTH] = MIN_DEPTH

    vmax = np.percentile(pred, 95)
    normalizer = mpl.colors.Normalize(vmin=pred.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(pred)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im


