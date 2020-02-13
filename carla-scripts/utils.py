import os

from keras.backend import set_session, tf


def get_model_params(model):
    """
    Extracts some info about the model from its input and output layers
    """

    forward_image_input_layer = model.get_layer('forward_image_input')
    steer_pred_output_layer = model.get_layer('steer_pred')

    if len(forward_image_input_layer.input_shape) == 5:
        (_, sequence_length, height, width, channels) = forward_image_input_layer.input_shape
    else:
        # CNN
        (_, height, width, channels) = forward_image_input_layer.input_shape
        sequence_length = None

    sine_steering = steer_pred_output_layer.output_shape[1] > 1
    return (height, width, channels), sequence_length, sine_steering


def init_tensorflow():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    set_session(sess)
