from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout, Conv2D, concatenate, \
    TimeDistributed, CuDNNLSTM, LSTM, BatchNormalization


def get_hegemax_model(seq_length, print_summary=True):
    forward_image_input = Input(shape=(seq_length, 160, 350, 3), name="forward_image_input")
    info_input = Input(shape=(seq_length, 3), name="info_input")
    hlc_input = Input(shape=(seq_length, 6), name="hlc_input")

    x = TimeDistributed(Cropping2D(cropping=((50, 0), (0, 0))))(forward_image_input)
    x = TimeDistributed(Lambda(lambda x: ((x / 255.0) - 0.5)))(x)
    x = TimeDistributed(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))(x)
    x = TimeDistributed(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))(x)
    x = TimeDistributed(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2), activation="relu"))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation="relu"))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation="relu"))(x)
    conv_output = TimeDistributed(Flatten())(x)

    x = concatenate([conv_output, info_input, hlc_input])

    x = TimeDistributed(Dense(100, activation="relu"))(x)
    x = CuDNNLSTM(10, return_sequences=False)(x)
    steer_pred = Dense(10, activation="tanh", name="steer_pred")(x)

    x = TimeDistributed(Cropping2D(cropping=((50, 0), (0, 0))))(forward_image_input)
    x = TimeDistributed(Lambda(lambda x: ((x / 255.0) - 0.5)))(x)
    x = TimeDistributed(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))(x)
    x = TimeDistributed(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))(x)
    x = TimeDistributed(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2), activation="relu"))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation="relu"))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation="relu"))(x)
    conv_output = TimeDistributed(Flatten())(x)

    x = concatenate([conv_output, info_input, hlc_input])

    x = TimeDistributed(Dense(100, activation="relu"))(x)
    x = CuDNNLSTM(10, return_sequences=False)(x)
    throtte_pred = Dense(1, name="throttle_pred")(x)
    brake_pred = Dense(1, name="brake_pred")(x)

    model = Model(inputs=[forward_image_input, info_input, hlc_input], outputs=[steer_pred, throtte_pred, brake_pred])

    if print_summary:
        model.summary()

    return model
