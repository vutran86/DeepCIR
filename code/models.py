import numpy as np
import pickle
import datetime
import glob
from scipy.constants import speed_of_light
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, Conv1D, Flatten, Dense, MaxPooling1D, AveragePooling1D, UpSampling1D, Multiply, Reshape, Add, Activation, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping

class WeightedSum(Layer):
    def __init__(self, w1, w2, w3, **kwargs):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        super(WeightedSum, self).__init__(**kwargs)
    def call(self, model_outputs):
        return self.w1 * model_outputs[0] + self.w2 * model_outputs[1] + self.w3 * model_outputs[2]
    def compute_output_shape(self, input_shape):
        return input_shape[0]

def uwbSimple():
    cir_input = Input(shape=(64, 2), name="cir")
    
    cirNet = Sequential()
    cirNet.add(Conv1D(8, 9, activation='relu', input_shape=(64,2,)))
    cirNet.add(Conv1D(16, 9, activation='relu'))
    cirNet.add(Conv1D(32, 7, activation='relu'))
    cirNet.add(Conv1D(64, 7, activation='relu'))
    cirNet.add(Conv1D(128, 5, activation='relu'))
    cirNet.add(Conv1D(256, 5, activation='relu'))
    cirNet.add(Flatten())
    cirNet.add(Dense(64, activation='relu'))
    cirNet.add(Dropout(0.35))
    cirNet.add(Dense(32, activation='relu'))
    cirNet.add(Dropout(0.35))
    cirNet.add(Dense(1))

    total = cirNet(cir_input)

    model = Model(inputs=cir_input, outputs=total, name="cirnet")
    return model

def uwbSimpleV2():
    cir_input = Input(shape=(64, 2), name="cir")
    
    cirNet = Sequential()
    cirNet.add(Conv1D(8, 9, activation='relu', input_shape=(64,2,), kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(16, 9, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(32, 7, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(64, 7, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(256, 5, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Flatten())
    cirNet.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Dropout(0.35))
    cirNet.add(Dense(32, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Dropout(0.35))
    cirNet.add(Dense(1, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))

    total = cirNet(cir_input)

    model = Model(inputs=cir_input, outputs=total, name="cirnet")
    return model

def uwbShared():
    poll_input = Input(shape=(64, 2), name="poll")
    resp_input = Input(shape=(64, 2), name="resp")
    final_input = Input(shape=(64, 2), name="final")
    
    cirNet = Sequential()
    cirNet.add(Conv1D(8, 9, activation='relu', input_shape=(64,2,)))
    cirNet.add(Conv1D(16, 9, activation='relu'))
    cirNet.add(Conv1D(32, 7, activation='relu'))
    cirNet.add(Conv1D(64, 7, activation='relu'))
    cirNet.add(Conv1D(128, 5, activation='relu'))
    cirNet.add(Conv1D(256, 5, activation='relu'))
    cirNet.add(Flatten())
    cirNet.add(Dense(64, activation='relu'))
    cirNet.add(Dropout(0.35))
    cirNet.add(Dense(32, activation='relu'))
    cirNet.add(Dropout(0.35))
    cirNet.add(Dense(1))

    fea_poll = cirNet(poll_input)
    fea_resp = cirNet(resp_input)
    fea_final = cirNet(final_input)

    total = WeightedSum(0.07812500000000135, 0.4218750000000001, 0.4999999999999985)([fea_final, fea_poll, fea_resp])

    model = Model(inputs=[final_input, poll_input, resp_input], outputs=total, name="cirnet")
    return model

def uwbSharedV2():
    poll_input = Input(shape=(64, 2), name="poll")
    resp_input = Input(shape=(64, 2), name="resp")
    final_input = Input(shape=(64, 2), name="final")
    
    cirNet = Sequential()
    cirNet.add(Conv1D(8, 9, activation='relu', input_shape=(64,2,), kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(16, 9, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(32, 7, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(64, 7, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(256, 5, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Flatten())
    cirNet.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Dropout(0.35))
    cirNet.add(Dense(32, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Dropout(0.35))
    cirNet.add(Dense(1, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))

    fea_poll = cirNet(poll_input)
    fea_resp = cirNet(resp_input)
    fea_final = cirNet(final_input)

    total = WeightedSum(0.07812500000000135, 0.4218750000000001, 0.4999999999999985)([fea_final, fea_poll, fea_resp])

    model = Model(inputs=[final_input, poll_input, resp_input], outputs=total, name="cirnet")
    return model

def uwbCombined():
    cir_input = Input(shape=(64, 6), name="com")
    
    cirNet = Sequential()
    cirNet.add(Conv1D(24, 9, activation='relu', input_shape=(64,6,)))
    cirNet.add(Conv1D(48, 9, activation='relu'))
    cirNet.add(Conv1D(96, 7, activation='relu'))
    cirNet.add(Conv1D(192, 7, activation='relu'))
    cirNet.add(Conv1D(384, 5, activation='relu'))
    cirNet.add(Conv1D(768, 5, activation='relu'))
    cirNet.add(Flatten())
    cirNet.add(Dense(128, activation='relu'))
    cirNet.add(Dropout(0.35))
    cirNet.add(Dense(64, activation='relu'))
    cirNet.add(Dropout(0.35))
    cirNet.add(Dense(1))

    out = cirNet(cir_input)

    model = Model(inputs=cir_input, outputs=out, name="cirnet")
    return model

def uwbCombinedV2():
    poll_input = Input(shape=(64, 2), name="poll")
    resp_input = Input(shape=(64, 2), name="resp")
    final_input = Input(shape=(64, 2), name="final")
    
    cirNet = Sequential()
    cirNet.add(Conv1D(8, 9, activation='relu', input_shape=(64,2,), kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(16, 9, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(32, 7, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(64, 7, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(256, 5, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))

    fea_poll = cirNet(poll_input)
    fea_resp = cirNet(resp_input)
    fea_final = cirNet(final_input)

    total = tf.concat([fea_final, fea_poll, fea_resp], axis=-1)

    total = Flatten()(total)
    total = Dropout(0.35)(total)
    total = Dense(128, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)
    total = Dropout(0.35)(total)
    total = Dense(64, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)
    total = Dropout(0.35)(total)
    total = Dense(1, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)

    model = Model(inputs=[final_input, poll_input, resp_input], outputs=total, name="cirnet")
    return model

def uwbCombined2In():
    poll_input = Input(shape=(64, 2), name="poll")
    resp_input = Input(shape=(64, 2), name="resp")
    
    cirNet = Sequential()
    cirNet.add(Conv1D(8, 9, activation='relu', input_shape=(64,2,), kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(16, 9, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(32, 7, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(64, 7, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(256, 5, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))

    fea_poll = cirNet(poll_input)
    fea_resp = cirNet(resp_input)

    total = tf.concat([fea_poll, fea_resp], axis=-1)

    total = Flatten()(total)
    total = Dropout(0.35)(total)
    total = Dense(128, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)
    total = Dropout(0.35)(total)
    total = Dense(64, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)
    total = Dropout(0.35)(total)
    total = Dense(1, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)

    model = Model(inputs=[poll_input, resp_input], outputs=total, name="cirnet")
    return model

def uwbCombined1In():
    poll_input = Input(shape=(64, 2), name="poll")
    
    cirNet = Sequential()
    cirNet.add(Conv1D(8, 9, activation='relu', input_shape=(64,2,), kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(16, 9, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(32, 7, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(64, 7, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(256, 5, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))

    total = cirNet(poll_input)

    total = Flatten()(total)
    total = Dropout(0.35)(total)
    total = Dense(128, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)
    total = Dropout(0.35)(total)
    total = Dense(64, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)
    total = Dropout(0.35)(total)
    total = Dense(1, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)

    model = Model(inputs=[poll_input], outputs=total, name="cirnet")
    return model

def uwbAE(): #Adapted from Fontaine's approach
    cir_input = Input(shape=(64, 2), name="cir")
    
    encoder = Sequential()
    encoder.add(Conv1D(32, 9, activation='relu', padding='same', input_shape=(64,2,), kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    encoder.add(AveragePooling1D(pool_size=4))
    encoder.add(Conv1D(64, 9, activation='relu', padding='same', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    encoder.add(Conv1D(128, 7, activation='relu', padding='same', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    encoder.add(Conv1D(256, 7, activation='relu', padding='same', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    encoder.add(Conv1D(512, 5, activation='relu', padding='same', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    fea = encoder(cir_input)

    decoder = Sequential()
    decoder.add(Conv1D(256, 7, activation='relu', padding='same', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    decoder.add(Conv1D(128, 7, activation='relu', padding='same', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    decoder.add(Conv1D(64, 9, activation='relu', padding='same', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    decoder.add(UpSampling1D(size=4))
    decoder.add(Conv1D(32, 9, activation='relu', padding='same', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    decoder.add(Conv1D(2, 9, activation='relu', padding='same', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))

    total = Flatten()(fea)
    total = Dense(64, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)
    total = Dropout(0.35)(total)
    total = Dense(32, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)
    total = Dropout(0.35)(total)
    total = Dense(1, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)

    denoise = decoder(fea)

    model = Model(inputs=cir_input, outputs=[total, denoise], name="cirnet")
    return model

def uwbOrig():
    cir_input = Input(shape=(64, 2), name="cir")
    
    cirNet = Sequential()
    cirNet.add(Conv1D(32, 16, activation='relu', input_shape=(64,2,), kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(MaxPooling1D(pool_size=2))
    cirNet.add(Conv1D(64, 8, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(MaxPooling1D(pool_size=2))
    cirNet.add(Conv1D(128, 2, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Flatten())
    cirNet.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Dropout(0.35))
    cirNet.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Dropout(0.25))
    cirNet.add(Dense(32, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Dropout(0.15))
    cirNet.add(Dense(16, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    # cirNet.add(Dropout(0.35))
    cirNet.add(Dense(8, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    # cirNet.add(Dropout(0.35))
    cirNet.add(Dense(1, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))

    total = cirNet(cir_input)

    model = Model(inputs=cir_input, outputs=total, name="cirnet")
    return model

def uwbMultiAnc():
    poll_input0 = Input(shape=(64, 2), name="poll0")
    resp_input0 = Input(shape=(64, 2), name="resp0")
    final_input0 = Input(shape=(64, 2), name="final0")

    poll_input1 = Input(shape=(64, 2), name="poll1")
    resp_input1 = Input(shape=(64, 2), name="resp1")
    final_input1 = Input(shape=(64, 2), name="final1")
    
    cirNet = Sequential()
    cirNet.add(Conv1D(8, 9, activation='relu', input_shape=(64,2,), kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(16, 9, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(32, 7, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(64, 7, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    cirNet.add(Conv1D(256, 5, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))

    fea_poll0 = cirNet(poll_input0)
    fea_resp0 = cirNet(resp_input0)
    fea_final0 = cirNet(final_input0)

    fea_poll1 = cirNet(poll_input1)
    fea_resp1 = cirNet(resp_input1)
    fea_final1 = cirNet(final_input1)

    total = tf.concat([fea_final0, fea_poll0, fea_resp0, fea_final1, fea_poll1, fea_resp1], axis=-1)

    total = Flatten()(total)
    total = Dropout(0.35)(total)
    total = Dense(128, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)
    total = Dropout(0.35)(total)
    total = Dense(64, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)
    total = Dropout(0.35)(total)
    total = Dense(1, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))(total)

    model = Model(inputs=[final_input0, poll_input0, resp_input0, final_input1, poll_input1, resp_input1], outputs=total, name="cirnet")
    return model


