#!/usr/bin/env python
# coding: utf-8

# model.py
"""
LSTM adopts the Keras Python package to predict time series steps and sequences, including 
sine wave and stock market data. The model is based on the LSTM that is widely used to predict 
the timeline event in the universe. Please note that it is slower to call on the LSTM from 
core.recurrent_v2 than keras.layers. 
"""


import os
import math
import numpy as np
import datetime as dt

from numpy import newaxis
from keras.layers import Dense, Activation, Dropout
from core.recurrent_v2 import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Timer():

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()

        print('Time taken: %s' % (end_dt - self.start_dt))


class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')

        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                     ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)]
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
        
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)]
        self.model.fit(data_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks, workers=1)
        
        print('[Model] Training Completed. Model saved as %s' % save_fname)

        timer.stop()

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data with only 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        pred = self.model.predict(data)
        pred = np.reshape(pred, (pred.size,))

        return pred

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            pred = []
            for j in range(prediction_len):
                pred.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], pred[-1], axis=0)
            prediction_seqs.append(pred)

        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        pred = []

        for i in range(len(data)):
            pred.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], pred[-1], axis=0)

        return pred