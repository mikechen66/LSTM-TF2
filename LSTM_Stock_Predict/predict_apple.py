#!/usr/bin/env python
# coding: utf-8

# predict_apple.py
"""
Please note that users can either run the script of predict_apple.py from the current directoy
or from the path of demos. 

$ python predict_apple.py

The difference is that both the relative path of sys are commenced in the current directory. 
"""


import sys, os
# -sys.path.append('..')
from lstm_call import StockModel
import tensorflow as tf 
# -os.chdir('..')


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Select the Apple stock for demonstration 
aapl_model = StockModel('AAPL')
aapl_model.loadStock()
model, history = aapl_model.train()
rmse = aapl_model.validate(model)
aapl_model.plotOneDayCurve(model)
aapl_model.plotFutureCurves(model)
aapl_model.plotBuySellPoints(model)
aapl_model.plotPortfolioReturn(model)