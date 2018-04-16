#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""File containing the hyper-parameters for itergan.py

    by: Ysbrand Galama
    latest version: 2018-03-19
    build for python 3.6 and TensorFlow 1.0
"""

import tensorflow as tf

NGF = 64
NDF = 64
USE_BIAS = False
INITIALIZER = 'normal'
LR = 0.0002
BETA1 = 0.5
LAMBDA_L1 = 100.0
LAMBDA_G2 = 0.1
LAMBDA_GAN = 1.0

EPS = 1e-12

CONV_INIT_XAVIER = tf.contrib.layers.xavier_initializer_conv2d()
CONV_INIT_MU = 0
CONV_INIT_STD_SMALL = 0.005
CONV_INIT_STD = 0.02
CONV_KW = 4
CONV_KH = 4
CONV_BIAS_INIT = tf.constant_initializer(0.0)

BATCHNORM_INITIALIZER = tf.random_normal_initializer(1.0, 0.02)
BATCHNORM_VARIANCE_EPS = 1e-5

GEN_LRELU = 0.2
GEN_DROPOUT = 0.5
DIS_LRELU = 0.2

WMAD_FOREGROUND_FRAQ = 2/3
EMA_DECAY = 0.99

MAX_STEPS = 2**32
