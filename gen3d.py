#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main method for implicit 3d representation learning based on pix2pix*

    To use this program, run ?> python gen3d.py --help

    used for implicit 3d representation learning based on pix2pix

    *Pix2pix: Isola, P., Zhu, J.-Y., Zhou, T. and Efros, A. A. (2016),
              'Image-to-image translation with conditional adversarial
              networks', arxiv

    code adapted from https://github.com/affinelayer/pix2pix-tensorflow

    by: Ysbrand Galama
    latest version: 2017-09-18
    build for python 3.6 and TensorFlow 1.0
"""

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import threading
# from scipy.linalg import svd
# from scipy.misc import imread
from itertools import cycle

from tensorflow.python import pywrap_tensorflow

import sys
assert sys.version_info >= (3, 6), 'Please update your python!'


# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _build_parser():
    """Built the argument parser from argparse"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='path to base folder of ALOI')
    parser.add_argument('--mode', required=True,
                        choices=['train', 'test', 'validate'])
    parser.add_argument('--output_dir', required=True,
                        help='where to put output files')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--checkpoint', default=None,
                        help='directory with checkpoint to resume training fro'
                        'm or use for testing')
    parser.add_argument('--datatype', default='basic_train', help='name of the'
                        ' iterator to use as data input')

    parser.add_argument('--max_steps', type=int,
                        help='number of training steps (0 to disable)')
    parser.add_argument('--max_epochs', type=int,
                        help='number of training epochs')
    parser.add_argument('--summary_freq', type=int, default=1000,
                        help='update summaries every summary_freq steps')
    parser.add_argument('--progress_freq', type=int, default=50,
                        help='display progress every progress_freq steps')
    parser.add_argument('--trace_freq', type=int, default=0,
                        help='trace execution every trace_freq steps')
    parser.add_argument('--display_freq', type=int, default=0,
                        help='write current training images every display_freq'
                        ' steps')
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='save model every save_freq steps, 0 to disable')
    parser.add_argument('--epoch_freq', type=int, default=2,
                        help='save model every epoch_freq steps in epoch dir')

    parser.add_argument('--step_num', type=int, default=6,
                        help='number of iterations for generator')
    parser.add_argument('--gpu_frac', type=float, default=0,
                        help='how much of the gpu can be used')
    parser.add_argument('--init', default=None, help='change initialisation')
    parser.add_argument('--mmad_loss', type=str2bool, default=False,
                        help='whether to use MMAD instead of L1')

    parser.add_argument('--sample_lambda', type=float, default=0,
                        help='if value > 0, adds a discriminator to each step '
                        'that uses single images')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of images in batch')

    parser.add_argument('--ngf', type=int, default=64,
                        help='number of generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64,
                        help='number of discriminator filters in first conv la'
                        'yer')
    parser.add_argument('--use_bias', type=str2bool, default=False,
                        help='whether to use bias in conv layer')

    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate for adam')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='momentum term of adam')
    parser.add_argument('--l1_weight', type=float, default=100.0,
                        help='weight on L1 term for generator gradient')
    parser.add_argument('--gan_weight', type=float, default=1.0,
                        help='weight on GAN term for generator gradient')

    return parser


a = None  # the parsed arguments, after calling main()
epoch = None

EPS = 1e-12
OBJECT_QUEUE = None
KINDS = ['inputs', 'outputs', 'targets', 'difference', 'between']

Examples = collections.namedtuple('Examples', 'paths, inputs, targets, count,'
                                  ' steps_per_epoch, iters, target_masks')
Model = collections.namedtuple('Model', 'outputs, predict_real, iters,'
                               ' predict_fake, discrim_loss,'
                               ' discrim_grads_and_vars, gen_loss_GAN,'
                               ' gen_loss_L1, gen_grads_and_vars, train, rest')

# ==== Generators for inputs and global vars
RAW_SHAPE = [288, 348, 3]
IMG_SHAPE = [256, 256, 3]
DATA_FILE = '{dir}/aloi_red2_view/png2/{id}/{id}_r{rot}.png'
MASK_FILE = '{dir}/aloi_mask2/mask2/{id}/{id}_r{rot}.png'
MASK_FILE0 = '{dir}/aloi_mask2/mask2/{id}/{id}_c1.png'


def get_generator(name):
    """Get an iterator to construct the necessary input files"""
    def basic_train():
        yield from ((id, x, 6) for id in range(1, 801)
                    for x in range(0, 72, 2))

    def basic_test(r=6):
        yield from ((id, x, r) for id in range(700, 800)
                    for x in range(1, 72, 2))
        yield from ((id, x, r) for id in range(801, 901)
                    for x in range(1, 72, 2))

    def step_wise_train():
        t = a.max_epochs // 4
        subs = [(1,)]*t + [(1, 2)]*t + [(1, 2, 3, 4, 5, 6)]*t + \
            [range(1, 37)]*(t+1)
        for rng in subs:
            yield from ((id, x, i) for id in range(1, 801)
                        for x in range(1, 72, 2*len(rng))
                        for i in rng)

    def varied_rot_train():
        yield from ((id, x, i) for id in range(1, 801, 3)
                    for x in range(1, 72, 8)
                    for i in range(1, 37, 3))

    def very_small():
        yield from ((id, x, i) for id in [1, 2] for x in [0, 18]
                    for i in [2, 8])

    def very_small_test():
        yield from ((id, x, i) for id in [1, 2] for x in [0, 18]
                    for i in [2, 8])
        yield from ((id, x, i) for id in [1, 2] for x in [0, 18]
                    for i in [2, 8])

    return {
        'basic_train': (cycle(basic_train()), 28800),
        'basic_test': (cycle(basic_test()), 3600),
        'rotate_test': (cycle(basic_test(18)), 3600),
        'step_wise_train': (step_wise_train(), 28800),
        'varied_rot_train': (cycle(varied_rot_train()), 28836),
        'very_small': (cycle(very_small()), 8),
        'very_small_test': (cycle(very_small_test()), 8),
    }[name]


# ==== Default ops
def resize(image, from_w, from_h, to_s, name='resize'):
    """Pad an image to become square, and then resize it.

    Arguments
    ---------
    image - tf.tensor
        The image of shape (width, height, channels)
    from_w - int
        The original width in pixels
    from_f - int
        The original height in pixels
    to_s - int
        The dimension of the new image

    Returns
    -------
    tf.tensor - The resized image"""
    with tf.variable_scope(name):
        if from_w < from_h:
            N = (from_h-from_w)//2
            padding = tf.constant([[N, N], [0, 0], [0, 0]], dtype=tf.int32)
        else:
            N = (from_w-from_h)//2
            padding = tf.constant([[0, 0], [N, N], [0, 0]], dtype=tf.int32)
        tmp = tf.pad(image, padding, 'CONSTANT')
        return tf.image.resize_images(tmp, [to_s]*2)


def preprocess(raw, mask=False, name='preprocess'):
    """From raw data, create a tensor of correct shape and dtype."""
    with tf.name_scope(name):
        raw = tf.image.convert_image_dtype(raw, dtype=tf.float32)
        raw.set_shape(RAW_SHAPE)
        raw = resize(raw, RAW_SHAPE[0], RAW_SHAPE[1], IMG_SHAPE[0])
        raw = tf.cast(raw, tf.bool) if mask else raw * 2 - 1
        return raw


def deprocess(image):
    """Change the range of an image from [-1, 1] to [0, 1]"""
    with tf.name_scope('deprocess'):
        return (image + 1) / 2


def conv(batch_input, out_channels, stride):
    """create tensorflow convolutions

    * Does not use a bias, as in the original lua code of pix2pix

    Arguments
    ---------
    batch_input - tf.tensor
        The input tensor of this layer
    out_channels - int
        The number of channels in the output image
    stride - int
        The stride of the filter

    Returns
    -------
    np.tensor - The tensor after convolution
    """
    with tf.variable_scope('conv'):
        if a.init == 'xavier':
            initial = tf.contrib.layers.xavier_initializer_conv2d()
        elif a.init == 'small':
            initial = tf.random_normal_initializer(0, 0.005)
        else:
            initial = tf.random_normal_initializer(0, 0.02)
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable('filter', [4, 4, in_channels, out_channels],
                                 dtype=tf.float32,
                                 initializer=initial
                                 )
        # [batch, in_height, in_width, in_channels],
        #  [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]],
                              mode='CONSTANT')
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1],
                            padding='VALID')
        if a.use_bias:
            biases = tf.get_variable('biases', [out_channels],
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def lrelu(x, a):
    """Implementation of Leaky ReLU

    Arguments
    ---------
    x - tf.tensor
        The input tensor
    a - float
        The alpha parameter of the Leaky ReLU

    Returns
    -------
    tf.tensor - The tensor after the relu
    """
    with tf.name_scope('lrelu'):
        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    """Performs instance normalisation using the tf.nn.batch_normalization

    Arguments
    ---------
    input - tf.tensor
        The input tensor

    Returns
    -------
    tf.tensor - the normalised output
    """
    with tf.variable_scope('batchnorm'):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable('offset', [channels], dtype=tf.float32,
                                 initializer=tf.zeros_initializer())
        scale = tf.get_variable('scale', [channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0,
                                                                         0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_eps = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset,
                                               scale,
                                               variance_epsilon=variance_eps)
        return normalized


def deconv(batch_input, out_channels):
    """Create transposed_convolution aka deconvolution layer

    Arguments
    ---------
    batch_input - tf.tensor
        The input of the layer
    out_channels - int
        The amount of channels of the output layer

    Returns
    -------
    tf.tensor - the output of the layer
    """
    with tf.variable_scope('deconv'):
        if a.init == 'xavier':
            initial = tf.contrib.layers.xavier_initializer_conv2d()
        elif a.init == 'small':
            initial = tf.random_normal_initializer(0, 0.005)
        else:
            initial = tf.random_normal_initializer(0, 0.02)

        batch, in_height, in_width, in_channels = [int(d) for d in
                                                   batch_input.get_shape()]
        filter = tf.get_variable('filter', [4, 4, out_channels, in_channels],
                                 dtype=tf.float32,
                                 initializer=initial
                                 )
        # [batch, in_height, in_width, in_channels],
        #  [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter,
                                      [batch, in_height*2, in_width*2,
                                       out_channels],
                                      [1, 2, 2, 1], padding='SAME')
        if a.use_bias:
            biases = tf.get_variable('biases', [out_channels],
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


# ==== Model functions
def init_model(epoch_size):
    """Create data queue for training.

    Returns
    -------
    Examples named tuple - object containing the necessarily information of the
        queue
    """
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception('input_dir does not exist')

    with tf.name_scope('queue_fill'):
        # The image placeholders
        with tf.name_scope('file_inputs'):
            paths = tf.placeholder(tf.string, shape=())
            input_data = tf.placeholder(tf.string, shape=())
            target_data = tf.placeholder(tf.string, shape=())
            mask_data = tf.placeholder(tf.string, shape=())
            iters = tf.placeholder(tf.int32, shape=())

        input_image = preprocess(tf.image.decode_png(input_data, channels=3))
        target_image = preprocess(tf.image.decode_png(target_data, channels=3))
        input_mask = preprocess(tf.image.decode_png(mask_data, channels=3), 1)

        # The tf queue that handles the reading of images
        q = tf.FIFOQueue(32, [tf.string, tf.float32, tf.float32, tf.bool,
                              tf.int32],
                         shapes=[[], IMG_SHAPE, IMG_SHAPE, IMG_SHAPE, []])
        enqueue_op = q.enqueue([paths, input_image, target_image, input_mask,
                                iters])
    paths_batch, inputs_batch, targets_batch, masks_batch, iters_batch = \
        q.dequeue_many(a.batch_size)

    steps_per_epoch = int(math.ceil(epoch_size / a.batch_size))
    examples = Examples(paths=paths_batch, inputs=inputs_batch,
                        targets=targets_batch, iters=iters_batch,
                        target_masks=masks_batch, count=epoch_size,
                        steps_per_epoch=steps_per_epoch)

    model = create_model(examples.inputs, examples.targets,
                         examples.target_masks, examples.iters)

    def load_and_enqueue():
        try:
            while True:
                id, x, i = next(OBJECT_QUEUE)
                rot = x*5
                irot = (rot + i*5) % 360
                input_file = DATA_FILE.format(dir=a.input_dir, id=id, rot=rot)
                tgt_file = DATA_FILE.format(dir=a.input_dir, id=id, rot=irot)
                if irot == 0:
                    mask_file = MASK_FILE0.format(dir=a.input_dir, id=id)
                else:
                    mask_file = MASK_FILE.format(dir=a.input_dir, id=id,
                                                 rot=irot)
                path = f'{id}_r{rot}-r{irot}.png'
                with open(input_file, 'rb') as fi, open(tgt_file, 'rb') as ft,\
                        open(mask_file, 'rb') as fm:
                    sess.run(enqueue_op, feed_dict={paths: path,
                                                    input_data: fi.read(),
                                                    target_data: ft.read(),
                                                    mask_data: fm.read(),
                                                    iters: i})
        except tf.errors.CancelledError:
            print('WARNING load_queue stopped')

    return examples, model, load_and_enqueue


def create_generator(generator_inputs, generator_outputs_channels):
    """Create the generator of pix2pix

    Arguments
    ---------
    generator_inputs - tf.tensor
        The input of the generator
    generator_outputs_channels - int
        The number of channels of the generator (always 3)

    Returns
    -------
    tf.tensor - the output of the generator
    """
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope('encoder_1'):
        output = conv(generator_inputs, a.ngf, stride=2)
        layers.append(output)

    layer_specs = [
        (a.ngf * 2, True),  # e_2: [b, 128, 128, ngf] => [b, 64, 64, ngf * 2]
        (a.ngf * 4, True),  # e_3: [b, 64, 64, ngf * 2] => [b, 32, 32, ngf * 4]
        (a.ngf * 8, True),  # e_4: [b, 32, 32, ngf * 4] => [b, 16, 16, ngf * 8]
        (a.ngf * 8, True),  # e_5: [b, 16, 16, ngf * 8] => [b, 8, 8, ngf * 8]
        (a.ngf * 8, True),  # e_6: [b, 8, 8, ngf * 8] => [b, 4, 4, ngf * 8]
        (a.ngf * 8, True),  # e_7: [b, 4, 4, ngf * 8] => [b, 2, 2, ngf * 8]
        (a.ngf * 8, False)  # e_8: [b, 2, 2, ngf * 8] => [b, 1, 1, ngf * 8]
    ]

    for out_channels, do_bn in layer_specs:
        with tf.variable_scope('encoder_%d' % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] =>
            #  [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            if do_bn:
                output = batchnorm(convolved)
            else:
                output = tf.identity(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),  # d_8: [b, 1, 1, ngf*8] => [b, 2, 2, ngf*8*2]
        (a.ngf * 8, 0.5),  # d_7: [b, 2, 2, ngf*8*2] => [b, 4, 4, ngf*8*2]
        (a.ngf * 8, 0.5),  # d_6: [b, 4, 4, ngf*8*2] => [b, 8, 8, ngf*8*2]
        (a.ngf * 8, 0.0),  # d_5: [b, 8, 8, ngf*8*2] => [b, 16, 16, ngf*8*2]
        (a.ngf * 4, 0.0),  # d_4: [b, 16, 16, ngf*8*2] => [b, 32, 32, ngf*4*2]
        (a.ngf * 2, 0.0),  # d_3: [b, 32, 32, ngf*4*2] => [b, 64, 64, ngf*2*2]
        (a.ngf, 0.0),      # d_2: [b, 64, 64, ngf*2*2] => [b, 128, 128, ngf*2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope('decoder_%d' % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] =>
            #  [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] =>
    #  [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope('decoder_1'):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_discriminator(discrim_inputs, discrim_targets=None, n_out=1):
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] =>
    #  [batch, height, width, in_channels * 2]
    if discrim_targets is None:
        input = discrim_inputs
    else:
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope('layer_1'):
        convolved = conv(input, a.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope('layer_%d' % (len(layers) + 1)):
            out_channels = a.ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last has stride 1
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope('layer_%d' % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=n_out, stride=1)
        if n_out == 1:
            output = tf.sigmoid(convolved)
        else:
            output = tf.nn.softmax(convolved)
        layers.append(output)

    return layers[-1]


def IterGAN(inputs, out_channels, iter_num, name='IterGAN'):
    if a.batch_size != 1:
        print('WARNING, current code only works correctly with batch=1. (or if'
              ' the variable-iters are synchronised with the batch-size')

    with tf.name_scope(name):
        max_iter = tf.reduce_max(iter_num)
        with tf.variable_scope('generator'):
            def body(i, inp, iters):
                out = create_generator(inp, out_channels)
                next_iters = tf.concat([iters, out], axis=2,
                                       name='between_steps')
                return i+1, out, next_iters

            def condition(i, inp, iters):
                return i < max_iter

            i0 = tf.constant(0)
            it_shape = tf.TensorShape([a.batch_size, IMG_SHAPE[0], None,
                                       IMG_SHAPE[2]])
            img_shape = inputs.get_shape()
            _, outputs, iters = tf.while_loop(
                condition, body, (i0, inputs, inputs),
                shape_invariants=(i0.get_shape(), img_shape, it_shape))
    return outputs, iters, img_shape, max_iter


def create_model(inputs, targets, target_masks, iter_num):
    """Create the full model for training/testing
    """
    out_channels = int(targets.get_shape()[-1])
    assert out_channels == int(inputs.get_shape()[-1])

    outputs, iters, img_shape, max_iter = IterGAN(inputs, out_channels,
                                                  iter_num, name='')

    rest = {}

    if a.sample_lambda > 0.0:
        with tf.name_scope('sample_steps'):
            i = tf.random_uniform((), maxval=max_iter, dtype=tf.int32)
            j = tf.random_uniform((), maxval=2, dtype=tf.int32)
            real_imgs = tf.stack([inputs, targets], name='real_imgs')
            d = IMG_SHAPE[1]
            with tf.name_scope('sample_fake'):
                sample_fake = iters[:, :, (i+1)*d:(i+2)*d, :]
                sample_fake.set_shape(img_shape)
            sample_real = real_imgs[j]
            with tf.variable_scope('disciminator_sample'):
                predict_sample_real = create_discriminator(sample_real)
            with tf.variable_scope('disciminator_sample', reuse=True):
                predict_sample_fake = create_discriminator(sample_fake)
            rest['sample'] = {'i': i,
                              'j': j,
                              'predict_real': predict_sample_real,
                              'predict_fake': predict_sample_fake,
                              'real_inp': sample_real,
                              'fake_inp': sample_fake
                              }

    # create two copies of discriminator, one for real and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator'):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope('discriminator_loss'):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) +
                                      tf.log(1 - predict_fake + EPS)))
        if a.sample_lambda > 0.0:
            discim_loss = discrim_loss + a.sample_lambda * \
                tf.reduce_mean(-(tf.log(predict_sample_real + EPS) +
                               tf.log(1 - predict_sample_fake + EPS)))

    with tf.name_scope('generator_loss'):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        if a.mmad_loss:
            with tf.name_scope('MMAD'):
                dif = tf.abs(targets - outputs, name='absdist')
                temp = tf.reduce_mean(dif)
                foreground_L1 = tf.reduce_mean(
                    tf.boolean_mask(dif, target_masks), name='foreground')
                neg_target_masks = tf.logical_not(target_masks, name='neg')
                background_L1 = tf.reduce_mean(
                    tf.boolean_mask(dif, neg_target_masks), name='background')
                gen_loss_L1 = 2*foreground_L1/3 + background_L1/3
                gen_loss_L1 = tf.where(tf.is_nan(gen_loss_L1),
                                       temp,
                                       gen_loss_L1)
        else:
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight
        if a.sample_lambda > 0.0:
            gen_loss = gen_loss + a.sample_lambda * \
                tf.reduce_mean(-tf.log(predict_sample_fake + EPS))

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    if a.mode in {'train'}:
        with tf.name_scope('discriminator_train'):
            discrim_tvars = [var for var in tf.trainable_variables()
                             if var.name.startswith('discriminator')]
            discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(
                discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(
                discrim_grads_and_vars)

        with tf.name_scope('generator_train'):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables()
                             if var.name.startswith('generator')]
                gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(
                    gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

        return Model(
            predict_real=predict_real,
            predict_fake=predict_fake,
            discrim_loss=ema.average(discrim_loss),
            discrim_grads_and_vars=discrim_grads_and_vars,
            gen_loss_GAN=ema.average(gen_loss_GAN),
            gen_loss_L1=ema.average(gen_loss_L1),
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=outputs,
            iters=tf.concat(iters, axis=2, name='between_steps'),
            train=tf.group(update_losses, incr_global_step, gen_train),
            rest=rest
        )

    else:
        return Model(
            predict_real=predict_real,
            predict_fake=predict_fake,
            discrim_loss=discrim_loss,
            discrim_grads_and_vars=tf.constant(0),
            gen_loss_GAN=gen_loss_GAN,
            gen_loss_L1=gen_loss_L1,
            gen_grads_and_vars=tf.constant(0),
            outputs=outputs,
            iters=tf.concat(iters, axis=2, name='between_steps'),
            train=tf.constant(0),
            rest=rest
        )


# ==== Other functions
def save_images(fetches, step=None, epoch=None):
    if epoch is None:
        image_dir = os.path.join(a.output_dir, 'images')
    else:
        image_dir = os.path.join(a.output_dir, f'epoch/e{epoch}')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches['paths']):
        name, _ = os.path.splitext(os.path.basename(in_path.decode('utf8')))
        fileset = {'name': name, 'step': step}
        for kind in KINDS:
            filename = name + '-' + kind + '.png'
            if step is not None:
                filename = f'{step:08d}-{filename!s}'
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, 'wb') as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    index_path = os.path.join(a.output_dir, 'index.html')
    if os.path.exists(index_path):
        index = open(index_path, 'a')
    else:
        index = open(index_path, 'w')
        index.write('<html><body><table><tr>')
        if step:
            index.write('<th>step</th>')
        index.write(''.join(f'<th>{i}</th>' for i in ['name'] + KINDS) +
                    '</tr>')

    for fileset in filesets:
        index.write('<tr>')

        if step:
            index.write(f"<td>{fileset['step']}</td>")
        index.write(f"<td>{fileset['name']}</td>")

        for kind in KINDS:
            index.write(f"<td><img src='images/{fileset[kind]}'></td>")

        index.write('</tr>\n')
    return index_path


def create_display_and_summary_ops(examples, model):
    """Reverse any processing on images so they can be written to disk or
    displayed to user, and create tf summaries"""
    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)
    iters = deprocess(model.iters)

    def convert(image, name='convert'):
        with tf.name_scope(name):
            return tf.image.convert_image_dtype(image, dtype=tf.uint8,
                                                saturate=True)
    converted_inputs = convert(inputs, 'convert_inputs')
    converted_targets = convert(targets, 'convert_targets')
    converted_outputs = convert(outputs, 'convert_outputs')
    converted_diff = convert(tf.abs(outputs-targets), 'convert_diff')
    converted_between = convert(iters, 'convert_between')

    with tf.name_scope('encode_images'):
        display_fetches = {
            'paths': examples.paths,
            'inputs': tf.map_fn(tf.image.encode_png, converted_inputs,
                                dtype=tf.string, name='input_pngs'),
            'targets': tf.map_fn(tf.image.encode_png, converted_targets,
                                 dtype=tf.string, name='target_pngs'),
            'outputs': tf.map_fn(tf.image.encode_png, converted_outputs,
                                 dtype=tf.string, name='output_pngs'),
            'difference': tf.map_fn(tf.image.encode_png, converted_diff,
                                    dtype=tf.string, name='diff_pngs'),
            'between': tf.map_fn(tf.image.encode_png, converted_between,
                                 dtype=tf.string, name='btwn_pngs'),
            'score': model.gen_loss_L1,
        }
    if a.mode in {'test', 'validate'}:
        qu = tf.FIFOQueue(32, [tf.string]*6, shapes=[(a.batch_size,)]*6)
        fetches_enq = qu.enqueue([
            display_fetches['paths'],
            display_fetches['inputs'],
            display_fetches['targets'],
            display_fetches['outputs'],
            display_fetches['difference'],
            display_fetches['between'],
        ])
        fetches_deq = qu.dequeue()

        def save_and_dequeue():
            try:
                while True:
                    fetch = sess.run(fetches_deq)
                    d_fetch = {k: v for v, k in zip(fetch, 'paths inputs targe'
                                                    'ts outputs difference bet'
                                                    'ween'.split())}
                    fsets = save_images(d_fetch, epoch=epoch)
                    append_index(fsets)
            except tf.errors.CancelledError:
                pass

    else:
        fetches_enq = save_and_dequeue = None

    # The summaries
    if a.sample_lambda > 0.0:
        with tf.name_scope('sample_summary'):
            tf.summary.image('real_sample',
                             model.rest['sample']['predict_real'])
            tf.summary.image('fake_sample',
                             model.rest['sample']['predict_fake'])
            tf.summary.image('real_inp',
                             model.rest['sample']['real_inp'])
            tf.summary.image('fake_inp',
                             model.rest['sample']['fake_inp'])
    with tf.name_scope('inputs_summary'):
        tf.summary.image('inputs', converted_inputs)

    with tf.name_scope('targets_summary'):
        tf.summary.image('targets', converted_targets)

    with tf.name_scope('outputs_summary'):
        tf.summary.image('outputs', converted_outputs)

    with tf.name_scope('predict_real_summary'):
        tf.summary.image('predict_real',
                         tf.image.convert_image_dtype(model.predict_real,
                                                      dtype=tf.uint8))

    with tf.name_scope('predict_fake_summary'):
        tf.summary.image('predict_fake',
                         tf.image.convert_image_dtype(model.predict_fake,
                                                      dtype=tf.uint8))

    tf.summary.scalar('discriminator_loss', model.discrim_loss)
    tf.summary.scalar('generator_loss_GAN', model.gen_loss_GAN)
    tf.summary.scalar('generator_loss_L1', model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + '/values', var)

    if a.mode == 'train':
        for grad, var in model.discrim_grads_and_vars+model.gen_grads_and_vars:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    return display_fetches, fetches_enq, save_and_dequeue


def test(display_fetches, fetches_enq, max_steps, epoch_size, lepoch=None):
    global epoch
    epoch = lepoch
    tmp = a.output_dir
    max_steps = min(max_steps, epoch_size)
    t0 = time.time()
    for dir in ['seen', 'unseen']:
        scores = []
        print(dir)
        a.output_dir = os.path.join(tmp, dir)
        for step in range(max_steps):
            _, results = sess.run((fetches_enq, display_fetches['score']))
            scores.append(results)
            if step % 10 == 0:
                print(f'at {step}'+' '*10, end='\r')
        with open(a.output_dir+'/score.npy', 'wb') as f:
            np.save(f, np.array(scores))
        print('wrote index at', a.output_dir)


def main():
    global OBJECT_QUEUE, sess
    # --- Initialise main function
    if tf.__version__.split('.')[0] != '1':
        raise Exception('Tensorflow version 1 required')

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(os.path.join(a.output_dir, 'epoch')):
        os.makedirs(os.path.join(a.output_dir, 'epoch'))

    # load some options from the checkpoint
    if a.mode in {'test', 'validate'}:
        if 'test' not in a.datatype:
            a.datatype = 'basic_test'
        if a.checkpoint is None:
            raise Exception('checkpoint required for test mode')
        options = {'ngf', 'ndf', 'sample_lambda', 'use_bias'}
        if os.path.exists(os.path.join(a.checkpoint, 'options.json')):
            with open(os.path.join(a.checkpoint, 'options.json')) as f:
                for key, val in json.loads(f.read()).items():
                    if key in options:
                        print('loaded', key, '=', val)
                        setattr(a, key, val)

    for k, v in a._get_kwargs():
        print(k, '=', v)

    with open(os.path.join(a.output_dir, 'options.json'), 'w') as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    # ---- Build model
    # inputs and targets are [batch_size, height, width, channels]
    OBJECT_QUEUE, epoch_size = get_generator(a.datatype)
    examples, model, load_and_enqueue = init_model(epoch_size)

    print(f'examples count = {examples.count}')

    with tf.name_scope('deprocess'):
        display_fetches, fetch_enq, save_and_dequeue = \
            create_display_and_summary_ops(examples, model)

    with tf.name_scope('parameter_count'):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v))
                                         for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)
    epoch_saver = tf.train.Saver(max_to_keep=None)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    # don't horde the full gpu
    config = tf.ConfigProto()
    if 0.0 < a.gpu_frac < 1.0:
        config.gpu_options.per_process_gpu_memory_fraction = a.gpu_frac
    elif a.gpu_frac < -0.5:
        config.gpu_options.allow_growth = True

    max_steps = 2**32
    if a.max_epochs is not None:
        max_steps = examples.steps_per_epoch * a.max_epochs
    if a.max_steps is not None:
        max_steps = a.max_steps

    # ---- start process
    with sv.managed_session(config=config) as sess:
        if a.mode is not 'validate'and a.checkpoint is not None:
            print('loading model from checkpoint')
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            print(checkpoint)
            if checkpoint is None:
                saver.restore(sess, a.checkpoint)
            else:
                saver.restore(sess, checkpoint)

        if a.mode in {'train'}:
            for _ in range(sess.run(sv.global_step)):
                next(OBJECT_QUEUE)
        t = threading.Thread(target=load_and_enqueue)
        t.start()
        if a.mode in {'test', 'validate'}:
            t2 = threading.Thread(target=save_and_dequeue)
            t2.start()

        print('parameter_count =', sess.run(parameter_count))

        if a.mode == 'validate':
            # validating, test for each checkpoint in the epoch folder
            checkpoints = [os.path.splitext(f)[0] for f in glob.glob(
                os.path.join(a.checkpoint, 'epoch/model*.index'))]
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            scores = np.zeros((3, len(checkpoints)))
            for checkpoint in checkpoints:
                print(f'loading next checkpoint: {checkpoint}')
                i = checkpoint.split('-')[-1]
                saver.restore(sess, checkpoint)
                test(display_fetches, fetch_enq, max_steps, epoch_size,
                     epoch=i)
            return
        elif a.mode == 'test':
            # testing
            test(display_fetches, fetch_enq, max_steps, epoch_size)
        else:
            # training
            start = time.time()
            start_step = int(sess.run(sv.global_step))
            for step in range(start_step, max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or
                                         step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    'train': model.train,
                    'global_step': sv.global_step,
                }

                if should(a.progress_freq):
                    fetches['discrim_loss'] = model.discrim_loss
                    fetches['gen_loss_GAN'] = model.gen_loss_GAN
                    fetches['gen_loss_L1'] = model.gen_loss_L1
                    fetches['paths'] = examples.paths

                if should(a.summary_freq):
                    fetches['summary'] = sv.summary_op

                if should(a.display_freq):
                    fetches['display'] = display_fetches

                results = sess.run(fetches, options=options,
                                   run_metadata=run_metadata)

                if should(a.summary_freq):
                    print('recording summary')
                    sv.summary_writer.add_summary(results['summary'],
                                                  results['global_step'])

                if should(a.display_freq):
                    print('saving display images')
                    filesets = save_images(results['display'],
                                           step=results['global_step'])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print('recording trace')
                    sv.summary_writer.add_run_metadata(
                        run_metadata, 'step_%d' % results['global_step'])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume
                    # from a checkpoint.
                    train_epoch = math.ceil(results['global_step'] /
                                            examples.steps_per_epoch)
                    train_step = (results['global_step'] - 1) % \
                        examples.steps_per_epoch + 1
                    rate = (step + 1 - start_step) * a.batch_size / \
                        (time.time() - start)
                    remaining = (max_steps - step + start_step) * a.batch_size\
                        / rate
                    print(f'progress  epoch {train_epoch}  step {train_step}  '
                          f'image/sec {rate:0.1f}  remaining {t_s(remaining)}')

                    print('discrim_loss', results['discrim_loss'])
                    print('gen_loss_GAN', results['gen_loss_GAN'])
                    print('gen_loss_L1', results['gen_loss_L1'])
                    print('paths', results['paths'])

                if should(a.save_freq):
                    print('saving model')
                    saver.save(sess, os.path.join(a.output_dir, 'model'),
                               global_step=sv.global_step)

                if step == 0 or \
                        (results['global_step'] / examples.steps_per_epoch) % \
                        a.epoch_freq == 0:
                    print('epoch save')
                    tmp = results['global_step'] // examples.steps_per_epoch
                    epoch_saver.save(sess, os.path.join(a.output_dir,
                                                        'epoch/model'),
                                     global_step=tmp)

                if sv.should_stop():
                    break


def t_s(sec):
    min = sec // 60
    h = min // 60
    d = h // 24
    return (f'{d:02.0f}d:' if d > 0 else '') + \
        (f'{h % 24:02.0f}u:' if h > 0 else '') + \
        (f'{min % 60:02.0f}m:' if min > 0 else '') + f'{sec % 60:02.0f}s'


if __name__ == '__main__':
    a = _build_parser().parse_args()
    main()
