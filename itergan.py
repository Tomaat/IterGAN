#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main method for implicit 3d representation learning based on pix2pix*

    To use this program, run ?> python itergan.py --help

    used for implicit 3d representation learning based on pix2pix

    *Pix2pix: Isola, P., Zhu, J.-Y., Zhou, T. and Efros, A. A. (2016),
              'Image-to-image translation with conditional adversarial
              networks', arxiv

    code adapted from https://github.com/affinelayer/pix2pix-tensorflow

    by: Ysbrand Galama
    latest version: 2018-03-19
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
from itertools import cycle

import baselines
import default

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
    parser.add_argument('--summary_freq', type=int, default=0,
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

    parser.add_argument('--gpu_frac', type=float, default=0,
                        help='how much of the gpu can be used')
    parser.add_argument('--init', default=default.INITIALIZER,
                        choices=['normal', 'small', 'xavier'],
                        help='change initialisation')
    parser.add_argument('--mmad_loss', type=str2bool, default=False,
                        help='whether to use MMAD instead of L1')

    parser.add_argument('--sample_lambda', type=float, default=0,
                        help='if value > 0, adds a discriminator to each step '
                        'that uses single images')
    parser.add_argument('--between_lambda', type=float, default=0,
                        help='if value > 0, adds a discriminator to each step '
                        'that uses input/output pairs of images')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of images in batch')

    parser.add_argument('--ngf', type=int, default=default.NGF,
                        help='number of generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=default.NDF,
                        help='number of discriminator filters in first conv la'
                        'yer')
    parser.add_argument('--use_bias', type=str2bool, default=default.USE_BIAS,
                        help='whether to use bias in conv layer')

    parser.add_argument('--lr', type=float, default=default.LR,
                        help='initial learning rate for adam')
    parser.add_argument('--beta1', type=float, default=default.BETA1,
                        help='momentum term of adam')
    parser.add_argument('--l1_weight', type=float, default=default.LAMBDA_L1,
                        help='weight on L1 term for generator gradient')
    parser.add_argument('--gan_weight', type=float, default=default.LAMBDA_GAN,
                        help='weight on GAN term for generator gradient')

    parser.add_argument('--baseline', default=None,
                        choices=['pix2pix', 'identity', 'projective'])
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
else:
    args = None
epoch_ = None

EPS = default.EPS
OBJECT_QUEUE = None
KINDS = ['inputs', 'outputs', 'targets', 'difference', 'between']
# KINDS = ['difference', 'between']
# KINDS = []

Examples = collections.namedtuple('Examples', 'paths, inputs, targets, count,'
                                  ' steps_per_epoch, iters, target_masks,'
                                  ' between_target')
Model = collections.namedtuple('Model', 'outputs, predict_real, iters,'
                               ' predict_fake, discrim_loss,'
                               ' discrim_grads_and_vars, gen_loss_GAN,'
                               ' gen_loss_L1, gen_grads_and_vars, train, rest')

# ==== Global vars and Generators for inputs
QSIZE = 32
IMG_SHAPE = [256, 256, 3]
DATA_FILE = '{dir}/img/{id}/{id}_r{rot}.png'
MASK_FILE = '{dir}/mask/{id}/{id}_r{rot}.png'


def get_generator(name, max_epochs):
    """Get an iterator to construct the necessary input files"""
    vkitti_max = {1: 447, 2: 233, 6: 270, 18: 339, 20: 837}

    def basic_train(r=6):
        yield from ((id, x, r) for x in range(0, 72, 2)
                    for id in range(1, 801))

    def basic_test(r=6):
        yield from ((id, x, r) for id in range(700, 800)
                    for x in range(1, 72, 2))
        yield from ((id, x, r) for id in range(801, 901)
                    for x in range(1, 72, 2))

    def vkitti_train():
        for scene in [1, 6, 18, 20]:
            for id in range(vkitti_max[scene]):
                for x in [0, 3, 6]:
                    if scene == 20 and id % 5 == 0 and x == 0:
                        continue
                    yield (scene*1000+id, x, 6)

    def vkitti_test_seen():
        for id in range(0, vkitti_max[20], 5):
            yield (20000+id, 0, 6)

    def step_wise_train():
        t = max_epochs // 4
        subs = [(1,)]*t + [(1, 2)]*t + [(1, 2, 3, 4, 5, 6)]*t + \
            [range(1, 37)]*(t+1)
        for rng in subs:
            yield from ((id, x, i) for x in range(1, 72, 2*len(rng))
                        for i in rng
                        for id in range(1, 801))

    def varied_rot_train():
        yield from ((id, x, i) for x in range(1, 72, 8)
                    for i in range(1, 37, 3)
                    for id in range(1, 801, 3))

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
        'basic_train_5deg': (cycle(basic_train(1)), 28800),
        'basic_test': (cycle(basic_test()), 3600),
        'rotate_test': (cycle(basic_test(18)), 3600),
        'step_wise_train': (step_wise_train(), 28800),
        'varied_rot_train': (cycle(varied_rot_train()), 28836),
        'very_small': (cycle(very_small()), 8),
        'very_small_test': (cycle(very_small_test()), 8),
        'vkitti_train': (cycle(vkitti_train()), 5511),
        'vkitti_test_seen': (cycle(vkitti_test_seen()), 168),
    }[name]


# ==== Default ops
def preprocess(raw, mask=False, *, name='preprocess'):
    """From raw data, create a tensor of correct shape and dtype."""
    with tf.name_scope(name):
        raw = tf.image.convert_image_dtype(raw, dtype=tf.float32)
        raw.set_shape(IMG_SHAPE)
        raw = tf.cast(raw, tf.bool) if mask else raw * 2 - 1
        return raw


def deprocess(image, *, name='deprocess'):
    """Change the range of an image from [-1, 1] to [0, 1]"""
    with tf.name_scope(name):
        return (image + 1) / 2


def conv(batch_input, out_channels, stride, *, init='normal', use_bias=False,
         name='conv'):
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
    name (kwarg) - string
        The variable-scope of these ops (default 'conv')

    Returns
    -------
    np.tensor - The tensor after convolution
    """
    with tf.variable_scope(name):
        if init == 'xavier':
            initial = CONV_INIT_XAVIER
        elif init == 'small':
            initial = tf.random_normal_initializer(default.CONV_INIT_MU,
                                                   default.CONV_INIT_STD_SMALL)
        else:
            initial = tf.random_normal_initializer(default.CONV_INIT_MU,
                                                   default.CONV_INIT_STD)
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable('filter', [default.CONV_KW, default.CONV_KH,
                                            in_channels, out_channels],
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
        if use_bias:
            biases = tf.get_variable('biases', [out_channels],
                                     initializer=default.CONV_BIAS_INIT)
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def lrelu(x, a, *, name='lrelu'):
    """Implementation of Leaky ReLU

    Arguments
    ---------
    x - tf.tensor
        The input tensor
    a - float
        The alpha parameter of the Leaky ReLU
    name (kwarg) - string
        The variable-scope of these ops (default 'lrelu')

    Returns
    -------
    tf.tensor - The tensor after the relu
    """
    with tf.name_scope(name):
        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input, *, name='batchnorm'):
    """Performs instance normalisation using the tf.nn.batch_normalization

    Arguments
    ---------
    input - tf.tensor
        The input tensor
    name (kwarg)- string
        The variable-scope of these ops (default 'batchnorm')

    Returns
    -------
    tf.tensor - the normalised output
    """
    with tf.variable_scope(name):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable('offset', [channels], dtype=tf.float32,
                                 initializer=tf.zeros_initializer())
        scale = tf.get_variable('scale', [channels], dtype=tf.float32,
                                initializer=default.BATCHNORM_INITIALIZER)
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_eps = default.BATCHNORM_VARIANCE_EPS
        normalized = tf.nn.batch_normalization(input, mean, variance, offset,
                                               scale,
                                               variance_epsilon=variance_eps)
        return normalized


def deconv(batch_input, out_channels, *, init='normal', use_bias=False,
           name='deconv'):
    """Create transposed_convolution aka deconvolution layer

    Arguments
    ---------
    batch_input - tf.tensor
        The input of the layer
    out_channels - int
        The amount of channels of the output layer
    name (kwarg)- string
        The variable-scope of these ops (default 'deconv')

    Returns
    -------
    tf.tensor - the output of the layer
    """
    with tf.variable_scope(name):
        if init == 'xavier':
            initial = CONV_INIT_XAVIER
        elif init == 'small':
            initial = tf.random_normal_initializer(default.CONV_INIT_MU,
                                                   default.CONV_INIT_STD_SMALL)
        else:
            initial = tf.random_normal_initializer(default.CONV_INIT_MU,
                                                   default.CONV_INIT_STD)

        batch, in_height, in_width, in_channels = [int(d) for d in
                                                   batch_input.get_shape()]
        filter = tf.get_variable('filter', [default.CONV_KW, default.CONV_KH,
                                            out_channels, in_channels],
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
        if use_bias:
            biases = tf.get_variable('biases', [out_channels],
                                     initializer=default.CONV_BIAS_INIT)
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


# ==== Model functions
def init_model(epoch_size, *, opt=args):
    """Create data queue for training.

    Returns
    -------
    Examples named tuple - object containing the necessarily information of the
        queue
    """
    if opt.input_dir is None or not os.path.exists(opt.input_dir):
        raise Exception('input_dir does not exist')

    with tf.name_scope('queue_fill'):
        # The image placeholders
        with tf.name_scope('file_inputs'):
            paths = tf.placeholder(tf.string, shape=())
            input_data = tf.placeholder(tf.string, shape=())
            target_data = tf.placeholder(tf.string, shape=())
            if opt.between_lambda > 0:
                between_data = tf.placeholder(tf.string, shape=())
                between_iter = tf.placeholder(tf.int32, shape=())
            mask_data = tf.placeholder(tf.string, shape=())
            iters = tf.placeholder(tf.int32, shape=())

        input_image = preprocess(tf.image.decode_png(input_data,
                                                     channels=IMG_SHAPE[2]))
        target_image = preprocess(tf.image.decode_png(target_data,
                                                      channels=IMG_SHAPE[2]))
        input_mask = preprocess(tf.image.decode_png(mask_data,
                                                    channels=IMG_SHAPE[2]),
                                True)
        if opt.between_lambda > 0:
            between_image = preprocess(tf.image.decode_png(
                between_data, channels=IMG_SHAPE[2]))

            # The tf queue that handles the reading of images (with between)
            q = tf.FIFOQueue(QSIZE, [tf.string, tf.float32, tf.float32,
                                     tf.bool, tf.int32, tf.int32, tf.float32],
                             shapes=[[], IMG_SHAPE, IMG_SHAPE, IMG_SHAPE, [],
                                     [], IMG_SHAPE])
            enqueue_op = q.enqueue([paths, input_image, target_image,
                                    input_mask, iters, between_iter,
                                    between_image])

        else:
            # The tf queue that handles the reading of images
            q = tf.FIFOQueue(QSIZE, [tf.string, tf.float32, tf.float32,
                                     tf.bool, tf.int32],
                             shapes=[[], IMG_SHAPE, IMG_SHAPE, IMG_SHAPE, []])
            enqueue_op = q.enqueue([paths, input_image, target_image,
                                    input_mask, iters])
    if opt.between_lambda > 0:
        paths_batch, inputs_batch, targets_batch, masks_batch, iters_batch, \
            *between_target = q.dequeue_many(opt.batch_size)
    else:
        paths_batch, inputs_batch, targets_batch, masks_batch, iters_batch = \
            q.dequeue_many(opt.batch_size)
        between_target = None

    steps_per_epoch = int(math.ceil(epoch_size / opt.batch_size))
    examples = Examples(paths=paths_batch, inputs=inputs_batch,
                        targets=targets_batch, iters=iters_batch,
                        target_masks=masks_batch, count=epoch_size,
                        steps_per_epoch=steps_per_epoch,
                        between_target=between_target)

    if opt.baseline == 'pix2pix':
        model = baselines.create_pix2pix_model(examples.inputs,
                                               examples.targets, opt=opt)
    elif opt.baseline == 'projective':
        model = baselines.create_projective_model(examples.inputs,
                                                  examples.targets)
    elif opt.baseline == 'identity':
        model = baselines.create_identity_model(examples.inputs,
                                                examples.targets)
    else:
        model = create_model(examples.inputs, examples.targets,
                             examples.target_masks, examples.iters,
                             examples.between_target)

    def load_and_enqueue():
        while True:
            try:
                id, x, i = next(OBJECT_QUEUE)
                rot = x*5
                irot = (rot + i*5) % 360
                input_file = DATA_FILE.format(dir=opt.input_dir, id=id,
                                              rot=rot)
                tgt_file = DATA_FILE.format(dir=opt.input_dir, id=id, rot=irot)
                mask_file = MASK_FILE.format(dir=opt.input_dir, id=id,
                                             rot=irot)
                if not os.path.exists(mask_file):
                    mask_file = os.path.join(opt.input_dir, 'MASK.png')
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
                break

    def load_and_enqueue_between():
        while True:
            try:
                id, x, i = next(OBJECT_QUEUE)
                j = np.random.randint(1, i, dtype=np.int32)
                rot = x*5
                irot = (rot + i*5) % 360
                brot = (rot + j*5) % 360
                input_file = DATA_FILE.format(dir=opt.input_dir, id=id,
                                              rot=rot)
                tgt_file = DATA_FILE.format(dir=opt.input_dir, id=id, rot=irot)
                mask_file = MASK_FILE.format(dir=opt.input_dir, id=id,
                                             rot=irot)
                btwn_file = DATA_FILE.format(dir=opt.input_dir, id=id,
                                             rot=brot)
                path = f'{id}_r{rot}-r{irot}.png'
                with open(input_file, 'rb') as fi, open(tgt_file, 'rb') as ft,\
                        open(mask_file, 'rb') as fm, \
                        open(btwn_file, 'rb') as fb:
                    sess.run(enqueue_op, feed_dict={paths: path,
                                                    input_data: fi.read(),
                                                    target_data: ft.read(),
                                                    mask_data: fm.read(),
                                                    between_data: fb.read(),
                                                    between_iter: j,
                                                    iters: i})
            except tf.errors.CancelledError:
                print('WARNING load_queue stopped')
                break

    if opt.between_lambda > 0:
        load_and_enqueue = load_and_enqueue_between
    return examples, model, load_and_enqueue


def create_generator(generator_inputs, generator_outputs_channels, *,
                     opt=args):
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
        output = conv(generator_inputs, opt.ngf, stride=2, init=opt.init,
                      use_bias=opt.use_bias)
        layers.append(output)

    layer_specs = [
        (opt.ngf * 2, True),  # e_2: [b, 128, 128, ngf] => [b, 64, 64, ngf * 2]
        (opt.ngf * 4, True),  # e_3: [b, 64, 64, ngf * 2] => [b, 32, 32, ngf * 4]
        (opt.ngf * 8, True),  # e_4: [b, 32, 32, ngf * 4] => [b, 16, 16, ngf * 8]
        (opt.ngf * 8, True),  # e_5: [b, 16, 16, ngf * 8] => [b, 8, 8, ngf * 8]
        (opt.ngf * 8, True),  # e_6: [b, 8, 8, ngf * 8] => [b, 4, 4, ngf * 8]
        (opt.ngf * 8, True),  # e_7: [b, 4, 4, ngf * 8] => [b, 2, 2, ngf * 8]
        (opt.ngf * 8, False)  # e_8: [b, 2, 2, ngf * 8] => [b, 1, 1, ngf * 8]
    ]

    for out_channels, do_bn in layer_specs:
        with tf.variable_scope('encoder_%d' % (len(layers) + 1)):
            rectified = lrelu(layers[-1], default.GEN_LRELU)
            # [batch, in_height, in_width, in_channels] =>
            #  [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2,
                             init=opt.init, use_bias=opt.use_bias)
            if do_bn:
                output = batchnorm(convolved)
            else:
                output = tf.identity(convolved)
            layers.append(output)

    layer_specs = [
        (opt.ngf * 8, default.GEN_DROPOUT),  # d_8: [b, 1, 1, ngf*8] => [b, 2, 2, ngf*8*2]
        (opt.ngf * 8, default.GEN_DROPOUT),  # d_7: [b, 2, 2, ngf*8*2] => [b, 4, 4, ngf*8*2]
        (opt.ngf * 8, default.GEN_DROPOUT),  # d_6: [b, 4, 4, ngf*8*2] => [b, 8, 8, ngf*8*2]
        (opt.ngf * 8, 0.0),  # d_5: [b, 8, 8, ngf*8*2] => [b, 16, 16, ngf*8*2]
        (opt.ngf * 4, 0.0),  # d_4: [b, 16, 16, ngf*8*2] => [b, 32, 32, ngf*4*2]
        (opt.ngf * 2, 0.0),  # d_3: [b, 32, 32, ngf*4*2] => [b, 64, 64, ngf*2*2]
        (opt.ngf, 0.0),      # d_2: [b, 64, 64, ngf*2*2] => [b, 128, 128, ngf*2]
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
            output = deconv(rectified, out_channels, init=opt.init,
                            use_bias=opt.use_bias)
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


def create_discriminator(discrim_inputs, discrim_targets=None, *, opt=args,
                         n_out=1):
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
        convolved = conv(input, opt.ndf, stride=2, init=opt.init,
                         use_bias=opt.use_bias)
        rectified = lrelu(convolved, default.DIS_LRELU)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope('layer_%d' % (len(layers) + 1)):
            out_channels = opt.ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last has stride 1
            convolved = conv(layers[-1], out_channels, stride=stride,
                             init=opt.init, use_bias=opt.use_bias)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, default.DIS_LRELU)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope('layer_%d' % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=n_out, stride=1,
                         init=opt.init, use_bias=opt.use_bias)
        if n_out == 1:
            output = tf.sigmoid(convolved)
        else:
            output = tf.nn.softmax(convolved)
        layers.append(output)

    return layers[-1]


def IterGAN(inputs, out_channels, iter_num, *, opt=args, name='IterGAN'):
    if opt.batch_size != 1:
        print('WARNING, current code only works correctly with batch=1. (or if'
              ' the variable-iters are synchronised with the batch-size')

    with tf.name_scope(name):
        max_iter = tf.reduce_max(iter_num)
        with tf.variable_scope('generator'):
            def body(i, inp, iters):
                out = create_generator(inp, out_channels, opt=opt)
                next_iters = tf.concat([iters, out], axis=2,
                                       name='between_steps')
                return i+1, out, next_iters

            def condition(i, inp, iters):
                return i < max_iter

            i0 = tf.constant(0)
            it_shape = tf.TensorShape([opt.batch_size, IMG_SHAPE[0], None,
                                       IMG_SHAPE[2]])
            img_shape = inputs.get_shape()
            _, outputs, iters = tf.while_loop(
                condition, body, (i0, inputs, inputs),
                shape_invariants=(i0.get_shape(), img_shape, it_shape))
    return outputs, iters, img_shape, max_iter


def create_model(inputs, targets, target_masks, iter_num, *,
                 opt=args, between_target=None):
    """Create the full model for training/testing
    """
    out_channels = int(targets.get_shape()[-1])
    assert out_channels == int(inputs.get_shape()[-1])

    outputs, iters, img_shape, max_iter = IterGAN(inputs, out_channels,
                                                  iter_num, opt=opt)

    rest = {}

    if opt.sample_lambda > 0.0:
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
                predict_sample_real = create_discriminator(sample_real,
                                                           opt=opt)
            with tf.variable_scope('disciminator_sample', reuse=True):
                predict_sample_fake = create_discriminator(sample_fake,
                                                           opt=opt)
            rest['sample'] = {'i': i,
                              'j': j,
                              'predict_real': predict_sample_real,
                              'predict_fake': predict_sample_fake,
                              'real_inp': sample_real,
                              'fake_inp': sample_fake
                              }
            rest['Dis2'] = tf.reduce_mean(-tf.log(predict_sample_fake + EPS))

    if opt.between_lambda > 0.0:
        assert between_target is not None
        with tf.name_scope('between_steps'):
            i, sample_tgt = between_target
            i = tf.reduce_max(i)
            d = IMG_SHAPE[1]
            sample_fake = iters[:, :, (i+1)*d:(i+2)*d, :]
            sample_fake.set_shape(img_shape)
            with tf.variable_scope('disciminator_between'):
                predict_between_real = create_discriminator(inputs, sample_tgt,
                                                            opt=opt)
            with tf.variable_scope('disciminator_between', reuse=True):
                predict_between_fake = create_discriminator(inputs,
                                                            sample_fake,
                                                            opt=opt)
            rest['between'] = {'i': i,
                               'predict_real': predict_between_real,
                               'predict_fake': predict_between_fake,
                               'between_targets': tf.image.convert_image_dtype(
                                    deprocess(sample_tgt), dtype=tf.uint8,
                                    saturate=True)
                               }
            rest['Dis2'] = tf.reduce_mean(-tf.log(predict_between_fake + EPS))

    # create two copies of discriminator, one for real and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator'):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets, opt=opt)

    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs, opt=opt)

    with tf.name_scope('discriminator_loss'):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) +
                                      tf.log(1 - predict_fake + EPS)))
        if opt.sample_lambda > 0.0:
            discim_loss = discrim_loss + opt.sample_lambda * \
                tf.reduce_mean(-(tf.log(predict_sample_real + EPS) +
                               tf.log(1 - predict_sample_fake + EPS)))
        if opt.between_lambda > 0.0:
            discim_loss = discrim_loss + opt.between_lambda * \
                tf.reduce_mean(-(tf.log(predict_between_real + EPS) +
                               tf.log(1 - predict_between_fake + EPS)))

    with tf.name_scope('generator_loss'):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        if opt.mmad_loss:
            with tf.name_scope('MMAD'):
                dif = tf.abs(targets - outputs, name='absdist')
                temp = tf.reduce_mean(dif)
                foreground_L1 = tf.reduce_mean(
                    tf.boolean_mask(dif, target_masks), name='foreground')
                neg_target_masks = tf.logical_not(target_masks, name='neg')
                background_L1 = tf.reduce_mean(
                    tf.boolean_mask(dif, neg_target_masks), name='background')
                gen_loss_L1 = default.WMAD_FOREGROUND_FRAQ*foreground_L1 + \
                    (1-default.WMAD_FOREGROUND_FRAQ)*background_L1
                gen_loss_L1 = tf.where(tf.is_nan(gen_loss_L1),
                                       temp,
                                       gen_loss_L1)
        else:
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * opt.gan_weight + gen_loss_L1 * opt.l1_weight
        if opt.sample_lambda > 0.0:
            gen_loss = gen_loss + opt.sample_lambda * \
                tf.reduce_mean(-tf.log(predict_sample_fake + EPS))
        if opt.between_lambda > 0.0:
            gen_loss = gen_loss + opt.between_lambda * \
                tf.reduce_mean(-tf.log(predict_between_fake + EPS))

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    if opt.mode in {'train'}:
        with tf.name_scope('discriminator_train'):
            discrim_tvars = [var for var in tf.trainable_variables()
                             if var.name.startswith('discriminator')]
            discrim_optim = tf.train.AdamOptimizer(opt.lr, opt.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(
                discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(
                discrim_grads_and_vars)

        with tf.name_scope('generator_train'):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables()
                             if var.name.startswith('generator')]
                gen_optim = tf.train.AdamOptimizer(opt.lr, opt.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(
                    gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=default.EMA_DECAY)
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
def save_images(fetches, *, opt=args, step=None, epoch=None):
    if epoch is None:
        image_dir = os.path.join(opt.output_dir, 'images')
    else:
        image_dir = os.path.join(opt.output_dir, f'epoch/e{epoch}')

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


def append_index(filesets, *, opt=args, step=False):
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    index_path = os.path.join(opt.output_dir, 'index.html')
    if os.path.exists(index_path):
        index = open(index_path, 'a')
    else:
        index = open(index_path, 'w')
        index.write('<html><body><table><tr>')
        if step:
            index.write('<th>step</th>')
        if epoch_ is not None:
            index.write('<th>epoch</th>')
        index.write(''.join(f'<th>{i}</th>' for i in ['name'] + KINDS) +
                    '</tr>')

    image_dir = 'images' if epoch_ is None else f'epoch/e{epoch_}'
    for fileset in filesets:
        index.write('<tr>')

        if step:
            index.write(f"<td>{fileset['step']}</td>")
        if epoch_ is not None:
            index.write(f"<td>{epoch_}</td>")
        index.write(f"<td>{fileset['name']}</td>")

        for kind in KINDS:
            index.write(f"<td><img src='{image_dir}/{fileset[kind]}'></td>")

        index.write('</tr>\n')
    return index_path


def create_display_and_summary_ops(examples, model, *, opt=args):
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
            'Dis': model.gen_loss_GAN,
            'Dis2': model.rest['Dis2'] if 'Dis2' in model.rest
            else tf.constant(np.nan),
        }
    if opt.mode in {'test', 'validate'}:
        qu = tf.FIFOQueue(32, [tf.string]*6, shapes=[(opt.batch_size,)]*6)
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
            while True:
                try:
                    fetch = sess.run(fetches_deq)
                    d_fetch = {k: v for v, k in zip(fetch, 'paths inputs targe'
                                                    'ts outputs difference bet'
                                                    'ween'.split())}
                    fsets = save_images(d_fetch, epoch=epoch_)
                    append_index(fsets)
                except tf.errors.CancelledError:
                    print('WARNING save_queue stopped')
                    break
                except RuntimeError as re:
                    if ('Attempted to use a closed Session.' in re.args):
                        print('WARNING save_queue stopped')
                        break

    else:
        fetches_enq = save_and_dequeue = None

    # The summaries
    if opt.sample_lambda > 0.0:
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

    if opt.mode == 'train':
        for grad, var in model.discrim_grads_and_vars+model.gen_grads_and_vars:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    return display_fetches, fetches_enq, save_and_dequeue


def test(display_fetches, fetches_enq, max_steps, epoch_size, *, opt=args,
         epoch=None):
    global epoch_
    if epoch is not None:
        epoch_ = epoch
        edir = f'/epoch/e{epoch}'
    else:
        edir = ''
    tmp = opt.output_dir
    max_steps = min(max_steps, epoch_size)
    t0 = time.time()
    for dir in (['seen'] if 'vkitti' in opt.datatype else ['seen', 'unseen']):
        scores = []
        d_scores = []
        d2_scores = []
        print(dir)
        opt.output_dir = os.path.join(tmp, dir)
        for step in range(max_steps):
            _, results, d, d2 = sess.run(
                (fetches_enq, display_fetches['score'], display_fetches['Dis'],
                 display_fetches['Dis2']))
            scores.append(results)
            d_scores.append(d)
            d2_scores.append(d2)
            if step % opt.progress_freq == 0:
                print(f'at {step} with score {results:.2f}/{d:.3f}/{d2:.3f}' +
                      ' '*10, end='\r')
        with open(opt.output_dir+edir+'/score.npy', 'wb') as f:
            np.save(f, np.array(scores))
        with open(opt.output_dir+edir+'/d_score.npy', 'wb') as f:
            np.save(f, np.array(d_scores))
        with open(opt.output_dir+edir+'/d2_score.npy', 'wb') as f:
            np.save(f, np.array(d2_scores))
        print('wrote index at', opt.output_dir)


def main(*, opt=args):
    global OBJECT_QUEUE, sess
    # --- Initialise main function
    if tf.__version__.split('.')[0] != '1':
        raise Exception('Tensorflow version 1 required')

    if opt.seed is None:
        opt.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    if not os.path.exists(os.path.join(opt.output_dir, 'epoch')):
        os.makedirs(os.path.join(opt.output_dir, 'epoch'))

    # load some options from the checkpoint
    if opt.mode in {'test', 'validate'}:
        if 'test' not in opt.datatype:
            opt.datatype = 'basic_test'
        if opt.baseline not in {'projective', 'identity'}:
            if opt.checkpoint is None:
                raise Exception('checkpoint required for test mode')
            options = {'ngf', 'ndf', 'sample_lambda', 'use_bias',
                       'between_lambda'}
            if os.path.exists(os.path.join(opt.checkpoint, 'options.json')):
                with open(os.path.join(opt.checkpoint, 'options.json')) as f:
                    for key, val in json.loads(f.read()).items():
                        if key in options:
                            print('loaded', key, '=', val)
                            setattr(opt, key, val)

    for k, v in opt._get_kwargs():
        print(k, '=', v)

    with open(os.path.join(opt.output_dir, 'options.json'), 'w') as f:
        f.write(json.dumps(vars(opt), sort_keys=True, indent=4))

    # ---- Build model
    # inputs and targets are [batch_size, height, width, channels]
    OBJECT_QUEUE, epoch_size = get_generator(opt.datatype, opt.max_epochs)
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

    logdir = opt.output_dir if (opt.trace_freq > 0 or opt.summary_freq > 0) \
        else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    # don't horde the full gpu
    config = tf.ConfigProto()
    if 0.0 < opt.gpu_frac < 1.0:
        config.gpu_options.per_process_gpu_memory_fraction = opt.gpu_frac
    elif opt.gpu_frac < -0.5:
        config.gpu_options.allow_growth = True

    max_steps = default.MAX_STEPS
    if opt.max_epochs is not None:
        max_steps = examples.steps_per_epoch * opt.max_epochs
    if opt.max_steps is not None:
        max_steps = opt.max_steps

    # ---- start process
    with sv.managed_session(config=config) as sess:
        if opt.mode is not 'validate'and opt.checkpoint is not None:
            print('loading model from checkpoint')
            checkpoint = tf.train.latest_checkpoint(opt.checkpoint)
            print(checkpoint)
            if checkpoint is None:
                saver.restore(sess, opt.checkpoint)
            else:
                saver.restore(sess, checkpoint)

        if opt.mode in {'train'}:
            for _ in range(sess.run(sv.global_step)):
                next(OBJECT_QUEUE)
        t = threading.Thread(target=load_and_enqueue)
        t.start()
        if opt.mode in {'test', 'validate'}:
            t2 = threading.Thread(target=save_and_dequeue)
            t2.start()

        print('parameter_count =', sess.run(parameter_count))

        if opt.mode == 'validate':
            # validating, test for each checkpoint in the epoch folder
            checkpoints = [os.path.splitext(f)[0] for f in glob.glob(
                os.path.join(opt.checkpoint, 'epoch/model*.index'))]
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            scores = np.zeros((3, len(checkpoints)))
            tmp = opt.output_dir
            for checkpoint in checkpoints:
                opt.output_dir = tmp
                print(f'loading next checkpoint: {checkpoint}')
                i = checkpoint.split('-')[-1]
                saver.restore(sess, checkpoint)
                test(display_fetches, fetch_enq, max_steps, epoch_size,
                     epoch=i)
            return
        elif opt.mode == 'test':
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
                if should(opt.trace_freq):
                    options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    'train': model.train,
                    'global_step': sv.global_step,
                }

                if should(opt.progress_freq):
                    fetches['discrim_loss'] = model.discrim_loss
                    fetches['gen_loss_GAN'] = model.gen_loss_GAN
                    fetches['gen_loss_L1'] = model.gen_loss_L1
                    fetches['paths'] = examples.paths

                if should(opt.summary_freq):
                    fetches['summary'] = sv.summary_op

                if should(opt.display_freq):
                    fetches['display'] = display_fetches

                results = sess.run(fetches, options=options,
                                   run_metadata=run_metadata)

                if should(opt.summary_freq):
                    print('recording summary')
                    sv.summary_writer.add_summary(results['summary'],
                                                  results['global_step'])

                if should(opt.display_freq):
                    print('saving display images')
                    filesets = save_images(results['display'],
                                           step=results['global_step'])
                    append_index(filesets, step=True)

                if should(opt.trace_freq):
                    print('recording trace')
                    sv.summary_writer.add_run_metadata(
                        run_metadata, 'step_%d' % results['global_step'])

                if should(opt.progress_freq):
                    # global_step will have the correct step count if we resume
                    # from a checkpoint.
                    train_epoch = math.ceil(results['global_step'] /
                                            examples.steps_per_epoch)
                    train_step = (results['global_step'] - 1) % \
                        examples.steps_per_epoch + 1
                    rate = (step + 1 - start_step) * opt.batch_size / \
                        (time.time() - start)
                    remaining = (max_steps - step + start_step) * \
                        opt.batch_size / rate
                    print(f'progress  epoch {train_epoch}  step {train_step}  '
                          f'image/sec {rate:0.1f}  remaining {t_s(remaining)}')

                    print('discrim_loss', results['discrim_loss'])
                    print('gen_loss_GAN', results['gen_loss_GAN'])
                    print('gen_loss_L1', results['gen_loss_L1'])
                    print('paths', results['paths'])

                if should(opt.save_freq):
                    print('saving model')
                    saver.save(sess, os.path.join(opt.output_dir, 'model'),
                               global_step=sv.global_step)

                if step == 0 or \
                        (results['global_step'] / examples.steps_per_epoch) % \
                        opt.epoch_freq == 0:
                    print('epoch save')
                    tmp = results['global_step'] // examples.steps_per_epoch
                    epoch_saver.save(sess, os.path.join(opt.output_dir,
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
    if 'vkitti' in args.datatype:
        IMG_SHAPE = [256, 768, 3]
    main()
