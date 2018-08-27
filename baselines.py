#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Baselines that can be used in itergan.py

    by: Ysbrand Galama
    latest version: 2018-03-19
    build for python 3.6 and TensorFlow 1.0
"""

import tensorflow as tf
import numpy as np
from scipy.linalg import svd

import default
from itergan import Model, create_generator, create_discriminator

import sys
assert sys.version_info >= (3, 6), 'Please update your python!'


def create_pix2pix_model(inputs, targets, *, opt=None):
    assert opt is not None, 'not possible to call without explicit options'
    out_channels = int(targets.get_shape()[-1])
    assert out_channels == int(inputs.get_shape()[-1])

    with tf.variable_scope('generator') as scope:
        outputs = create_generator(inputs, out_channels, opt=opt)

    iters = [outputs]
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
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + default.EPS) +
                                      tf.log(1 - predict_fake + default.EPS)))

    with tf.name_scope('generator_loss'):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + default.EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * opt.gan_weight + gen_loss_L1 * \
            opt.l1_weight

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    if opt.mode in {'train'}:
        with tf.name_scope('discriminator_train'):
            discrim_tvars = [var for var in tf.trainable_variables()
                             if var.name.startswith('discriminator')]
            discrim_optim = tf.train.AdamOptimizer(opt.lr,
                                                   opt.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(
                discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(
                discrim_grads_and_vars)

        with tf.name_scope('generator_train'):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables()
                             if var.name.startswith('generator')]
                gen_optim = tf.train.AdamOptimizer(opt.lr,
                                                   opt.beta1)
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
            rest={}
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
            rest={}
        )


def create_projective_model(inputs, targets):
    # original points
    XYZ = np.array([[0, -10, 20, 1], [0, 10, 20, 1], [0, 10, 0, 1],
                    [0, -10, 0, 1]]).transpose()

    def R(r):
        r = np.deg2rad(r)
        return np.array([[np.cos(r), -np.sin(r), 0, 0],
                         [np.sin(r), np.cos(r), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    # rotated points
    XYZp = R(30) @ XYZ

    # camera matrices
    def Mi(f):
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1/f]])

    Me = np.array([[0, 1, 0, 0], [0, 0, 1, 30], [-1, 0, 0, -124.5]])

    # get image-space coordinates
    xyl = Mi(-124.5) @ Me @ XYZ
    xy = xyl / xyl[2, :]
    xylp = Mi(-124.5) @ Me @ XYZp
    xyp = xylp / xylp[2, :]

    # get projective transformation
    x = xy[0, :]+128
    y = xy[1, :]+128
    xp = xyp[0, :]+128
    yp = xyp[1, :]+128
    o = np.ones(x.shape)
    z = np.zeros(x.shape)
    even = np.array([x, y, o, z, z, z, -xp*x, -xp*y, -xp])
    odd = np.array([z, z, z, x, y, o, -yp*x, -yp*y, -yp])
    M = np.hstack((even, odd)).transpose()
    *_, v = svd(M)
    t = v[-1, :]
    t = t / t[-1]
    T = tf.constant(t[:-1], dtype=tf.float32)

    iters = [inputs]
    with tf.name_scope('transform'):
        outputs = tf.contrib.image.transform(inputs, T)

    with tf.name_scope('unused'):
        predict_real = tf.constant(0, dtype=tf.float32)
        predict_fake = tf.constant(0, dtype=tf.float32)
        discrim_loss = tf.constant(0, dtype=tf.float32)
        discrim_grads_and_vars = tf.constant(0, dtype=tf.float32)
        gen_loss_GAN = tf.constant(0, dtype=tf.float32)
        gen_loss_L1 = tf.constant(0, dtype=tf.float32)
        gen_grads_and_vars = tf.constant(0, dtype=tf.float32)
        update_losses = tf.constant(0, dtype=tf.float32)
        gen_train = tf.constant(0, dtype=tf.float32)

    ema = tf.train.ExponentialMovingAverage(decay=default.EMA_DECAY)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

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
        rest={}
    )


def create_identity_model(inputs, targets):
    iters = [inputs]
    outputs = tf.identity(inputs)

    with tf.name_scope('unused'):
        predict_real = tf.constant(0, dtype=tf.float32)
        predict_fake = tf.constant(0, dtype=tf.float32)
        discrim_loss = tf.constant(0, dtype=tf.float32)
        discrim_grads_and_vars = tf.constant(0, dtype=tf.float32)
        gen_loss_GAN = tf.constant(0, dtype=tf.float32)
        gen_loss_L1 = tf.constant(0, dtype=tf.float32)
        gen_grads_and_vars = tf.constant(0, dtype=tf.float32)
        update_losses = tf.constant(0, dtype=tf.float32)
        gen_train = tf.constant(0, dtype=tf.float32)

    ema = tf.train.ExponentialMovingAverage(decay=default.EMA_DECAY)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

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
        rest={}
    )
