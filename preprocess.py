#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocess the data (pad and resize)

    by: Ysbrand Galama
    latest version: 2018-03-19
    build for python 3.6 and TensorFlow 1.0
"""

import tensorflow as tf
import threading
import os

import sys
assert sys.version_info >= (3, 6), 'Please update your python!'


QSIZE = 32

IMAGE_ID = ((id, x) for id in range(1, 1001, 1) for x in range(0, 72, 1))

TOT = 72000
RAW_SHAPE = [288, 384, 3]
IMG_SHAPE = [256, 256, 3]
DATA_FILE = '{dir}/aloi_red2_view/png2/{id}/{id}_r{rot}.png'
MASK_FILE = '{dir}/aloi_mask2/mask2/{id}/{id}_r{rot}.png'
MASK_FILE0 = '{dir}/aloi_mask2/mask2/{id}/{id}_c1.png'


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


def preprocess(raw, name='preprocess'):
    """From raw data, create a tensor of correct shape and dtype."""
    with tf.name_scope(name):
        raw = tf.image.convert_image_dtype(raw, dtype=tf.float32)
        raw.set_shape(RAW_SHAPE)
        return raw


def init(input_dir, output_dir, mask=False):
    """Create data queue for training.

    Returns
    -------
    Examples named tuple - object containing the necessarily information of the
        queue
    """
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception('input_dir does not exist')
    if not os.path.exists(output_dir):
        for i in range(1, 1001, 1):
            os.makedirs(output_dir+f'/{i}')

    with tf.name_scope('input_queue'):
        # The image placeholders
        with tf.name_scope('file_inputs'):
            paths = tf.placeholder(tf.string, shape=())
            input_data = tf.placeholder(tf.string, shape=())

        input_image = preprocess(tf.image.decode_png(input_data, channels=3))
        # The tf queue that handles the reading of images
        q_in = tf.FIFOQueue(QSIZE, [tf.string, tf.float32],
                            shapes=[[], RAW_SHAPE])
        enqueue_op = q_in.enqueue([paths, input_image])
        queue_close = q_in.close()

    with tf.name_scope('process'):
        out_path, raw_image = q_in.dequeue()
        image = resize(raw_image, RAW_SHAPE[0], RAW_SHAPE[1], IMG_SHAPE[0])
        if mask:
            image = tf.cast(tf.cast(image, tf.bool), tf.float32)
        img = tf.image.convert_image_dtype(image, dtype=tf.uint8,
                                           saturate=True)
        output_data = tf.image.encode_png(img)
        q_out = tf.FIFOQueue(QSIZE, [tf.string, tf.string],
                             shapes=[[], []])
        enqueue_out = q_out.enqueue([out_path, output_data])

    with tf.name_scope('output_queue'):
        dequeue_op = q_out.dequeue()

    def load_and_enqueue():
        while True:
            try:
                id, x = next(IMAGE_ID)
                rot = (x*5) % 360
                if mask:
                    if rot == 0:
                        input_file = MASK_FILE0.format(dir=input_dir, id=id)
                    else:
                        input_file = MASK_FILE.format(dir=input_dir, id=id,
                                                      rot=rot)
                else:
                    input_file = DATA_FILE.format(dir=input_dir, id=id,
                                                  rot=rot)
                path = f'{output_dir}/{id}/{id}_r{rot}.png'
                with open(input_file, 'rb') as fi:
                    sess.run(enqueue_op, feed_dict={paths: path,
                                                    input_data: fi.read()})
            except tf.errors.CancelledError:
                print('WARNING load_queue stopped')
                break
            except StopIteration:
                print('WARNING load_queue exausted')
                sess.run(queue_close)
                break

    def save_and_dequeue():
        while True:
            try:
                out_path, img_data = sess.run(dequeue_op)
                with open(out_path, 'wb') as fo:
                    fo.write(img_data)
            except (tf.errors.CancelledError, RuntimeError):
                print('WARNING load_queue stopped')
                break

    return enqueue_out, q_out.close(), load_and_enqueue, save_and_dequeue


def init_v(input_dir, output_dir):
    """Create data queue for training.

    Returns
    -------
    Examples named tuple - object containing the necessarily information of the
        queue
    """
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception('input_dir does not exist')

    with tf.name_scope('input_queue'):
        # The image placeholders
        with tf.name_scope('file_inputs'):
            paths = tf.placeholder(tf.string, shape=())
            input_data = tf.placeholder(tf.string, shape=())

        input_image = preprocess(tf.image.decode_png(input_data, channels=3))
        # The tf queue that handles the reading of images
        q_in = tf.FIFOQueue(QSIZE, [tf.string, tf.float32],
                            shapes=[[], RAW_SHAPE])
        enqueue_op = q_in.enqueue([paths, input_image])
        queue_close = q_in.close()

    with tf.name_scope('process'):
        out_path, raw_image = q_in.dequeue()
        with tf.variable_scope('resize'):
            image = tf.image.resize_images(raw_image[:, 58:-59, :], [256, 768])
        img = tf.image.convert_image_dtype(image, dtype=tf.uint8,
                                           saturate=True)
        output_data = tf.image.encode_png(img)
        q_out = tf.FIFOQueue(QSIZE, [tf.string, tf.string],
                             shapes=[[], []])
        enqueue_out = q_out.enqueue([out_path, output_data])

    with tf.name_scope('output_queue'):
        dequeue_op = q_out.dequeue()

    def load_and_enqueue():
        mask_path = os.path.dirname(output_dir)+'/MASK.png'
        white_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x04\xda\x00\x00\x01w\x08\x00\x00\x00\x00VZy\x01\x00\x00\x04\xe0IDATx\x9c\xed\xd4\xc1\t\x00 \x10\xc00u\xff\x9d\xcf%\x04\xa1$\x13\xf4\xd5=\x0b\xa0\xe6\xfc\x0e\x00x\xcf\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80 k\x03\x82\xac\r\x08\xb26 \xc8\xda\x80\xa0\x0b\x03S\x03\xedZU\x80\x04\x00\x00\x00\x00IEND\xaeB`\x82'
        sess.run(enqueue_op, feed_dict={paths: mask_path,
                                        input_data: white_png})
        for scene in [1, 2, 6, 18, 20]:
            scene_path = os.path.join(
                input_dir, f'vkitti_1.3.1_rgb/vkitti_1.3.1_rgb/{scene:04d}/')
            max_id = len(os.listdir(scene_path+'clone'))
            for r, p in [(0, '30-deg-left'), (15, '15-deg-left'),
                         (30, 'clone'), (45, '15-deg-right'),
                         (60, '30-deg-right')]:
                for id in range(max_id):
                    input_file = scene_path+f'{p}/{id:05d}.png'
                    idx = scene*1000+id
                    path = f'{output_dir}/{idx}/{idx}_r{r}.png'
                    print(f'at {path}', end='\r')
                    with open(input_file, 'rb') as fi:
                        sess.run(enqueue_op, feed_dict={paths: path,
                                                        input_data: fi.read()})
        raise tf.errors.CancelledError

    def save_and_dequeue():
        while True:
            try:
                out_path, img_data = sess.run(dequeue_op)
                dir = os.path.dirname(out_path)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                with open(out_path, 'wb') as fo:
                    fo.write(img_data)
            except (tf.errors.CancelledError, RuntimeError):
                print('WARNING load_queue stopped')
                break

    return enqueue_out, q_out.close(), load_and_enqueue, save_and_dequeue


def main(input_dir, mask=False, vkitti=False):
    global sess
    if vkitti:
        output_dir = os.path.join(input_dir, 'vkitti_preprocessed/img')
        enqueue, queue_close, load_and_enqueue, save_and_dequeue = \
            init_v(input_dir, output_dir)
    else:
        output_dir = os.path.join(input_dir, 'aloi_preprocessed/' +
                                  ('mask' if mask else 'img'))
        enqueue, queue_close, load_and_enqueue, save_and_dequeue = \
            init(input_dir, output_dir, mask)

    sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        t1 = threading.Thread(target=load_and_enqueue)
        t2 = threading.Thread(target=save_and_dequeue)
        t1.start()
        t2.start()
        print('started')
        i = 0
        while True:
            print(f'{i}/{TOT}', end='\r')
            i += 1
            try:
                sess.run(enqueue)
            except tf.errors.CancelledError:
                print('WARNING load_queue stopped')
                t1.join()
                sess.run(queue_close)
                t2.join()
                break


def error():
    print('Please use as follows\n> python preprocess.py <aloi_dir>, to proces'
          's the aloi colour data, \n> python preprocess.py <aloi_dir> mask, t'
          'o process the binary masks, and\n python preprocess.py <vkitti_dir>'
          ' vkitti, to process the vkitti data')
    sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        error()
    args = {s.lower() for s in sys.argv}
    if 'vkitti' in args:
        RAW_SHAPE = [375, 1242, 3]
        tmp1 = os.path.join(sys.argv[1], 'vkitti_1.3.1_rgb/vkitti_1.3.1_rgb')
        if not os.path.isdir(tmp1):
            error()
        main(sys.argv[1], vkitti=True)
    else:
        tmp1 = os.path.join(sys.argv[1], 'aloi_red2_view')
        tmp2 = os.path.join(sys.argv[1], 'aloi_mask2')
        if not (os.path.isdir(tmp1) or os.path.isdir(tmp2)):
            error()
        mask = 'mask' in args
        main(sys.argv[1], mask)
