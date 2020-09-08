import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
from matplotlib import pyplot as plt
import math
import random


class Pix2Pix_Model:
    def __init__(self, dataset_path, name, buffer_size, batch_size):
        self.path = dataset_path
        self.name = name
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.image_size = 256
        self.input_image = None
        self.real_image = None

    def load_image(self, image_file):
        """ Load the raw image. """

        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)

        inp = tf.image.rgb_to_grayscale(image)

        self.input_image = tf.cast(inp, tf.float32)
        self.real_image = tf.cast(image, tf.float32)

    def resize(self):
        """ Resize image. """

        self.input_image = tf.image.resize(
            self.input_image, [self.image_size, self.image_size],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.real_image = tf.image.resize(
            self.real_image, [self.image_size, self.image_size],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def normalize(self):
        """ Normalize. """

        self.input_image = (self.input_image / 127.5) - 1
        self.real_image = (self.real_image / 127.5) - 1

    def random_crop(self):
        """ Take random crop at self.image_size. """

        offset_h = math.floor(random.random() *
                              (self.input_image.shape[0] - self.image_size))
        offset_w = math.floor(random.random() *
                              (self.input_image.shape[1] - self.image_size))

        self.input_image = tf.image.crop_to_bounding_box(
            self.input_image, offset_h, offset_w, self.image_size,
            self.image_size)
        self.real_image = tf.image.crop_to_bounding_box(
            self.real_image, offset_h, offset_w, self.image_size,
            self.image_size)

    def greyscale(self):
        """ Convert to greyscale. """

        self.input_image = tf.image.rgb_to_grayscale(self.input_image)

    def downscale_upscale(self):
        """ Downscale using AREA filtering, upscale using BICUBIC
        to get blurry image. """

        width = tf.shape(self.input_image)[1]
        height = tf.shape(self.input_image)[0]
        self.input_image = tf.image.resize(self.input_image,
                                           [height // 24, width // 24],
                                           method=tf.image.ResizeMethod.AREA)
        self.input_image = tf.image.resize(
            self.input_image, [height, width],
            method=tf.image.ResizeMethod.BICUBIC)

    def random_brightness(self):
        """ Apply random brightness. """

        self.input_image = tf.image.random_brightness(self.input_image, 0.2)

    def random_contrast(self):
        """ Apply random contrast. """

        self.input_image = tf.image.random_contrast(self.input_image, 0.85,
                                                    1.15)

    def random_saturation(self):
        """ Apply random saturation. """

        self.input_image = tf.image.random_saturation(self.input_image, 0, 1)
