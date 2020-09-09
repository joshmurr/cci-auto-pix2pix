#  import numpy as np
import tensorflow as tf
#  from tensorflow import keras
#  import os
#  import time
#  from matplotlib import pyplot as plt
import math
import random


class Dataset:
    def __init__(self, args):
        self.path = args.dataset_path
        self.name = args.name
        self.image_size = 256

        self.train_dataset, self.test_dataset = self.init_dataset(
            args.dataset_path, args.buffer_size, args.batch_size)

    def init_dataset(self, dataset_path, buffer_size, batch_size):
        train_dataset = tf.data.Dataset.list_files(dataset_path +
                                                   '/train/*.jpg')
        train_dataset = train_dataset.map(
            self.load_image_train,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size)
        train_dataset = train_dataset.batch(batch_size)
        test_dataset = tf.data.Dataset.list_files(dataset_path + '/test/*.jpg')
        test_dataset = test_dataset.map(self.load_image_test)
        test_dataset = test_dataset.batch(batch_size)

        return train_dataset, test_dataset

    def load_image(self, image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)

        inp = tf.image.rgb_to_grayscale(image)

        input_image = tf.cast(inp, tf.float32)
        real_image = tf.cast(image, tf.float32)

        return input_image, real_image

    def resize(self, input_image, real_image, height, width):
        input_image = tf.image.resize(
            input_image, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(
            real_image, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image

    def normalize(self, input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image

    def random_crop(self, input_image, real_image):
        offset_h = math.floor(random.random() *
                              (input_image.shape[0] - self.image_size))
        offset_w = math.floor(random.random() *
                              (input_image.shape[1] - self.image_size))

        input_image = tf.image.crop_to_bounding_box(input_image, offset_h,
                                                    offset_w, self.image_size,
                                                    self.image_size)
        real_image = tf.image.crop_to_bounding_box(real_image, offset_h,
                                                   offset_w, self.image_size,
                                                   self.image_size)

        return input_image, real_image

    def greyscale(self, input_image):
        return tf.image.rgb_to_grayscale(input_image)

    def downscale_upscale(self, input_image):
        width = tf.shape(input_image)[1]
        height = tf.shape(input_image)[0]
        input_image = tf.image.resize(input_image, [height // 24, width // 24],
                                      method=tf.image.ResizeMethod.AREA)
        input_image = tf.image.resize(input_image, [height, width],
                                      method=tf.image.ResizeMethod.BICUBIC)

        return input_image

    @tf.function()
    def random_jitter(self, input_image, real_image):
        input_image, real_image = self.resize(input_image, real_image, 286,
                                              286)
        input_image, real_image = self.random_crop(input_image, real_image)
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def load_image_train(self, image_file):
        input_image, real_image = self.load_image(image_file)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image = self.downscale_upscale(input_image)
        input_image = tf.image.random_brightness(input_image, 0.2)
        input_image = tf.image.random_contrast(input_image, 0.85, 1.15)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def load_image_test(self, image_file):
        input_image, real_image = self.load_image(image_file)
        input_image, real_image = self.resize(input_image, real_image,
                                              self.image_size, self.image_size)
        input_image = self.downscale_upscale(input_image)
        input_image = tf.image.random_brightness(input_image, 0.2)
        input_image = tf.image.random_contrast(input_image, 0.85, 1.15)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image
