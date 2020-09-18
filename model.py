import tensorflow as tf
import os
import time
import numpy as np
# import numLayers from utils
from matplotlib import pyplot as plt


class Model:
    def __init__(self, dataset, root_dir, input_size, output_size):
        self.output_channels = 3
        self.input_size = np.array(input_size)
        self.input_size = np.append(self.input_size, 1)
        self.output_size = np.array(output_size)
        self.output_size = np.append(self.output_size, 1)
        self._lambda = 100
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.dataset = dataset
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,
                                                                beta_1=0.5)
        self.root_dir = root_dir
        self.checkpoint_dir = os.path.join(root_dir, 'checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'chkpt')
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters,
                                   size,
                                   strides=2,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters,
                                            size,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def Generator(self):
        # inputs = tf.keras.layers.Input(shape=[256, 256, 1])
        inputs = tf.keras.layers.Input(shape=self.input_size)

        down_stack = [
            self.downsample(64, 4,
                            apply_batchnorm=False),  # (bs, 128, 128, 64)
            self.downsample(128, 4),  # (bs, 64, 64, 128)
            self.downsample(256, 4),  # (bs, 32, 32, 256)
            self.downsample(512, 4),  # (bs, 16, 16, 512)
            self.downsample(512, 4),  # (bs, 8, 8, 512)
            self.downsample(512, 4),  # (bs, 4, 4, 512)
            self.downsample(512, 4),  # (bs, 2, 2, 512)
            self.downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            self.upsample(512, 4),  # (bs, 16, 16, 1024)
            self.upsample(256, 4),  # (bs, 32, 32, 512)
            self.upsample(128, 4),  # (bs, 64, 64, 256)
            self.upsample(64, 4),  # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(
            self.output_channels,
            4,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='tanh')  # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output),
                                    disc_generated_output)

        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self._lambda * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp,
                                         tar])  # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512,
                                      4,
                                      strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(
                                          zero_pad1)  # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(
            leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1,
                                      4,
                                      strides=1,
                                      kernel_initializer=initializer)(
                                          zero_pad2)  # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output),
                                     disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output),
                                          disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    @tf.function
    def train_step(self, input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target],
                                                  training=True)
            disc_generated_output = self.discriminator(
                [input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
                disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output,
                                                disc_generated_output)

            generator_gradients = gen_tape.gradient(
                gen_total_loss, self.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(
                zip(generator_gradients, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(discriminator_gradients,
                    self.discriminator.trainable_variables))

    def save_images(self, model, dataset, dir, n):
        for inp, tar in dataset.take(1):
            prediction = model(inp, training=True)
            fig = plt.figure(figsize=(15, 15))
            plt.imshow(prediction[0] * 0.5 + 0.5)
            plt.axis('off')
            fig.savefig('{}/{}.png'.format(dir, n))
            plt.close()

    def fit(self, epochs):
        for epoch in range(epochs):
            start = time.time()

            self.save_images(self.generator, self.dataset.test_dataset,
                             os.path.join(self.root_dir, 'figures'), epoch)
            print("Epoch: ", epoch)

            # Train
            for n, (input_image,
                    target) in self.dataset.train_dataset.enumerate():
                print('.', end='')
                if (n + 1) % 100 == 0:
                    print()
                self.train_step(input_image, target, epoch)
            print()

            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time taken for epoch {} is {} sec\n'.format(
                epoch + 1,
                time.time() - start))
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def save_models(self):
        self.generator.save(os.path.join(self.root_dir, 'models/generator.h5'))
        self.discriminator.save(
            os.path.join(self.root_dir, 'models/discriminator.h5'))
