import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display
from helpers import getNumLayers


class Model:
    def __init__(self, dataset, root_dir, input_size, output_size, notebook):
        self.output_channels = 3
        self.input_size = [input_size, input_size, 1]
        self.output_size = [output_size, output_size, 1]
        self.numDownLayers = getNumLayers(self.input_size[0])
        self.numUpLayers = getNumLayers(self.output_size[0])
        self.down_stack_list, self.up_stack_list = self.createLayersList(
            self.numDownLayers, self.numUpLayers)
        self._lambda = 100
        self.notebook = notebook
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.dataset = dataset
        self.generator = self.Generator()

        # self.generator.summary()

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

    def createLayersList(self, numDown, numUp):
        sequence = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        return sequence[numDown - 1::-1], sequence[1:numUp]

    def Generator(self):
        # inputs = tf.keras.layers.Input(shape=[256, 256, 1])
        inputs = tf.keras.layers.Input(shape=self.input_size)

        # Generate Down and Up stacks
        down_stack = [
            self.downsample(n, 4, apply_batchnorm=(i != 0))
            for i, n in enumerate(self.down_stack_list)
        ]

        up_stack = [
            self.upsample(n, 4, apply_dropout=(i < 3))
            for i, n in enumerate(self.up_stack_list)
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

    def traverseDownStack(self, res, funcs, i):
        """
        A recursive function which traverses through the stack of
        functions given as an array, passing the result of the previous to the
        next function.
        """
        if i == len(funcs):
            return res
        else:
            new_res = funcs[i](res)
            return self.traverseDownStack(new_res, funcs, i + 1)

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=self.input_size, name='input_image')
        tar = tf.keras.layers.Input(shape=self.input_size, name='target_image')

        # (bs, input_shape, input_shape, channels*2)
        x = tf.keras.layers.concatenate([inp, tar])

        num_down_stack = self.numDownLayers - 5

        down_stack = [
            self.downsample(n, 4, apply_batchnorm=(i != 0))
            for i, n in enumerate(self.down_stack_list[:num_down_stack])
        ]

        # Recursively traverse generated downstack to
        # reach shape of (bs, 32, 32, 256)
        down3 = self.traverseDownStack(x, down_stack, 0)

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

    def generate_images(model, test_input, tar):
        prediction = model.predict(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    def fit(self, epochs):
        for epoch in range(epochs):
            start = time.time()

            if self.notebook:
                display.clear_output(wait=True)

                for example_input, example_target in self.dataset.test_dataset.take(
                        1):
                    self.generate_images(self.generator, example_input,
                                         example_target)

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
