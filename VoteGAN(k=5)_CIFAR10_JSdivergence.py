import os
import glob
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import time
import PIL
import imageio

from IPython import display

import tensorflow as tf
from tensorflow.keras import datasets, layers, Sequential, Model
from tensorflow.keras import metrics
from tensorflow.keras import constraints

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['NCCL_TOPOLOGY'] = 'CUBEMESH'

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#number of discriminators
num_disc = 5

#load CIFAR10
(train_CIFAR10, train_labels), (test_CIFAR10, test_labels) = datasets.cifar10.load_data()
train_images = train_CIFAR10.astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
test_images = test_CIFAR10.astype('float32')
test_images = (test_images - 127.5) / 127.5  # Normalize to [-1, 1]

target_size = 6000
target_images = train_images[:target_size]


def EvenlySplitting(dataset, size=5):
    chunk_size = dataset.shape[0] // size

    splitted_datasets = []
    for i in range(size):
        temp_data = dataset[i * chunk_size:(i + 1) * chunk_size]
        splitted_datasets.append(temp_data)

    return splitted_datasets


train_datasets = EvenlySplitting(target_images, size=num_disc)
print(train_datasets[0].shape)

#GAN 빌드
noise_dim = 100
nrow = target_images.shape[1]
ncol = target_images.shape[2]
channels = target_images.shape[3]
input_shape = (nrow, ncol, channels)

#생성자
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4 * 4 * 256, use_bias=False, input_shape=(100,)))

    model.add(layers.Reshape((4, 4, 256)))
    assert model.output_shape == (None, 4, 4, 256)  # Note: None is the batch size
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(256, (5, 5),
                               strides=(1, 1),
                               padding='same',
                               use_bias=False,
                               # kernel_initializer=tf.keras.initializers.GlorotUniform()
                               ))
    assert model.output_shape == (None, 4, 4, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(128, (5, 5),
                               strides=(2, 2),
                               padding='same',
                               use_bias=False,
                               # kernel_initializer=tf.keras.initializers.GlorotUniform()
                               ))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(64, (5, 5),
                               strides=(2, 2),
                               padding='same',
                               use_bias=False,
                               # kernel_initializer=tf.keras.initializers.GlorotUniform()
                               ))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(channels, (5, 5),
                               strides=(2, 2),
                               padding='same',
                               use_bias=False,
                               # kernel_initializer=tf.keras.initializers.GlorotUniform(),
                               activation='tanh'))
    assert model.output_shape == (None, nrow, ncol, channels)

    return model

generator = make_generator_model()
noise = tf.random.normal([1, 100])

#판별자
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(64, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      input_shape=[nrow, ncol, channels]))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv2D(128, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      # kernel_initializer=tf.keras.initializers.GlorotUniform()
                      ))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv2D(256, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      # kernel_initializer=tf.keras.initializers.GlorotUniform()
                      ))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

discriminators = []
for i in range(num_disc):
    discriminators.append(make_discriminator_model())

#DCGAN loss_bce
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# # RMSprop
# generator_optimizer = tf.keras.optimizers.RMSprop(5e-5)
# discriminator_optimizer = tf.keras.optimizers.RMSprop(5e-5)
# Adam
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
discriminator_optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)




#############training######################
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

def generate_and_save_images(model, epoch, test_input):

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) / 2)
        plt.axis('off')

    plt.savefig('images/epoch_{:04d}.png'.format(epoch))
    plt.show()


@tf.function
def train_step(batch_size, zipped_image_batch):
    for _ in range(2):
        all_disc_loss = []
        for i in range(num_disc):
            noise = tf.random.normal([batch_size, noise_dim])
            with tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)
                real_output = discriminators[i](zipped_image_batch[i], training=True)
                fake_output = discriminators[i](generated_images, training=True)

                disc_loss = discriminator_loss(real_output, fake_output)

                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminators[i].trainable_variables)
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                            discriminators[i].trainable_variables))
            all_disc_loss.append(disc_loss)


    for _ in range(1):
        with tf.GradientTape() as gen_tape:
            all_gen_loss = []
            for i in range(num_disc):
                noise = tf.random.normal([batch_size, noise_dim])
                generated_images = generator(noise, training=True)
                fake_output = discriminators[i](generated_images, training=True)
                gen_loss = generator_loss(fake_output)

                all_gen_loss.append(gen_loss)

            avg_gen_loss = tf.reduce_mean(all_gen_loss)
            gradients_of_generator = gen_tape.gradient(avg_gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return avg_gen_loss, all_disc_loss


def train(train_datasets, epochs, batch_size):
    train_batches = []
    for i in range(num_disc):
        train_batches.append(
            tf.data.Dataset.from_tensor_slices(train_datasets[i]).shuffle(
                train_datasets[i].shape[0]).batch(batch_size,
                                                  drop_remainder=True))

    gen_loss_log, disc_loss_log = [], []
    for epoch in range(epochs):
        start = time.time()

        for zipped_image_batch in zip(*train_batches):
            gen_loss, all_disc_loss = train_step(batch_size, zipped_image_batch)
        gen_loss_log.append(gen_loss.numpy())
        disc_loss_log.append(all_disc_loss)

        if (epoch + 1) % 100 == 0:
            display.clear_output(wait=True)
            generate_and_save_images(generator, epoch + 1, seed)

        print('gen_loss: {}, disc_loss: {}'.format(
            gen_loss.numpy(), [loss.numpy() for loss in all_disc_loss]))
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    return gen_loss_log, disc_loss_log


checkpoint_dir = './training_checkpoints/Vote-DCGAN(k={})'.format(num_disc)
checkpoint_prefix = os.path.join(checkpoint_dir, "v0.1")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator0=discriminators[0],
    discriminator1=discriminators[1])



EPOCHS = 1500
BATCH_SIZE = 64
gen_loss_log, disc_loss_log = train(train_datasets, epochs=EPOCHS, batch_size=BATCH_SIZE)

#Membership Infernce Attack
def WBattack_priv(X, X_comp, discriminators):
    Dat = np.concatenate([X, X_comp])
    Pred = []

    for i in range(len(discriminators)):
        Pred += [discriminators[i].predict(Dat)[:, 0]]

    p_mean = np.mean(Pred, axis=0)
    p_max = np.max(Pred, axis=0)

    In_mean = np.argsort(-p_mean)
    In_mean = In_mean[:len(X)]

    In_max = np.argsort(-p_max)
    In_max = In_max[:len(X)]

    Acc_max = np.sum(1. * (In_max < len(X))) / len(X)
    Acc_mean = np.sum(1. * (In_mean < len(X))) / len(X)

    print('White-box attack accuracy (max):', Acc_max)
    print('White-box attack accuracy (mean):', Acc_mean)

    return (Acc_max, Acc_mean)
WBattack_priv(target_images, test_images[:target_size], discriminators)