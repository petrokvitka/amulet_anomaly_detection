#!/usr/bin/env python

"""
This script uses a created dataset of MFCCs to train a DCGAN.
The training of the DCGAN is a complex process and to avoid the
loss of information, Checkpoints are saved during the training.

@author: Anastasiia Petrova
@contact: petrokvitka@gmail.com

"""

import os
import time
import tensorflow as tf
import numpy as np
from glob import glob
import datetime
import random
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import sys


def generator(z, output_channel_dim, training):
    """
    Create the Generator.
    :param z: the input stream
    :param output_channel_dim: specify the output dimensions
    :param training: true if the training is required
    :returns: the last layer with tanh activation
    """
    with tf.compat.v1.variable_scope("generator", reuse = not training):

        fully_connected = tf.compat.v1.layers.dense(z, 5*11*256)
        fully_connected = tf.reshape(fully_connected, (-1, 5, 11, 256))
        fully_connected = tf.nn.leaky_relu(fully_connected)

        # 5x11x256 -> 10x22x128
        trans_conv1 = tf.compat.v1.layers.conv2d_transpose(inputs=fully_connected,
                                                 filters=128,
                                                 kernel_size=[5,5],
                                                 strides=[2,2],
                                                 padding="SAME",
                                                 kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV),
                                                 name="trans_conv1")
        batch_trans_conv1 = tf.compat.v1.layers.batch_normalization(inputs = trans_conv1,
                                                          training=training,
                                                          epsilon=EPSILON,
                                                          name="batch_trans_conv1")
        trans_conv1_out = tf.nn.leaky_relu(batch_trans_conv1,
                                           name="trans_conv1_out")

        # 10x22x128 -> 20x44x64
        trans_conv2 = tf.compat.v1.layers.conv2d_transpose(inputs=trans_conv1_out,
                                                 filters=64,
                                                 kernel_size=[5,5],
                                                 strides=[2,2],
                                                 padding="SAME",
                                                 kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV),
                                                 name="trans_conv2")
        batch_trans_conv2 = tf.compat.v1.layers.batch_normalization(inputs = trans_conv2,
                                                          training=training,
                                                          epsilon=EPSILON,
                                                          name="batch_trans_conv2")
        trans_conv2_out = tf.nn.leaky_relu(batch_trans_conv2,
                                           name="trans_conv2_out")

        # 20x44x64 -> 20x44x3
        logits = tf.compat.v1.layers.conv2d_transpose(inputs=trans_conv2_out,
                                            filters=3,
                                            kernel_size=[5,5],
                                            strides=[1,1],
                                            padding="SAME",
                                            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV),
                                            name="logits")
        out = tf.tanh(logits, name="out")
        return out


def discriminator(x, reuse):
    """
    Create the Discriminator.
    :param x: the input stream
    :param reuse: true if no training is required
    :returns: the last layer with sigmoid activation
    """
    with tf.compat.v1.variable_scope("discriminator", reuse=reuse):

        # 20x44x3 -> 10x22x64
        conv1 = tf.compat.v1.layers.conv2d(inputs=x,
                                 filters=64,
                                 kernel_size=[5,5],
                                 strides=[2,2],
                                 padding="SAME",
                                 kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV),
                                 name='conv1')
        batch_norm1 = tf.compat.v1.layers.batch_normalization(conv1,
                                                    training=True,
                                                    epsilon=EPSILON,
                                                    name='batch_norm1')
        conv1_out = tf.nn.leaky_relu(batch_norm1,
                                     name="conv1_out")

        # 10x22x64-> 5x11x128
        conv2 = tf.compat.v1.layers.conv2d(inputs=conv1_out,
                                 filters=128,
                                 kernel_size=[5, 5],
                                 strides=[2, 2],
                                 padding="SAME",
                                 kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV),
                                 name='conv2')
        batch_norm2 = tf.compat.v1.layers.batch_normalization(conv2,
                                                    training=True,
                                                    epsilon=EPSILON,
                                                    name='batch_norm2')
        conv2_out = tf.nn.leaky_relu(batch_norm2,
                                     name="conv2_out")

        flatten = tf.reshape(conv2_out, (-1, 5*11*128))
        logits = tf.compat.v1.layers.dense(inputs=flatten,
                                 units=1,
                                 activation=None)
        out = tf.sigmoid(logits)
        return out, logits


def model_loss(input_real, input_z, output_channel_dim):
    """
    Calculate the model loss.
    :param input_real: use the real input from the dataset
    :param input_z: input for the generator
    :param output_channel_dim: specify the output dimensions
    :returns: the loss of the discriminator and the loss of the generator
    """
    g_model = generator(input_z, output_channel_dim, True)

    noisy_input_real = input_real + tf.random.normal(shape=tf.shape(input_real),
                                                     mean=0.0,
                                                     stddev=random.uniform(0.0, 0.1),
                                                     dtype=tf.float32)

    d_model_real, d_logits_real = discriminator(noisy_input_real, reuse=False)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_model_real)*random.uniform(0.9, 1.0)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_model_fake)))
    d_loss = tf.reduce_mean(0.5 * (d_loss_real + d_loss_fake))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_model_fake)))
    return d_loss, g_loss


def model_optimizers(d_loss, g_loss):
    """
    Create optimizers.
    :param d_loss: calculated discriminator loss
    :param g_loss: calculated generator loss
    :returns: discriminator optimizer and generator optimizer
    """
    t_vars = tf.compat.v1.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith("generator")]
    d_vars = [var for var in t_vars if var.name.startswith("discriminator")]

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    gen_updates = [op for op in update_ops if op.name.startswith('generator') or op.name.startswith('discriminator')]

    with tf.control_dependencies(gen_updates):
        d_train_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=LR_D, beta1=BETA1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=LR_G, beta1=BETA1).minimize(g_loss, var_list=g_vars)
    return d_train_opt, g_train_opt


def model_inputs(real_dim, z_dim):
    """
    Prepare inputs for the model.
    :param real_dim: dimensions of the real input
    :param z_dim: dimensions of the generated input
    :returns: inputs and the learning rates
    """
    inputs_real = tf.compat.v1.placeholder(tf.float32, (None, *real_dim), name='inputs_real')
    #inputs_real = tf.keras.Input(name='inputs_real', shape=(None, *real_dim), dtype=tf.dtypes.float32)
    inputs_z = tf.compat.v1.placeholder(tf.float32, (None, z_dim), name="input_z")
    #inputs_z = tf.keras.Input(name='input_z', shape=(None, z_dim), dtype=tf.dtypes.float32)
    learning_rate_G = tf.compat.v1.placeholder(tf.float32, name="lr_g")
    #learning_rate_G = tf.keras.Input(shape=(), name='lr_g', dtype=tf.dtypes.float32)
    learning_rate_D = tf.compat.v1.placeholder(tf.float32, name="lr_d")
    #learning_rate_D = tf.keras.Input(shape=(), name='lr_d', dtype=tf.dtypes.float32)
    return inputs_real, inputs_z, learning_rate_G, learning_rate_D


def show_samples(sample_images, name, epoch):
    """
    Save created MFCCs as images.
    :param sample_images: MFCCs to show
    :param name: name for saving the MFCC
    :param epoch: the epoch when the MFCC was generated
    """
    figure, axes = plt.subplots(1, len(sample_images), figsize = (IMAGE_SIZE_H, IMAGE_SIZE_W))
    for index, axis in enumerate(axes):
        axis.axis('off')
        image_array = sample_images[index]
        axis.imshow(image_array)
        image = Image.fromarray(image_array)
        image.save(name+"_"+str(epoch)+"_"+str(index)+".png")
        plt.imsave(name+"_"+str(epoch)+"_"+str(index)+".jpg", arr = image_array, format = "jpg")
    plt.savefig(name+"_"+str(epoch)+".png", bbox_inches='tight', pad_inches=0)
    #plt.show()
    plt.close()


def test(sess, input_z, out_channel_dim, epoch):
    """
    Generate MFCCs for the visualization.
    :param sess: the current state of the DCGAN model
    :param input_z: the input stream
    :param out_channel_dim: dimensions of the generated signal
    :param epoch: epoch for the generating the MFCCs
    """
    example_z = np.random.uniform(-1, 1, size=[SAMPLES_TO_SHOW, input_z.get_shape().as_list()[-1]])
    samples = sess.run(generator(input_z, out_channel_dim, False), feed_dict={input_z: example_z})
    final_output = tf.compat.v1.identity(samples, name = "predictions")
    sample_images = [((sample + 1.0) * 127.5).astype(np.uint8) for sample in samples]

    show_samples(sample_images, OUTPUT_DIR + "samples", epoch)


def summarize_epoch(epoch, sess, d_losses, g_losses, input_z, data_shape, saver):
    """
    Summarize the training after an epoch.
    :param epoch: the epoch to summarize
    :param sess: the current state of the trained model
    :param d_losses: loss of the Discriminator
    :param g_losses: loss of the Generator
    :param input_z: the input stream
    :param data_shape: the shape of the input dataset
    :param saver: the saver for the Checkpoints
    """
    print("\nEpoch {}/{}".format(epoch, EPOCHS),
          "\nD Loss: {:.5f}".format(np.mean(d_losses[-MINIBATCH_SIZE:])),
          "\nG Loss: {:.5f}".format(np.mean(g_losses[-MINIBATCH_SIZE:])))
    fig, ax = plt.subplots()
    plt.plot(d_losses, label='Discriminator', alpha=0.6)
    plt.plot(g_losses, label='Generator', alpha=0.6)
    plt.title("Losses")
    plt.legend()
    plt.savefig(OUTPUT_DIR + "losses_" + str(epoch) + ".png")
    #plt.show()
    plt.close()
    saver.save(sess, OUTPUT_DIR + "model_" + str(epoch) + ".ckpt")

    test(sess, input_z, data_shape[3], epoch)


def get_batch(dataset):
    """
    Get a batch of data from the dataset.
    :param dataset: the MFCCs dataset
    :returns: normalized batch and the files
    """
    files = random.sample(dataset, BATCH_SIZE)
    batch = []
    for file in files:
        if random.choice([True, False]):
            batch.append(np.asarray(Image.open(file).transpose(Image.FLIP_LEFT_RIGHT)))
        else:
            batch.append(np.asarray(Image.open(file)))
    batch = np.asarray(batch)
    normalized_batch = (batch / 127.5) - 1.0
    return normalized_batch, files


def train(data_shape, epoch, checkpoint_path):
    """
    The main function for the training.
    :param data_shape: the shape of the dataset
    :param epoch: the number of epochs for the training
    :param checkpoint_path: the path to the saved checkpoint
    """
    input_images, input_z, lr_G, lr_D = model_inputs(data_shape[1:], NOISE_SIZE)
    d_loss, g_loss = model_loss(input_images, input_z, data_shape[3])
    d_opt, g_opt = model_optimizers(d_loss, g_loss)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        if checkpoint_path is not None:
            saver.restore(sess, checkpoint_path)

        iteration = 0
        d_losses = []
        g_losses = []

        for epoch in range(EPOCH, EPOCHS):
            epoch_dataset = DATASET.copy()

            for i in range(MINIBATCH_SIZE):
                iteration_start_time = time.time()
                iteration += 1
                batch_images, used_files = get_batch(epoch_dataset)
                [epoch_dataset.remove(file) for file in used_files]

                batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, NOISE_SIZE))
                _ = sess.run(d_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_D: LR_D})
                d_losses.append(d_loss.eval({input_z: batch_z, input_images: batch_images}))

                _ = sess.run(g_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_G: LR_G})
                g_losses.append(g_loss.eval({input_z: batch_z}))

                elapsed_time = round(time.time()-iteration_start_time, 3)
                remaining_files = len(epoch_dataset)
                print("\rEpoch: " + str(epoch) +
                      ", iteration: " + str(iteration) +
                      ", d_loss: " + str(round(d_losses[-1], 3)) +
                      ", g_loss: " + str(round(g_losses[-1], 3)) +
                      ", duration: " + str(elapsed_time) +
                      ", minutes remaining: " + str(round(remaining_files/BATCH_SIZE*elapsed_time/60, 1)) +
                      ", remaining files in batch: " + str(remaining_files)
                      , sep=' ', end=' ', flush=True)
            summarize_epoch(epoch, sess, d_losses, g_losses, input_z, data_shape, saver)


# Get input from user and set Hyperparameters
parser = argparse.ArgumentParser(description = "Train DCGAN.")
parser.add_argument('--epochs', type = int, help = "Epochs for DCGAN training.", default = 1)
parser.add_argument('--dataset', help = "Path to the dataset.", required = True)
parser.add_argument('--dataset_list', help = "Name of the txt file containing the filenames of MFCCs in the dataset.", required = True)
parser.add_argument('--input_mfccs', help = "Name of the directory with MFCCs in the dataset.", required = True)
parser.add_argument('--output_dir', help = "Path to the output directory.", required = True)
parser.add_argument('--train', action = 'store_true', help = "Check if you want to train and save a model.")
parser.add_argument('--detect', action = 'store_true', help = "Check if you want to create MFCCs to detect anomalies.")
parser.add_argument('--checkpoint', help = "For anomaly detection use a trained model and provide a path to a checkpoint here.")
args = parser.parse_args()

def check_path(path):
    if not path.endswith("/"):
        path = path + "/"
    return path

args.dataset = check_path(args.dataset)
args.input_mfccs = check_path(args.input_mfccs)
args.output_dir = check_path(args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    print("A new directory {} is created!".format(args.output_dir))


# Hyperparameters
IMAGE_SIZE_W = 44
IMAGE_SIZE_H = 20
NOISE_SIZE = 100
LR_D = 0.00002
LR_G = 0.0002
BATCH_SIZE = 64
EPOCH = 0 #Non-zero only if we are loading a checkpoint
EPOCHS = args.epochs #1 #750
BETA1 = 0.5
WEIGHT_INIT_STDDEV = 0.02
EPSILON = 0.005
SAMPLES_TO_SHOW = 5


# Data
BASE_PATH = args.dataset #"../../mfccs/"
DATASET_LIST_PATH = BASE_PATH + args.dataset_list #BASE_PATH + "files.txt"
INPUT_DATA_DIR = BASE_PATH + args.input_mfccs #BASE_PATH + "img/"
OUTPUT_DIR = args.output_dir #"./dcgan_mfccs_output/" #"./dcgan_mfccs_output/test/"
#MODEL_PATH = BASE_PATH + "models/" + "model_" + str(EPOCH) + ".ckpt"
DATASET = [INPUT_DATA_DIR + str(line).rstrip() for line in open(DATASET_LIST_PATH,"r")]
DATASET_SIZE = len(DATASET)
MINIBATCH_SIZE = DATASET_SIZE // BATCH_SIZE

if args.train:
    # python train_dcgan.py --dataset ../../mfccs/ --dataset_list files.txt --input_mfccs img/ --output_dir ./dcgan_mfccs_output/test_output/ --train
    print("Training")
    # Training
    with tf.Graph().as_default():
        train(data_shape=(DATASET_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, 3),
              epoch=EPOCH,
              checkpoint_path=None)
elif args.detect:
    # python train_dcgan.py --dataset ../../mfccs/ --dataset_list good.txt --input_mfccs test/ --output_dir ./dcgan_mfccs_output/test_output/ --detect --checkpoint ./dcgan_mfccs_output/model_749.ckpt
    if args.checkpoint:
        # Prediction
        with tf.Graph().as_default():
            train(data_shape=(DATASET_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, 3),
                epoch = 1,
                checkpoint_path = args.checkpoint) #"./dcgan_mfccs_output/model_749.ckpt")
                #this results in several mfccs in the output directory
    else:
        print("There was no checkpoint provided.")
        sys.exit()
else:
    print("Please indicate what do you want this script to do:\n to train a model, use the parameter --train\n to create MFCCs for anomaly detection, use the parameter --detect")
    sys.exit()


"""
TEST_DATASET_LIST_PATH = "../../mfccs/bad.txt"
TEST_INPUT_DATA_DIR = "../../mfccs/test"
OUTPUT_DIR = "./dcgan_mfccs_output/test/bad/"
DATASET = [TEST_INPUT_DATA_DIR + str(line).rstrip() for line in open(TEST_DATASET_LIST_PATH,"r")]
DATASET_SIZE = len(DATASET)
MINIBATCH_SIZE = DATASET_SIZE // BATCH_SIZE
# Prediction
with tf.Graph().as_default():
    train(data_shape=(DATASET_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, 3),
        epoch = 1,
        checkpoint_path="./dcgan_mfccs_output/model_749.ckpt")
"""
"""
TEST_DATASET_LIST_PATH = "../../mfccs/defect.txt"
TEST_INPUT_DATA_DIR = "../../mfccs/test"
OUTPUT_DIR = "./dcgan_mfccs_output/test/defect/"
DATASET = [TEST_INPUT_DATA_DIR + str(line).rstrip() for line in open(TEST_DATASET_LIST_PATH,"r")]
DATASET_SIZE = len(DATASET)
MINIBATCH_SIZE = DATASET_SIZE // BATCH_SIZE
# Prediction
with tf.Graph().as_default():
    train(data_shape=(DATASET_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, 3),
        epoch = 1,
        checkpoint_path="./dcgan_mfccs_output/model_749.ckpt")
"""
"""
TEST_DATASET_LIST_PATH = "../../mfccs/good.txt"
TEST_INPUT_DATA_DIR = "../../mfccs/test"
OUTPUT_DIR = "./dcgan_mfccs_output/test/good/"
DATASET = [TEST_INPUT_DATA_DIR + str(line).rstrip() for line in open(TEST_DATASET_LIST_PATH,"r")]
DATASET_SIZE = len(DATASET)
MINIBATCH_SIZE = DATASET_SIZE // BATCH_SIZE
# Prediction
with tf.Graph().as_default():
    train(data_shape=(DATASET_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, 3),
        epoch = 1,
        checkpoint_path="./dcgan_mfccs_output/model_749.ckpt")
        #this results in several mfccs in the output directory
"""
