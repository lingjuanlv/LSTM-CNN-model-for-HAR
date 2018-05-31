#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : HAR.py
from __future__ import print_function
import os
import logging
import random

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import models_lstm_cnn
import socket, pickle

FORMAT = "%(levelname)s:%(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

tf.app.flags.DEFINE_string("data_dir", "data", "directory to dataset")
tf.app.flags.DEFINE_string("data_type", "HAR", "dataset type")
# tf.app.flags.DEFINE_string("data_type", "MH", "dataset type")
tf.app.flags.DEFINE_integer("hidden", 28, "Hidden units for RNN")
tf.app.flags.DEFINE_integer("num_layers", 1, "Layers for RNN")
tf.app.flags.DEFINE_integer("epochs", 100, "How many epochs for training")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training")
tf.app.flags.DEFINE_float("lr", 0.001, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 0.8, "Dropout percent")
tf.app.flags.DEFINE_boolean("bidir", False, "boolean; if set use bidirectional rnn")
tf.app.flags.DEFINE_boolean("training", True, "boolean; if set train model")

FLAGS = tf.app.flags.FLAGS


def read_data(path):
    """ Read dataset from disk"""
    X_signals = []
    X_path = path[:-1]
    Y_path = path[-1]

    for signal_type_path in X_path:
        file = open(signal_type_path, 'rb')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.strip().split() for row in file
            ]]
        )
        file.close()

    with open(Y_path) as f:
        if FLAGS.data_type == "HAR":
            Y_labels = [int(row.strip())-1 for row in f]
        else:
            Y_labels = [row.strip() for row in f]

    return zip(np.transpose(np.array(X_signals), (1, 2, 0)), np.array(Y_labels, dtype=np.int32))

def create_model(session, classes, steps, features, train_dir):
    model = models_lstm_cnn.HAR(FLAGS.hidden,
                               FLAGS.num_layers,
                               [(30, 5), (40, 10), (50, 15), (60, 20)],
                               classes,
                               FLAGS.batch_size,
                               steps,
                               features,
                               FLAGS.lr,
                               bidir=FLAGS.bidir)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model

def train(data_path):
    test_losses = []
    dev_accuracies = []
    train_losses = []
    train_accuracies = []
    dataset = read_data(data_path)
    data_size = len(dataset)
    train_num = int(data_size * 0.8)
    training  = dataset[:train_num]
    val = dataset[train_num:]
    random.shuffle(training)

    steps = training[0][0].shape[0]
    features = training[0][0].shape[1]
    if FLAGS.data_type == "HAR":
        classes = 6
    else:
        classes = 13
    print("steps: {} features: {}".format(steps, features))
    step = 0
    batches = train_num // FLAGS.batch_size
    val_batches = len(val) // FLAGS.batch_size

    # Checkpoin save path
    if FLAGS.data_type == "HAR":
        train_dir = "HAR_lstmcnn_train_dir" 
    else:
        train_dir = "MH_lstmcnn_train_dir" 

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    checkpoint_path = os.path.join(train_dir, "training.ckpt")

    with tf.Session() as sess:
        model = create_model(sess, classes, steps, features, train_dir)
        for epoch in range(FLAGS.epochs):
            loss = 0.0
            accu = 0.0
            for batch_x, batch_y in model.get_batch(training, True):
                batch_loss, batch_accu = model.step(sess, batch_x, batch_y, FLAGS.dropout, True)
                loss += batch_loss / batches
                accu += batch_accu / batches
                step += 1

            logging.info("{:03d}/{:03d} loss: {:2f} accuracy: {:2f}".format(epoch+1, FLAGS.epochs,
                                                                            loss, accu))
            val_accu = 0.0
            for val_x, val_y in model.get_batch(val, True):
                val_accu += model.step(sess, val_x, val_y, 1.0 , False) / val_batches
            logging.info("    eval accuracy {:4f}".format(val_accu))

            train_accuracies.append(accu)
            dev_accuracies.append(val_accu)
        print("")
        print("best dev accuracy: {}".format(max(dev_accuracies)))
        print("")

def main(_):
    # Those are separate normalised input features for the neural network
    if FLAGS.data_type == "HAR":
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        DATASET_PATH = "data/"

        TRAIN = "train/"
        TEST = "test/"

        X_train_signals_paths = [
            DATASET_PATH + TRAIN + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
        ]
        X_train_signals_paths.append(os.path.join(DATASET_PATH+TRAIN, "y_train.txt"))
    else:
        INPUT_SIGNAL_TYPES = ["chest_acc_X", "chest_acc_Y", "chest_acc_Z",
                              "left_ankle_acc_X", "left_ankle_acc_Y", "left_ankle_acc_Z",
                              "left_ankle_gyro_X", "left_ankle_gyro_Y", "left_ankle_gyro_Z",
                              "left_ankle_mag_X", "left_ankle_mag_Y", "left_ankle_mag_Z",
                              "right_arm_acc_X", "right_arm_acc_Y", "right_arm_acc_Z",
                              "right_arm_gyro_X", "right_arm_gyro_Y", "right_arm_gyro_Z",
                              "right_arm_mag_X", "right_arm_mag_Y", "right_arm_mag_Z",
                             ]
        DATASET_PATH = "MHdataset/"

        X_train_signals_paths = [
            DATASET_PATH + signal + ".txt" for signal in INPUT_SIGNAL_TYPES
        ]
        X_train_signals_paths.append(os.path.join(DATASET_PATH, "y.txt"))

    print("\n" + "Dataset is now located at: " + DATASET_PATH)
    if FLAGS.training:
        train(X_train_signals_paths)
    else:
        pass

if __name__ == "__main__":
    tf.app.run()
