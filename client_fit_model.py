from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from keras.preprocessing.image import load_img
from keras import layers
from datetime import datetime
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import random


class Generator(keras.utils.Sequence):

    def __init__(self, input_img_paths, target_img_paths, batch_size, img_size, ):
        self.x = input_img_paths
        self.y = target_img_paths
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return len(self.y) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = np.array(
            [cv2.resize(cv2.cvtColor(cv2.imread(file_name, -1), cv2.COLOR_BGR2RGB),
                        (self.img_size[1], self.img_size[0]))
             for file_name in batch_x])
        batch_y = np.array(
            [(cv2.resize(cv2.imread(file_name, -1), (self.img_size[1], self.img_size[0])) > 0).astype(np.uint8) for
             file_name in batch_y])
        batch_y = np.expand_dims(batch_y, -1)

        return batch_x / 255, batch_y / 1


class learning_fit(object):
    def __init__(self, mt, ec, bs, pr, cr):
        self.model_type = mt
        self.epochs = ec
        self.batch_size = bs
        self.params = pickle.loads(pr)
        self.round = cr

    def gen_train_val_data(self):
        img_size = (128, 128)
        batch_size = 16

        train_image_dir = "/home/mhk/PycharmProjects/FederatedLearning-gRPC-seg-main/DataSet/Train/Positive"
        train_mask_dir = "/home/mhk/PycharmProjects/FederatedLearning-gRPC-seg-main/DataSet/Train/masks"

        train_image_paths = sorted(
            [
                os.path.join(train_image_dir, fname)
                for fname in os.listdir(train_image_dir)
                if fname.endswith(".jpg")
            ]
        )
        train_mask_paths = sorted(
            [
                os.path.join(train_mask_dir, fname)
                for fname in os.listdir(train_mask_dir)
                if fname.endswith(".jpg") and not fname.startswith(".")
            ]
        )

        val_samples = 6213
        random.Random(1337).shuffle(train_image_paths)
        random.Random(1337).shuffle(train_mask_paths)
        train_input_img_paths = train_image_paths[:val_samples]
        train_target_img_paths = train_mask_paths[:val_samples]
        val_input_img_paths = train_image_paths[val_samples:]
        val_target_img_paths = train_mask_paths[val_samples:]

        train_data_gen = Generator(
            train_input_img_paths, train_target_img_paths, batch_size, img_size
        )

        val_data_gen = Generator(val_input_img_paths, val_target_img_paths, batch_size, img_size)

        return train_data_gen, val_data_gen

    def change_model_layers(self):
        # Unet
        img_size = (128, 128)
        inputs = keras.Input(shape=img_size + (3,))

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)

        # Define the model
        model = keras.Model(inputs, outputs)

        return model

    def train_model_tosave(self, params):
        logdir = f"send_logs/logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.round}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
        local_model = self.change_model_layers()

        local_model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=['accuracy'])

        if params != None:
            local_model.set_weights(params)
        else:
            local_model.set_weights(self.params)

        train_gen, val_gen = self.gen_train_val_data()

        local_model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=tensorboard_callback)

        print("### predict start ###")

        # Export model
        # export_path = "?"
        # local_model.save(export_path, save_format="tf")

        return local_model

    def Predict(self, model):

        print("### predict start ###")

        img_size = (128, 128)
        batch_size = 16

        predict_image_dir = "/home/mhk/PycharmProjects/FederatedLearning-gRPC-seg-main/PredictData/Predict"

        none = '/home/mhk/PycharmProjects/FederatedLearning-gRPC-seg-main/PredictData/none/noncrack_noncrack_concrete_wall_41_57.jpg.jpg'

        train_image_paths = sorted(
            [
                os.path.join(predict_image_dir, fname)
                for fname in os.listdir(predict_image_dir)
                if fname.endswith(".jpg")
            ]
        )

        idx = 0
        amount = 0
        list = []

        none = []

        for idx in range(0, len(train_image_paths)):
            none.append(
                '/home/mhk/PycharmProjects/FederatedLearning-gRPC-seg-main/PredictData/none/noncrack_noncrack_concrete_wall_41_57.jpg.jpg')

        predict_image_gen = Generator(train_image_paths, none, batch_size, img_size)

        val_preds = model.predict(predict_image_gen)

        for prediction in val_preds:
            pred = np.dstack([prediction, prediction, prediction])
            pred = (pred * 255).astype(np.uint8)
            cv2.imwrite('round{}.png'.format(idx), pred)
            img = cv2.imread('round{}.png'.format(idx))

            area, round2, length, mount = self.contour(img)
            amount += mount
            list.append([])
            list[idx].append(area)
            list[idx].append(round2)
            list[idx].append(length)
            idx += 1

        return list, amount

    def manage_train(self, params=None, cr=None):  # cr:current_round
        print(f"### Model Training - Round: {cr} ###")
        if self.params == list():
            return []

        if params is not None:
            params = pickle.loads(params)
        lmodel = self.train_model_tosave(params)
        params = lmodel.get_weights()

        if cr == 5:
            self.Predict(lmodel)
            print(" predict done ")
        print("### Save model weight to ./saved_weight/ ###")
        with open('./saved_weight/weights.pickle', 'wb') as fw:
            pickle.dump(params, fw)
