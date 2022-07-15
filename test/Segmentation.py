import os
import cv2
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 데이터 전처리
img_size = (128, 128)
num_classes = 3
batch_size = 16

train_image_dir = "crack_segmentation_dataset/train/images"
train_mask_dir = "crack_segmentation_dataset/train/masks"

test_image_dir = "crack_segmentation_dataset/test/images"
test_mask_dir = "crack_segmentation_dataset/test/masks"

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

test_image_paths = sorted(
    [
        os.path.join(test_image_dir, fname)
        for fname in os.listdir(train_image_dir)
        if fname.endswith(".jpg")
    ]
)
test_mask_paths = sorted(
    [
        os.path.join(test_mask_dir, fname)
        for fname in os.listdir(train_mask_dir)
        if fname.endswith(".jpg") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(train_image_paths))

for input_path, target_path in zip(train_image_paths[:10], train_mask_paths[:10]):
    print(input_path, "|", target_path)


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


# Split our img paths into a training and a validation set

val_samples = 6213
random.Random(1337).shuffle(train_image_paths)
random.Random(1337).shuffle(train_mask_paths)
train_input_img_paths = train_image_paths[:-val_samples]
train_target_img_paths = train_mask_paths[:-val_samples]
val_input_img_paths = train_image_paths[-val_samples:]
val_target_img_paths = train_mask_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = Generator(
    train_input_img_paths, train_target_img_paths, batch_size, img_size
)

val_gen = Generator(val_input_img_paths, val_target_img_paths, batch_size, img_size)


# unet model

def get_model(img_size, num_classes):
    # Unet

    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = tf.keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)

# 모델 학습

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.

# Loss Function


callbacks = [
    keras.callbacks.ModelCheckpoint("crack_segmentation.h5", save_best_only=True)
]

import tensorflow as tf

model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=['accuracy'])

epochs = 60
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
model.save("my_model")






