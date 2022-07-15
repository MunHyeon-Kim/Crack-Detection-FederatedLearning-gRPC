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



model = keras.models.load_model("my_model")



val_gen = Generator(val_input_img_paths, val_target_img_paths, batch_size, img_size)
val_preds = model.predict(val_gen)

prediction = val_preds[1]




pred = np.dstack([prediction, prediction, prediction])
pred = (pred * 255).astype(np.uint8)

imz = cv2.imwrite('pred2.png', pred)


img = cv2.imread('pred2.png')

def contour(img):
    img2 = img.copy()
    img3 = img.copy()

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    for cnt in contours:
        epsilon1 = 0.01 * cv2.arcLength(cnt, True)
        epsilon2 = 0.1 * cv2.arcLength(cnt, True)
        approx1 = cv2.approxPolyDP(cnt, epsilon1, True)
        approx2 = cv2.approxPolyDP(cnt, epsilon2, True)

        cv2.drawContours(img, [cnt], 0, (0, 0, 255), 1)
        cv2.drawContours(img2, [approx1], 0, (0, 0, 255), 1)
        cv2.drawContours(img3, [approx2], 0, (0, 0, 255), 1)

    print(' contour 면적 : ', area)
    print(' contour 길이 : ', perimeter)

    cv2.imwrite('contour/img1.jpg', img)
    cv2.imwrite('contour/img2.jpg', img2)
    cv2.imwrite('contour/img3.jpg', img3)


contour(img)
