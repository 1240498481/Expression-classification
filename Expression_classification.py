# -*- coding:utf-8 -*-
"""
 @Author   : Kai
 @Time     : 2021/6/4 18:25
 @Email    : 1240498481@qq.com
 @FileName : fenlei.py
 @Software : PyCharm

 实现表情分类
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib


# 指定数据集目录
data_dir = pathlib.Path("./datas/ck+/")

# 查看图片数量
image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)


# 查看数据
roses = list(data_dir.glob('anger/*'))
PIL.Image.open(str(roses[0]))


# 查看图片大小
PIL.Image.open(str(roses[0])).size


# 为加载器定义一些参数
batch_size = 128
img_height = 48
img_width = 48


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


class_names = train_ds.class_names
print(class_names)


# 显示数据集中的九张图片
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(images[i].numpy().astype('uint8'))
    plt.title(class_names[labels[i]])
    plt.axis('off')


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)


normalized_ds = train_ds.map(lambda x,y : (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))


num_classes = 7

model = Sequential([
   layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
   layers.Conv2D(16, 3, padding='same', activation='relu'),
   layers.MaxPooling2D(),
   layers.Conv2D(32, 3, padding='same', activation='relu'),
   layers.MaxPooling2D(),
   layers.Conv2D(64, 3, padding='same', activation='relu'),
   layers.MaxPooling2D(),
   layers.Flatten(),
   layers.Dense(128, activation='relu'),
   layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])


model.summary()


epochs = 25
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=epochs
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


model.save('./models/Expression classification.h5')
model = keras.models.load_model('./models/Expression classification.h5')


file_path = './datas/ck+/anger/S010_004_00000017.png'
img = keras.preprocessing.image.load_img(file_path)
x = keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
scores = model.predict(x)[0]


for i in range(len(class_names)):
  print(f"{scores[i] * 100} is {class_names[i]}")


