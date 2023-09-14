## 1. 라이브러리 및 데이터셋 불러오기
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# CIFAR-10 데이터셋 불러오기
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

## 2. CNN 모델 구성하기
train_images, test_images = train_images / 255.0, test_images / 255.0
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

## 3. Model Compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

## 4. Model 학습


history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
## 5. 성능 평가 
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

model.save("tuto_model1")
model.save("tuto_model1.h5")

print("\nTest accuracy:", test_acc)

