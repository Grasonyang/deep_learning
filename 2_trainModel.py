import tensorflow as tf
import numpy as np
import os

# 建立 .model 資料夾 (如果不存在)
if not os.path.exists('.model'):
    os.makedirs('.model')

# 載入資料
x_train = np.load('.dataset/x_train.npy')
y_train = np.load('.dataset/y_train.npy')
x_test = np.load('.dataset/x_test.npy')
y_test = np.load('.dataset/y_test.npy')

# 資料預處理
# 將像素值正規化到 0-1 之間
x_train = x_train / 255.0
x_test = x_test / 255.0

# 擴展維度以符合 CNN 輸入要求 (batch, height, width, channels)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 建立模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, epochs=10)

# 評估模型
print("\n評估模型準確率：")
model.evaluate(x_test, y_test, verbose=2)

# 儲存模型
model.save('.model/mnist_model.h5')

print("\n模型訓練完成並已儲存至 .model/mnist_model.h5")

