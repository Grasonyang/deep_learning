import tensorflow as tf
import numpy as np
import os

# 建立 .dataset 資料夾 (如果不存在)
if not os.path.exists('.dataset'):
    os.makedirs('.dataset')

# 載入 MNIST 資料集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 將資料儲存為 .npy 檔案
np.save('.dataset/x_train.npy', x_train)
np.save('.dataset/y_train.npy', y_train)
np.save('.dataset/x_test.npy', x_test)
np.save('.dataset/y_test.npy', y_test)

print("MNIST 資料集已成功下載並儲存至 .dataset 資料夾中。")
print(f"訓練資料維度: {x_train.shape}")
print(f"訓練標籤維度: {y_train.shape}")
print(f"測試資料維度: {x_test.shape}")
print(f"測試標籤維度: {y_test.shape}")
