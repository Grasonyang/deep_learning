import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 載入模型
try:
    model = tf.keras.models.load_model('.model/mnist_model.h5')
except (IOError, ImportError) as e:
    print("錯誤：無法載入模型。請先執行 2_trainModel.py 進行訓練。")
    print(e)
    exit()

# 圖片所在資料夾
image_folder = '.dataset/.images'

# 檢查圖片資料夾是否存在
if not os.path.exists(image_folder) or not os.listdir(image_folder):
    print(f"錯誤：找不到圖片資料夾 '{image_folder}' 或資料夾為空。")
    print("請將您手寫的學號數字圖片 (例如 1.png, 2.png, ...) 放入此資料夾中。")
    exit()

# 依檔名排序讀取所有圖片
try:
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith('.png')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
except ValueError:
     print(f"錯誤：圖片檔名無法轉換為數字排序。請確保檔名為 '1.png', '2.png' 等格式。")
     exit()


print("辨識您的學號數字：")

result = []
for image_file in image_files:
    try:
        # 載入圖片並轉換為灰階
        img_path = os.path.join(image_folder, image_file)
        img = Image.open(img_path).convert('L')

        # 調整圖片大小並轉換為 numpy array
        img = img.resize((28, 28))
        img_array = np.array(img)

        # 圖片預處理：反轉顏色 (黑白互換) 並正規化
        # MNIST 資料是白字黑底，一般手寫是黑字白底，所以需要反轉
        img_array = 255 - img_array
        img_array = img_array / 255.0

        # 擴展維度以符合 CNN 模型輸入 (1, 28, 28, 1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        # 進行預測
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions)
        result.append(str(predicted_label))
        print(f"  檔案 '{image_file}': 辨識結果為 -> {predicted_label}")

    except FileNotFoundError:
        print(f"  檔案 '{image_file}' 不存在。")
    except Exception as e:
        print(f"  處理檔案 '{image_file}' 時發生錯誤: {e}")

print("學號為: S" + ''.join(result))
