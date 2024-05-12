from PIL import Image
import numpy as np
import os

# 資料集路徑
data_path = "/Users/wenqingwei/Desktop/LSC/A1_test/A1"

# 加載圖像和標籤
def load_data(data_path):
    images = []
    labels = []
    for filename in os.listdir(data_path):
        if filename.endswith("_rgb.png"):
            # 加載圖像
            image = Image.open(os.path.join(data_path, filename))
            # 加載對應的標籤
            label_filename = filename.replace("_rgb.png", "_label.png")
            label = Image.open(os.path.join(data_path, label_filename))
            images.append(np.array(image))
            labels.append(np.array(label))
    return np.array(images), np.array(labels)

# 加載資料
images, labels = load_data(data_path)

print("圖像數量:", len(images))
print("標籤數量:", len(labels))
