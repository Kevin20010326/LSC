from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose

# 步驟 1：加載資料
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
print("第一张图像维度:", images[0].shape)
print("第一张标签维度:", labels[0].shape)



# 步驟 2：資料預處理
def preprocess_data(images, labels):
    # 將圖像和標籤調整為模型所需的大小（256x256）
    image_size = (256, 256)
    images_resized = [array_to_img(img, scale=False).resize(image_size) for img in images]
    labels_resized = [array_to_img(label, scale=False).resize(image_size) for label in labels]

    # 正規化處理
    X = np.array([img_to_array(img) / 255.0 for img in images_resized])
    y = np.array([img_to_array(label) / 255.0 for label in labels_resized])

    # 分割資料集為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = preprocess_data(images, labels)
print("训练集图像形状:", X_train.shape)
print("测试集图像形状:", X_test.shape)

# 步驟 3：建立和訓練模型
def unet_model(input_shape):
    inputs = Input(input_shape)
    inputs = Input(input_shape)
    
    # Contracting Path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Expansive Path
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
    up6 = concatenate([up6, drop4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def train_model(model, X_train, y_train, X_test, y_test):
    # 編譯模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 訓練模型
    model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# 步驟 4：評估模型
def evaluate_model(model, X_test, y_test):
    # 評估模型
    loss, accuracy = model.evaluate(X_test, y_test)
    print("測試集損失:", loss)
    print("測試集準確率:", accuracy)

# 主函數
def main():
    # 資料集路徑
    data_path = "/Users/wenqingwei/Desktop/LSC/A1_test/A1"

    # 步驟 1：加載資料
    images, labels = load_data(data_path)
    print("圖像數量:", len(images))
    print("標籤數量:", len(labels))

    # 步驟 2：資料預處理
    X_train, X_test, y_train, y_test = preprocess_data(images, labels)
    print("訓練集圖像形狀:", X_train.shape)
    print("測試集圖像形狀:", X_test.shape)

    # 步驟 3：建立和訓練模型
    input_shape = (256, 256, 3)  # 圖像尺寸為 256x256，3通道（RGB）
    model = unet_model(input_shape)
    train_model(model, X_train, y_train, X_test, y_test)

    # 步驟 4：評估模型
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
