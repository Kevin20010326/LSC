from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import  array_to_img
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose

# 步驟 1：加載資料
# 資料集路徑
data_path = "/Users/wenqingwei/Desktop/LSC/A1_test/A1"

# 加載圖像和標籤
def load_data(data_path):
    images = []
    labels = []
    for filename in os.listdir(data_path):
        if filename.endswith("_rgb.png"):
            # 加载图像
            image = Image.open(os.path.join(data_path, filename))
            # 加载对应的标签
            label_filename = filename.replace("_rgb.png", "_label.png")
            label = Image.open(os.path.join(data_path, label_filename))
            # 转换为灰度图像，即单通道图像
            label = label.convert('L')
            images.append(np.array(image))
            labels.append(np.array(label))
            print("图像:", filename, "形状:", np.array(image).shape)
            print("标签:", label_filename, "形状:", np.array(label).shape)
    return np.array(images), np.array(labels)

# 加载数据
data_path = "/Users/wenqingwei/Desktop/LSC/A1_test/A1"
images, labels = load_data(data_path)

print("圖像數量:", len(images))
print("標籤數量:", len(labels))
print("第一张图像维度:", images[0].shape)
print("第一张标签维度:", labels[0].shape)



# 步驟 2：資料預處理
def preprocess_data(images, labels, image_size):
    # Preprocess images
    images_resized = [Image.fromarray(image).resize(image_size) for image in images]
    images_resized = [np.array(image) for image in images_resized]

    # Preprocess labels
    labels_resized = [np.expand_dims(label, axis=-1) for label in labels]

    # Convert labels to images
    labels_images = [array_to_img(label, scale=False) for label in labels_resized]

    return images_resized, labels_images

# Example usage
image_size = (224, 224)
X_train, X_test  = preprocess_data(images, labels, image_size)
print("训练集图像形状:", len(X_train))
print("测试集图像形状:", len(X_test))

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
    # 加载数据
    data_path = "/Users/wenqingwei/Desktop/LSC/A1_test"

    images, labels = load_data(data_path)
    print("图像数量:", len(images))
    print("标签数量:", len(labels))
    print("第一张图像维度:", images[0].shape)
    print("第一张标签维度:", labels[0].shape)

    # 数据预处理
    X_train, X_test, y_train, y_test = preprocess_data