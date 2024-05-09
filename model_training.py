import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data_preprocessing import LeafSegmentationCNN

# 定義數據轉換
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 將圖像調整為 64x64
    transforms.ToTensor(),         # 將圖像轉換為 PyTorch 張量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化到[-1, 1]範圍內
])


# 加載數據集
dataset = ImageFolder(root="/Users/wenqingwei/Desktop/LSC/A1_test", transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 初始化模型
model = LeafSegmentationCNN()

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss() # type: ignore
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
for epoch in range(10):  # 假設進行 10 個 epoch 的訓練
    for images, labels in train_loader:
        optimizer.zero_grad()  # 梯度歸零
        outputs = model(images)  # 前向傳播
        loss = criterion(outputs, labels)  # 計算損失
        loss.backward()  # 反向傳播
        optimizer.step()  # 更新權重
    print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")
