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

# 獲取資料集的類別數量
num_classes = len(dataset.classes)

# 初始化模型
model = LeafSegmentationCNN()

# 創建訓練數據加載器
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")
