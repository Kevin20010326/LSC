import torchvision.transforms as transforms # type: ignore
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 定義數據轉換
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 調整大小為 64x64
    transforms.ToTensor()  # 轉換為 PyTorch 張量
])

# 加載數據集
dataset = ImageFolder(root="A1_dataset_path", transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型
model = LeafSegmentationCNN()

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
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
