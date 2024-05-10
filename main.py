import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import LeafDataset
from model import UNet

# 定義資料轉換
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),  # 將RGB圖像轉換為單通道灰度圖像
    transforms.Normalize((0.5,), (0.5,))
])

# 載入訓練和測試資料
train_dataset = LeafDataset(data_dir='/Users/wenqingwei/Desktop/LSC/A1_test', transform=transform)
test_dataset = LeafDataset(data_dir='/Users/wenqingwei/Desktop/LSC/A1_test', transform=transform)


# 定義資料載入器
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1, out_channels=1).to(device)  # 輸入和輸出通道數皆為1

# 定義損失函數和優化器
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 評估模型
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        test_loss += loss.item() * images.size(0)
test_loss /= len(test_dataset)
print(f'Test Loss: {test_loss:.4f}')
