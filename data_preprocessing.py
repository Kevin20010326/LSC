import torch.nn as nn
import torch.nn.functional as F

class LeafSegmentationCNN(nn.Module):
    def __init__(self):
        super(LeafSegmentationCNN, self).__init__()
        # 定義卷積層
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # 定義全連接層
        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 2)  # 輸出 2 類：前景和背景

    def forward(self, x):
        # 向前傳遞
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 64 * 64)  # 將特徵圖展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 接下來，將你之前的程式碼添加在這之後
