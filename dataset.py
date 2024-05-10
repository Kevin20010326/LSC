import os
from PIL import Image
from torch.utils.data import Dataset

class LeafDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images_dir = os.path.join(data_dir, 'A1')
        self.masks_dir = os.path.join(data_dir, 'A1')
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('_rgb.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = img_name.replace('_rgb.png', '_label.png')
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
