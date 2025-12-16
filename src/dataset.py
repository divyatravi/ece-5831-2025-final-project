import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FolderDeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: path to test_images (which has real/ and fake/ subfolders)
        """
        self.samples = []  # list of (image_path, label)

        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        for fname in os.listdir(real_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                self.samples.append((os.path.join(real_dir, fname), 0))  # 0 = real

        for fname in os.listdir(fake_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                self.samples.append((os.path.join(fake_dir, fname), 1))  # 1 = fake

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # same as training
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
