from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random

class MyDataset(Dataset):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", "bmp"]
    def __init__(self, img_dir, transform_input=None, transform_target=None):
        img_dir = Path(img_dir)
        self.img_paths = [
            p for p in img_dir.iterdir() if p.suffix in MyDataset.IMG_EXTENSIONS
        ]
        self.transform_input = transform_input
        self.transform_target = transform_target
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        h = random.random()
        v = random.random()
        if self.transform_input is not None:
            input = self.transform_input(img)
        if self.transform_target is not None:
            target = self.transform_target(img)
        if h > 0.5:
            input = transforms.RandomHorizontalFlip(p=1.0)(input)
            target = transforms.RandomHorizontalFlip(p=1.0)(target)
        if v > 0.5:
            input = transforms.RandomVerticalFlip(p=1.0)(input)
            target = transforms.RandomVerticalFlip(p=1.0)(target)
        return input, target

if __name__ == "__main__":
    transform_input = transforms.Compose([
        transforms.Resize((540,960)),
        transforms.ToTensor()
        ])
    transform_target = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = MyDataset("./video", transform_input, transform_target)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for inputs, targets in dataloader:
        img = targets[0].to('cpu').detach().numpy().transpose(1,2,0)
        img = cv2.resize(img, (1024, 540))
        plt.imshow(img)
        plt.show()
        print(inputs.shape, targets.shape)