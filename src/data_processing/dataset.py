import os
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms

class VimeoDataset(Dataset):
    def __init__(self, root_dir="data/processed_frames", train=True):
        self.root_dir = root_dir
        # Load triplet list (e.g., "00001/0001")
        split_file = os.path.join("data\\vimeo_triplet", "tri_trainlist.txt" if train else "tri_testlist.txt")
        with open(split_file, "r") as f:
            self.triplets = [line.strip() for line in f.readlines()]
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet_path = os.path.join(self.root_dir, self.triplets[idx])
        # Load frames (im1.png, im2.png, im3.png)
        frame1 = cv2.imread(os.path.join(triplet_path, "im1.png"))
        frame2 = cv2.imread(os.path.join(triplet_path, "im2.png"))  # Target frame
        frame3 = cv2.imread(os.path.join(triplet_path, "im3.png"))

        # Convert to tensors
        frame1_tensor = self.transform(frame1).float()
        frame3_tensor = self.transform(frame3).float()
        target_tensor = self.transform(frame2).float()

        return frame1_tensor, frame3_tensor, target_tensor