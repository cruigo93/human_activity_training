import torch
import albumentations as albu
import cv2
import numpy as np
from sklearn.preprocessing import label_binarize

from typing import List, Dict
import os


CLASSES = ['listening_to_music', 'sitting', 'sleeping', 'eating', 'drinking', 'fighting', 'clapping',
           'using_laptop', 'running', 'cycling', 'dancing', 'texting', 'laughing', 'hugging', 'calling']


class HumanDataset:
    def __init__(self, image_dir: str, items: List[Dict], transforms: albu.transforms, is_testing: bool = False) -> None:
        self.image_dir = image_dir
        self.items = items
        self.transforms = transforms
        self.is_testing = is_testing

    def __getitem__(self, idx):
        item = self.items[idx]
        image = cv2.imread(os.path.join(self.image_dir, item["filename"]), 1)
        image = self.transforms(image=image)["image"]
        # label = torch.tensor(np.array([item["label"]]))
        if self.is_testing:
            label = CLASSES.index(item["label"])
            label_encoding = [0 for _ in range(len(CLASSES))]
            label_encoding[label] = 1
            label = torch.tensor(label_encoding).float()

            return {
                "image": image,
                "label": label
            }
        return {"image": image}

    def __len__(self):
        return len(self.items)
