import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor


def get_train_transforms():
    train_transforms = albu.Compose([
        albu.PadIfNeeded(min_height=128, min_width=128, p=1),
        albu.Resize(height=128, width=128),
        albu.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ])
    return train_transforms
