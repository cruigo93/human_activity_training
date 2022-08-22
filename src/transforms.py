import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor


def get_train_transforms():
    train_transforms = albu.Compose([
        albu.Resize(height=256, width=256),
        albu.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ])
    return train_transforms
