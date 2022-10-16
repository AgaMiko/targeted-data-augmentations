import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from datasets.hair_transform import RandomHairTransform
from datasets.frame_transform import RandomFrameTransform
     
def get_transforms(image_size):

    transforms_train = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(),
        ToTensorV2()
    ])
    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(),
        ToTensorV2()
    ])

    return transforms_train, transforms_val
         
def get_augmentation(transform):
    return lambda img:transform(image=np.array(img))
