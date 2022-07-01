import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from datasets.hair_transform import RandomHairTransform
from datasets.frame_transform import RandomFrameTransform
     
def get_transforms(image_size):

    transforms_train = albumentations.Compose([
        # albumentations.Transpose(p=0.5),
        # albumentations.VerticalFlip(p=1),
        # albumentations.HorizontalFlip(p=0.5),
        RandomFrameTransform(p=0),
        # albumentations.RandomBrightness(limit=0.2, p=1),
        # albumentations.RandomContrast(limit=0.2, p=1),
        # albumentations.OneOf([
        #     albumentations.MotionBlur(blur_limit=3),
        #     albumentations.MedianBlur(blur_limit=3),
        #     albumentations.GaussianBlur(blur_limit=3),
        #     albumentations.GaussNoise(var_limit=(1.0, 10.0)),
        # ], p=1),

        # albumentations.OneOf([
        #     albumentations.OpticalDistortion(distort_limit=1.0),
        #     albumentations.GridDistortion(num_steps=5, distort_limit=1.),
        #     albumentations.ElasticTransform(alpha=3),
        # ], p=1),
        # albumentations.CLAHE(clip_limit=4.0, p=0.7),
        # albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        # albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, border_mode=0, p=0.85),
        # albumentations.Resize(image_size, image_size),
        # albumentations.Cutout(max_h_size=int(image_size * 0.25), max_w_size=int(image_size * 0.25), num_holes=1, p=0.1),
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(),
        ToTensorV2()
    ])
    transforms_val = albumentations.Compose([
        # RandomHairTransform(p=1),
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(),
        ToTensorV2()
    ])

    return transforms_train, transforms_val
         
def get_augmentation(transform):
    return lambda img:transform(image=np.array(img))
