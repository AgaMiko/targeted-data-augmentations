import albumentations
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

from datasets.frame_transform import RandomFrameTransform
from datasets.hair_transform import RandomHairTransform


def get_transforms(image_size, im_dir=None, type_aug='frame', mask_dir=None, aug_p=1.0, rotate=True, mask_nr='random'):
    if type_aug in ["hair_short", "hair_medium","hair_dense", "ruler","hair_and_ruler"]:
        if mask_dir in [None, False, ""]:
            raise ValueError(
                "You did not provide mask_dir. Provide path to directory with hair or ruler masks.")
        transforms_train = albumentations.Compose([
            RandomHairTransform(im_dir=im_dir,
                                p=aug_p, 
                                mask_dir=mask_dir,
                                rotate=rotate, 
                                mask_nr=mask_nr),
            albumentations.Resize(image_size, image_size),
            albumentations.CenterCrop(image_size, image_size),
            albumentations.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif type_aug == "frame":
        if mask_dir in [None, False, ""]:
            raise ValueError(
                "You did not provide mask_dir. Provide path to directory with hair or ruler masks.")
        transforms_train = albumentations.Compose([
            RandomFrameTransform(
                p=aug_p, mask_dir=mask_dir, rotate=rotate, mask_nr=mask_nr),
            albumentations.Resize(image_size, image_size),
            albumentations.CenterCrop(image_size, image_size),
            albumentations.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif type_aug == "normal":
        transforms_train = albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            albumentations.CenterCrop(image_size, image_size),
            albumentations.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        raise ValueError
    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return transforms_train, transforms_val


def get_augmentation(transform):
    return lambda img: transform(image=np.array(img))
