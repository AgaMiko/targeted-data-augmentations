import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from datasets.hair_transform import RandomHairTransform
from datasets.frame_transform import RandomFrameTransform


def get_transforms(image_size, type_aug='frame', mask_dir = None, aug_p=1.0):
    if type_aug == "hair_and_ruler":
        if mask_dir in [None, False, ""]:
            raise ValueError("You did not provide mask_dir. Provide path to directory with hair or ruler masks.")
        transforms_train = albumentations.Compose([
            RandomHairTransform(
                p=aug_p, mask_dir=mask_dir),
            albumentations.Resize(image_size, image_size),
            albumentations.CenterCrop(image_size, image_size),
            albumentations.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif type_aug == "short":
        if mask_dir in [None, False, ""]:
            raise ValueError("You did not provide mask_dir. Provide path to directory with hair or ruler masks.")
        transforms_train = albumentations.Compose([
            RandomHairTransform(
                p=aug_p, mask_dir=mask_dir),
            albumentations.Resize(image_size, image_size),
            albumentations.CenterCrop(image_size, image_size),
            albumentations.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif type_aug == "medium":
        if mask_dir in [None, False, ""]:
            raise ValueError("You did not provide mask_dir. Provide path to directory with hair or ruler masks.")
        transforms_train = albumentations.Compose([
            RandomHairTransform(
                p=aug_p, mask_dir=mask_dir),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif type_aug == "dense":
        if mask_dir in [None, False, ""]:
            raise ValueError("You did not provide mask_dir. Provide path to directory with hair or ruler masks.")
        transforms_train = albumentations.Compose([
            RandomHairTransform(
                p=aug_p, mask_dir=mask_dir),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif type_aug == "ruler":
        if mask_dir in [None, False, ""]:
            raise ValueError("You did not provide mask_dir. Provide path to directory with hair or ruler masks.")
        transforms_train = albumentations.Compose([
            RandomHairTransform(
                p=aug_p,mask_dir=mask_dir),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif type_aug == "frame":
        if mask_dir in [None, False, ""]:
            raise ValueError("You did not provide mask_dir. Provide path to directory with hair or ruler masks.")
        transforms_train = albumentations.Compose([
            RandomFrameTransform(
                p=aug_p, mask_dir=mask_dir),
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
    return lambda img:transform(image=np.array(img))

