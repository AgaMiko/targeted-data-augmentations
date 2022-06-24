from distutils.command.config import config
import torch
from torch.utils.data import DataLoader, sampler
from torchvision import datasets
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger
from efficientnet_pytorch import EfficientNet
from utils.sampler_utils import create_weights_vector
from models.model import LesionClassification
import yaml
import os
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config_path",
        type=str,
        default='config.yml',
        help="path to .yaml config file specifying hyperparameters of different model sections."
    )
    return parser


def main(args):
    with open(args.model_config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    for arch_nr, model_name in enumerate(config["training_plan"]["architectures"]["names"]):
        for mode in config["training_plan"]["augmentations"]:
            batch_size = config["training_plan"]["architectures"]["batch_sizes"][arch_nr]
            img_size = config["training_plan"]["architectures"]["input_sizes"][arch_nr]
            
            if mode == "normal":
                probabilities = [0.0]
            else:
                probabilities = config["training_plan"]["probabilities"]
                
            for p in probabilities:
                # prepare model for training
                model = LesionClassification(model=model_name,
                                            num_classes=2,
                                            lr=5e-4,
                                            lr_decay=0.9)
                model_checkpoint = ModelCheckpoint(monitor="val_loss",
                                                verbose=True,
                                                filename="{epoch}_{val_loss:.4f}")
                
                save_checkpoint_path = os.path.join(config["general"]["dir_to_save"],model_name+"_"+mode+"_"+str(p)+".ckpt")
                
                # load transforms
                if mode == "normal":
                    from datasets.transforms import get_transforms, get_augmentation
                    train_transform, test_transform = get_transforms(img_size)
                else:
                    from datasets.transforms_aug import get_transforms, get_augmentation
                    mask_dir = config["mask_dirs"]["train"][mode]
                    train_transform, test_transform = get_transforms(
                        img_size, type_aug=mode, mask_dir=mask_dir, aug_p=p)

                train_augmentation = get_augmentation(train_transform)

                # load images from folders
                train_set = datasets.ImageFolder(root=config["general"]["train_dir"],
                                                transform=train_augmentation)
                test_set = datasets.ImageFolder(root=config["general"]["test_dir"],
                                                transform=get_augmentation(test_transform))

                # prepare weighted sampler
                weights = create_weights_vector(train_set.targets,
                                                weights=[0.2, 0.8])
                weighted_sampler = sampler.WeightedRandomSampler(
                    weights, len(weights))

                # prepare dataloaders with samplers
                train_loader = DataLoader(train_set, batch_size=batch_size,
                                        sampler=weighted_sampler, num_workers=config["general"]["num_workers"])
                test_loader = DataLoader(
                    test_set, batch_size=batch_size, num_workers=config["general"]["num_workers"])

                neptune_tags = [model_name, mode, str(p)]
                logger = NeptuneLogger(project_name=config["general"]["nepune_project_name"],
                                    tags=neptune_tags)

                # prepare trained
                trainer = pl.Trainer(gpus=config["general"]["gpu"],
                                    max_epochs=config["general"]["epochs"],
                                    callbacks=[model_checkpoint],
                                    logger=logger,
                                    log_every_n_steps=5,
                                    accumulate_grad_batches=config["general"]["accumulate_grad_batches"],
                                    fast_dev_run=True)
                trainer.fit(model, train_loader, test_loader)

                trainer.save_checkpoint(save_checkpoint_path)
    
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
    