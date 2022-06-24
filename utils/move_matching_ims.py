#!/usr/bin/python
from PIL import Image
import os, sys
from shutil import copyfile

source_name = "/media/agnieszka/Data/data/skin-lesion/aug/mask_realhair/"
source_imgs = "/media/agnieszka/Data/data/siim-isic/jpeg-isic2019-768x768/malignant/"
dest = "/media/agnieszka/Data/data/skin-lesion/aug/im_realhair/"
dest2 = "/media/agnieszka/Data/data/skin-lesion/aug/mask_realhair_matching/"
dirs = os.listdir(source_name)
dirs_imgs = os.listdir(source_imgs)
for item in dirs:
    if os.path.isfile(source_name+item):
        for image in dirs_imgs:
            if item == image:
                copyfile(source_imgs + item, dest + item)
                copyfile(source_name + item, dest2 + item)