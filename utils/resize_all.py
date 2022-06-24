#!/usr/bin/python
from PIL import Image
import os, sys

source = "/media/agnieszka/Data/data/HR/hair_masks/blurred_ramella/"
dest = "/media/agnieszka/Data/data/skin-lesion/aug/mask/"
dirs = os.listdir(source)

def resize():
    for item in dirs:
        if os.path.isfile(source+item):
            im = Image.open(source+item)
            f, e = os.path.splitext(dest+item)
            imResize = im.resize((768,768), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)

resize()