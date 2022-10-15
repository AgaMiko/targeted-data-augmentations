import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import random
import glob
import numpy

class RandomHairTransform:
    """Rotate by one of the given angles."""

    def __init__(self, mask_dir="/media/agnieszka/Data/data/skin-lesion/aug/mask_merged/",
                im_dir="/media/agnieszka/Data/data/skin-lesion/aug/source_merged/",
                image_type = "jpg",
                p=0.5):
        self.mask_list =  glob.glob(mask_dir + "*."+ image_type)
        self.im_dir = im_dir
        self.len = len(self.mask_list)
        self.p = float(p)
    
    def __call__(self, image, **params):
        p = random.randint(0, 100)
        if p <= self.p*100:
            source_image = Image.fromarray(image)
            angle = random.randint(0, 360)
            hair_nr =  random.randint(0, self.len-1)
            im_path = self.im_dir + self.mask_list[hair_nr].split("/")[-1]
            mask = Image.open(self.mask_list[hair_nr]).convert('L')
            im = Image.open(im_path).convert('RGB')
            im = im.rotate(angle=angle)
            mask = mask.rotate(angle=angle)
            temp_image = Image.composite(im, source_image, mask)
            return {'image': numpy.array(temp_image,dtype=numpy.dtype("uint8"))}
        else:
            return {'image': image}
    

    